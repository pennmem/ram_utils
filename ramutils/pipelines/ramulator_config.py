import os.path

from bptools.jacksheet import read_jacksheet

from ramutils.parameters import StimParameters
from ramutils.tasks import *


def make_stim_params(subject, anodes, cathodes, root='/'):
    """Construct :class:`StimParameters` objects from anode and cathode labels
    for a specific subject.

    Parameters
    ----------
    subject : str
    anodes : List[str] anodes
        anode labels
    cathodes : List[str]
        cathode labels
    root : str
        root directory to search for jacksheet

    Returns
    -------
    stim_params : List[StimParams]

    """
    path = os.path.join(root, 'data', 'eeg', subject, 'docs', 'jacksheet.txt')
    jacksheet = read_jacksheet(path)

    stim_params = []

    for i in range(len(anodes)):
        anode = anodes[i]
        cathode = cathodes[i]
        anode_idx = jacksheet[jacksheet.label == anode].index[0]
        cathode_idx = jacksheet[jacksheet.label == cathode].index[0]
        stim_params.append(
            StimParameters(
                label='-'.join([anode, cathode]),
                anode=anode_idx,
                cathode=cathode_idx
            )
        )

    return stim_params


#FIXME: I don't like that make_ramulator_config has to know about the paths
# object. I'd prefer the paths object and this function to be decoupled similar
# to how this function no longer needs to know about the ExperimentParams object
def make_ramulator_config(subject, experiment, paths, anodes, cathodes,
                          vispath=None, **kwargs):
    """Generate configuration files for a Ramulator experiment.

    Parameters
    ----------
    subject : str
        Subject ID
    experiment : str
        Experiment to generate configuration file for
    paths : FilePaths
    anodes : List[str]
        List of stim anode contact labels
    cathodes : List[str]
        List of stim cathode contact labels
    vispath : str
        Path to save task graph visualization to if given.
    kwargs
        start_time: float
            Start of the period in the EEG to consider for each event
        end_time: float
            End of the period to consider
        buffer_time: float
            Buffer time
        freqs: array_like
            List of frequencies to use when applying Wavelet Filter
        log_powers: bool
            Whether to take the logarithm of the powers
        filt_order: Int
            Filter order to use in Butterworth filter
        width: Int
            Wavelet width to use in Wavelet Filter
        penalty_param: Float
            Penalty parameter to use
        penalty_type: str
            Type of penalty to use for regularized model (ex: L2)
        solver: str
            Solver to use when fitting the model (ex: liblinear)
        encoding_only: bool
            Indicator for if encoding-only classifier should be used, i.e. do
            not train on retrieval events
        encoding_multiplier: float
            Scaling factor for encoding events (required if using FR sample
            weighting schme)
        combine_events: bool
            For PAL experiments, indicates if record-only sessions should be
            combined, or if only PAL1 sessions should be used for training
        pal_mutiplier: float
            Scaling factor for PAL events (required if using PAL weighting
            scheme)
        scheme: str
            Sample weighting scheme to use (options: EQUAL, PAL, FR). See
            get_sample_weights for details

    Returns
    -------
    The path to the generated configuration zip file.

    """
    stim_params = make_stim_params(subject, anodes, cathodes, paths.root)

    ec_pairs = generate_pairs_from_electrode_config(subject, paths)
    excluded_pairs = reduce_pairs(ec_pairs, stim_params, True)
    used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)
    final_pairs = generate_pairs_for_classifier(ec_pairs, excluded_pairs)

    events = preprocess_events(subject,
                               experiment,
                               encoding_only=kwargs['encoding_only'],
                               combine_events=kwargs['combine_events'],
                               root=paths.root)

    # FIXME: If PTSA is updated to not remove events behind this scenes, this
    # won't be necessary. Or, if we can remove bad events before passing to
    # compute powers, then we won't have to catch the events
    powers, task_events = compute_normalized_powers(events,
                                                     kwargs['start_time'],
                                                     kwargs['end_time'],
                                                     kwargs['buf'],
                                                     kwargs['freqs'],
                                                     kwargs['log_powers'],
                                                     kwargs['filt_order'],
                                                     kwargs['width'])
    reduced_powers = reduce_powers(powers, used_pair_mask, len(kwargs['freqs']))

    sample_weights = get_sample_weights(task_events, **kwargs)

    classifier = train_classifier(reduced_powers,
                                  task_events,
                                  sample_weights,
                                  kwargs['C'],
                                  kwargs['penalty_type'],
                                  kwargs['solver'])

    cross_validation_results = perform_cross_validation(classifier,
                                                        reduced_powers,
                                                        task_events,
                                                        kwargs['n_perm'],
                                                        **kwargs)

    container = serialize_classifier(classifier,
                                     final_pairs,
                                     reduced_powers,
                                     task_events,
                                     sample_weights,
                                     cross_validation_results,
                                     subject)

    config_path = generate_ramulator_config(subject,
                                            experiment,
                                            container,
                                            stim_params,
                                            paths,
                                            ec_pairs,
                                            excluded_pairs)

    if vispath is not None:
        config_path.visualize(filename=vispath)

    return config_path.compute()
