import os.path

from bptools.jacksheet import read_jacksheet

from ramutils.constants import EXPERIMENTS
from ramutils.exc import MultistimNotAllowedException
from ramutils.parameters import StimParameters
from ramutils.tasks import *


def make_stim_params(subject, anodes, cathodes, min_amplitudes=None,
                     max_amplitudes=None, target_amplitudes=None,  root='/'):
    """Construct :class:`StimParameters` objects from anode and cathode labels
    for a specific subject.

    Parameters
    ----------
    subject : str
    anodes : List[str]
        anode labels
    cathodes : List[str]
        cathode labels
    min_amplitudes : List[float]
        Minimum stim amplitudes (when applicable)
    max_amplitudes : List[float]
        Maximum stim amplitudes (when applicable)
    target_amplitudes : List[float]
        Target stim amplitudes (when applicable)
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

        params = StimParameters(
            # FIXME: figure out better way to generate labels (read config file?)
            label='_'.join([anode, cathode]),
            anode=anode_idx,
            cathode=cathode_idx
        )

        if min_amplitudes is not None:
            params.min_amplitude = min_amplitudes[i]
            params.max_amplitude = max_amplitudes[i]
        else:
            params.target_amplitude = target_amplitudes[i]

        stim_params.append(params)

    return stim_params


def make_ramulator_config(subject, experiment, paths, stim_params,
                          exp_params=None, vispath=None):
    """ Generate configuration files for a Ramulator experiment

    Parameters
    ----------
    subject : str
        Subject ID
    experiment : str
        Experiment to generate configuration file for
    paths : FilePaths
    stim_params : List[StimParams]
        Stimulation parameters for this experiment.
    exp_params : ExperimentParameters
        Parameters for the experiment.
    vispath : str
        Path to save task graph visualization to if given.

    Returns
    -------
    The path to the generated configuration zip file.
    """
    if len(stim_params) > 1 and experiment not in EXPERIMENTS['multistim']:
        raise MultistimNotAllowedException

    ec_pairs = generate_pairs_from_electrode_config(subject, paths)
    excluded_pairs = reduce_pairs(ec_pairs, stim_params, True)
    used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)
    final_pairs = generate_pairs_for_classifier(ec_pairs, excluded_pairs)

    # Special case handling of Amplitude determination
    if experiment == "AmplitudeDetermination":
        container = None
        config_path = generate_ramulator_config(subject,
                                                experiment,
                                                container,
                                                stim_params,
                                                paths,
                                                ec_pairs,
                                                excluded_pairs)
        return config_path.compute()

    kwargs = exp_params.to_dict()
    events = preprocess_events(subject,
                               experiment,
                               kwargs['baseline_removal_start_time'],
                               kwargs['retrieval_time'],
                               kwargs['empty_epoch_duration'],
                               kwargs['pre_event_buf'],
                               kwargs['post_event_buf'],
                               encoding_only=kwargs['encoding_only'],
                               combine_events=kwargs['combine_events'],
                               root=paths.root)

    # FIXME: If PTSA is updated to not remove events behind this scenes, this
    # won't be necessary. Or, if we can remove bad events before passing to
    # compute powers, then we won't have to catch the events
    powers, task_events = compute_normalized_powers(events,
                                                    kwargs['encoding_start_time'],
                                                    kwargs['encoding_end_time'],
                                                    kwargs['encoding_buf'],
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
