import os.path

from bptools.jacksheet import read_jacksheet
from ptsa.data.readers import JsonIndexReader

from ramutils.parameters import StimParameters
from ramutils.pipelines.eventprep import preprocess_fr_events
from ramutils.tasks import *


def make_stim_params(subject, anodes, cathodes, root='/'):
    """Construct :class:`StimParameters` objects from anode and cathode labels
    for a specific subject.

    Parameters
    ----------
    subject : str
    andoes : List[str] anodes
        anode labels
    cathodes : List[str]
        cathode labels
    root : str
        root directory to search for jacksheet

    Returns
    -------
    stim_params : List[StimParams]

    """
    path = os.path.join(root, 'data', 'eeg', subject, 'docs', 'jacksheet')
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


# FIXME: optionally infer exp_params from experiment name
def make_ramulator_config(subject, experiment, paths, anodes, cathodes,
                          exp_params, vispath=None):
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
    exp_params : ExperimentParameters
    vispath : str
        Path to save task graph visualization to if given.

    Returns
    -------
    The path to the generated configuration zip file.

    """
    jr = JsonIndexReader(os.path.join(paths.root, "protocols", "r1.json"))
    stim_params = make_stim_params(subject, anodes, cathodes, paths.root)

    # FIXME: update logic to work with PAL, AmplitudeDetermination
    if "FR" not in experiment:
        raise RuntimeError("Only FR-like experiments supported now.")

    encoding_events, retrieval_events = preprocess_fr_events(jr, subject)

    ec_pairs = generate_pairs_from_electrode_config(subject, paths)
    excluded_pairs = reduce_pairs(ec_pairs, stim_params, True)
    used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)
    final_pairs = generate_pairs_for_classifier(ec_pairs, excluded_pairs)

    # FIXME: If PTSA is updated to not remove events behind this scenes, this
    # won't be necessary. Or, if we can remove bad events before passing to
    # compute powers, then we won't have to catch the events
    encoding_powers, good_encoding_events = compute_powers(encoding_events, exp_params)
    retrieval_powers, good_retrieval_events = compute_powers(retrieval_events, exp_params)
    normalized_encoding_powers = normalize_powers_by_session(encoding_powers, good_encoding_events)
    normalized_retrieval_powers = normalize_powers_by_session(retrieval_powers, good_retrieval_events)

    task_events = combine_events([good_encoding_events, good_retrieval_events])
    powers = combine_encoding_retrieval_powers(task_events,
                                               normalized_encoding_powers,
                                               normalized_retrieval_powers)
    reduced_powers = reduce_powers(powers, used_pair_mask, len(exp_params.freqs))

    sample_weights = get_sample_weights(task_events, exp_params)
    classifier = train_classifier(powers, task_events, sample_weights, exp_params)
    cross_validation_results = perform_cross_validation(classifier, reduced_powers,
                                                        task_events, exp_params)

    container = serialize_classifier(classifier, final_pairs, reduced_powers,
                                     task_events, sample_weights,
                                     cross_validation_results,
                                     subject)

    config_path = generate_ramulator_config(subject, 'FR6', container, stim_params,
                                            paths, ec_pairs, excluded_pairs)

    if vispath is not None:
        config_path.visualize(filename=vispath)

    return config_path.compute()
