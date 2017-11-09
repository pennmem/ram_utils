import os.path

from ptsa.data.readers import JsonIndexReader

from ramutils.tasks.classifier import *
from ramutils.tasks.events import *
from ramutils.tasks.montage import *
from ramutils.tasks.odin import *
from ramutils.tasks.powers import *


def make_ramulator_config(subject, experiment, paths, exp_params):
    """Generate configuration files for a Ramulator experiment.

    Parameters
    ----------
    subject : str
        Subject ID
    experiment : str
        Experiment to generate configuration file for
    paths : FilePaths
    exp_params : ExperimentParameters

    Returns
    -------
    The path to the generated configuration zip file.

    """
    jr = JsonIndexReader(os.path.join(paths.root, "protocols", "r1.json"))

    # FIXME
    stim_params = None

    fr_events = read_fr_events(jr, subject, cat=False)
    catfr_events = read_fr_events(jr, subject, cat=True)
    events = concatenate_events(fr_events, catfr_events)

    pairs = load_pairs(paths.pairs)
    excluded_pairs = reduce_pairs(pairs, stim_params, True)

    ec_pairs = generate_pairs_from_electrode_config(subject, paths)

    powers = compute_powers(events, exp_params)
    classifier, xval, sample_weights = compute_classifier(events, powers, exp_params, paths)
    container = serialize_classifier(classifier, pairs, powers, events, sample_weights, xval, subject)

    config_path = generate_ramulator_config(subject, experiment, container,
                                            stim_params, paths, ec_pairs, excluded_pairs)

    return config_path
