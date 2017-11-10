import os.path

from bptools.jacksheet import read_jacksheet
from ptsa.data.readers import JsonIndexReader

from ramutils.parameters import StimParameters
from ramutils.tasks.classifier import *
from ramutils.tasks.events import *
from ramutils.tasks.montage import *
from ramutils.tasks.odin import *
from ramutils.tasks.powers import *


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


def make_ramulator_config(subject, experiment, paths, anodes, cathodes, exp_params):
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

    Returns
    -------
    The path to the generated configuration zip file.

    """
    jr = JsonIndexReader(os.path.join(paths.root, "protocols", "r1.json"))
    stim_params = make_stim_params(subject, anodes, cathodes, paths.root)

    fr_events = read_fr_events(jr, subject, cat=False)
    catfr_events = read_fr_events(jr, subject, cat=True)
    events = concatenate_events(fr_events, catfr_events)

    pairs = load_pairs(paths.pairs)
    excluded_pairs = reduce_pairs(pairs, stim_params, True)

    ec_pairs = generate_pairs_from_electrode_config(subject, paths)

    if experiment != 'ampdet':
        powers = compute_powers(events, exp_params)
        classifier, xval, sample_weights = compute_classifier(events, powers, exp_params, paths)
        container = serialize_classifier(classifier, pairs, powers, events, sample_weights, xval, subject)
    else:
        container = None

    config_path = generate_ramulator_config(subject, experiment, container,
                                            stim_params, paths, ec_pairs, excluded_pairs)

    return config_path
