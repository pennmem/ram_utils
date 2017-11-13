"""Pipeline for creating reports."""

from ptsa.data.readers import JsonIndexReader

from ramutils.constants import EXPERIMENTS
from ramutils.pipelines.eventprep import preprocess_fr_events
from ramutils.tasks import *


def make_report(subject, experiment, paths, classifier=None, exp_params=None,
                sessions=None, vispath=None):
    """Run a report.

    Parameters
    ----------
    subject : str
        Subject ID
    experiment : str
        Experiment to generate report for
    paths : FilePaths
    classifier : ClassifierContainer
        For experiments that ran with a classifier, the container detailing the
        classifier that was actually used. When not given, a new classifier will
        be trained to (hopefully) recreate what was actually used.
    exp_params : ExperimentParameters
        When given, overrides the inferred default parameters to use for an
        experiment.
    sessions : list
        For reports that span sessions, sessions to read data from.
        When not given, all available sessions are used for reports.
    vispath : str
        Filename for task graph visualization.

    Returns
    -------
    report_path : str
        Path to generated report.

    Notes
    -----
    Eventually this will return an object that summarizes all output of the
    report rather than the report itself.

    """
    jr = JsonIndexReader(os.path.join(paths.root, "protocols", "r1.json"))

    if "FR" in experiment:
        encoding_events, retrieval_events = preprocess_fr_events(jr, subject)
    else:
        raise RuntimeError("only FR supported so far")

    # FIXME: can this be centralized?
    ec_pairs = generate_pairs_from_electrode_config(subject, paths)
    excluded_pairs = reduce_pairs(ec_pairs, stim_params, True)
    used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)
    final_pairs = generate_pairs_for_classifier(ec_pairs, excluded_pairs)

    if classifier is None:
        pass  # TODO: compute powers, train classifier

    # TODO: Compute powers

    if experiment in (EXPERIMENTS['closed_loop'] + EXPERIMENTS['ps']):
        pass  # TODO: compute stim table

    # TODO: generate summary

    # TODO: generate plots, generate tex, generate PDF
