"""Pipeline for creating reports."""

from ramutils.tasks import *


def make_report(subject, experiment, classifier=None, exp_params=None,
                sessions=None, vispath=None):
    """Run a report.

    Parameters
    ----------
    subject : str
        Subject ID
    experiment : str
        Experiment to generate report for
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
