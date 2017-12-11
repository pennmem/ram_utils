"""Tasks related to summarizing an experiment. Used primarily in reporting
results.

"""

import numpy as np

from ._wrapper import task
from ramutils.events import validate_single_experiment, validate_single_session, select_math_events, \
    extract_experiment_from_events, extract_sessions
from ramutils.exc import *
from ramutils.log import get_logger
from ramutils.reports.summary import *

logger = get_logger()

__all__ = [
    'summarize_sessions',
    'summarize_math',
    'summarize_stim_sessions'
]


@task()
def summarize_sessions(events, joint=False):
    """Generate a summary of by unique session/experiment

    Parameters
    ----------
    events : np.recarray
        Events

    Returns
    -------
    summary : list
        List of SessionSummary objects for the proper experiment type.

    Raises
    ------
    TooManyExperimentsError
        If the events span more than one session.

    Notes
    -----
    The experiment type is inferred from the events.

    FIXME: make work with all experiments

    """

    if not joint:
        validate_single_experiment(events)

    # Since this takes 'cleaned' task events, we know the session numbers
    # have been made unique if cross-experiment events are given
    sessions = extract_sessions(events)

    summaries = []
    for session in sessions:
        experiment = extract_experiment_from_events(events)[0]
        # FIXME: recall_probs
        if experiment in ['FR1']:
            summary = FRSessionSummary()

        # FIXME: recall_probs, ps4
        elif experiment in ['FR5']:
            summary = FR5SessionSummary()

        # FIXME: other experiments
        else:
            raise UnsupportedExperimentError("Unsupported experiment: {}".format(experiment))

        summary.populate(events[events.session == session])
        summaries.append(summary)

    return summary


@task()
def summarize_math(events, joint=False):
    """ Generate a summary math event summary of a single experiment session

    Parameters
    ----------
    events: np.recarray
        Events from single experiment session
    joint: Bool
        Indicates if the given events are part of a joint event, and therefore
        multiple experiments should be allowed

    Returns
    -------
    summary: list
        List of MathSummary objects

    """
    if not joint:
        validate_single_experiment(events)

    math_events = select_math_events(events)
    if len(math_events) == 0:
        raise RuntimeError("No math events found when trying to summarize math "
                           "distractor period")

    sessions = extract_sessions(math_events)
    summaries = []
    for session in sessions:
        summary = MathSummary()
        summary.populate(math_events[math_events.session == session])
        summaries.append(summary)

    return summaries


@task()
def summarize_stim_sessions():
    return





