"""Tasks related to summarizing an experiment. Used primarily in reporting
results.

"""

import numpy as np

from ._wrapper import task
from ramutils.events import validate_single_experiment, validate_single_session, select_math_events, extract_experiment_from_events
from ramutils.exc import *
from ramutils.log import get_logger
from ramutils.reports.summary import *

logger = get_logger()

__all__ = [
    'summarize_session',
    'summarize_math'
]


@task()
def summarize_session(events):
    """Generate a summary of a single experiment session.

    Parameters
    ----------
    events : np.recarray
        Events from a single

    Returns
    -------
    summary : SessionSummary
        Summary object for the proper experiment type.

    Raises
    ------
    TooManySessionsError
        If the events span more than one session.

    Notes
    -----
    The experiment type is inferred from the events.

    FIXME: make work with all experiments

    """
    validate_single_session(events)
    validate_single_experiment(events)

    # session = sessions[0]
    experiment = extract_experiment_from_events(events)[0]

    # FIXME: recall_probs
    if experiment in ['FR1']:
        summary = FRSessionSummary()
        summary.populate(events)

    # FIXME: recall_probs, ps4
    elif experiment in ['FR5']:
        summary = FR5SessionSummary()
        summary.populate(events)

    # FIXME: other experiments
    else:
        raise UnsupportedExperimentError("Unsupported experiment: {}".format(experiment))

    return summary


@task()
def summarize_math(events):
    """ Generate a summary math event summary of a single experiment session

    Parameters
    ----------
    events: np.recarray
        Events from single experiment session

    Returns
    -------
    summary: MathSummary
        Math summary object

    """
    validate_single_experiment(events)
    validate_single_session(events)

    math_events = select_math_events(events)
    if len(math_events) == 0:
        raise RuntimeError("Not math events found when trying to summarize math distractor period")
    summary = MathSummary()
    summary.populate(events)

    return summary
