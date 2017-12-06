"""Tasks related to summarizing an experiment. Used primarily in reporting
results.

"""

import numpy as np

from ._wrapper import task
from ramutils.exc import *
from ramutils.log import get_logger
from ramutils.reports.summary import *

logger = get_logger()

__all__ = [
    'summarize_session',
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
    sessions = np.unique(events.session)
    if len(sessions) != 1:
        raise TooManySessionsError("events should be pre-filtered to be from a single session")

    experiments = np.unique(events.experiment)
    if len(experiments) != 1:
        raise TooManyExperimentsError("events should only come from one experiment")

    # session = sessions[0]
    experiment = experiments[0]

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
