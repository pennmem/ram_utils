"""Tasks related to summarizing an experiment. Used primarily in reporting
results.

"""

import numpy as np
import pandas as pd

from ._wrapper import task
from ramutils.events import validate_single_experiment, select_math_events, \
    extract_experiment_from_events, extract_sessions, select_session_events, \
    select_stim_table_events, extract_stim_information, \
    select_encoding_events, extract_event_metadata
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
def summarize_sessions(all_events, task_events, joint=False):
    """Generate a summary of by unique session/experiment

    Parameters
    ----------
    all_events: np.recarray
        Full set of events
    task_events : np.recarray
        Event subset used for classifier training
    joint: Bool
        Indicator for if a joint report is being created. This will disable
        checks for single-experiment events

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
        validate_single_experiment(task_events)

    # Since this takes 'cleaned' task events, we know the session numbers
    # have been made unique if cross-experiment events are given
    sessions = extract_sessions(task_events)

    summaries = []
    for session in sessions:
        experiment = extract_experiment_from_events(task_events)[0]
        # FIXME: recall_probs
        if experiment in ['FR1', 'catFR1']:
            summary = FRSessionSummary()

        # FIXME: recall_probs, ps4
        elif experiment in ['FR5', 'catFR5']:
            summary = FR5SessionSummary()

        # FIXME: other experiments
        else:
            raise UnsupportedExperimentError("Unsupported experiment: {}".format(experiment))

        summary.populate(task_events[task_events.session == session],
                         raw_events=all_events[all_events.session == session])
        summaries.append(summary)

    return summaries


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
def summarize_stim_sessions(all_events, task_events,
                            encoding_classifier_summaries):
    """ Construct stim session summaries """
    sessions = extract_sessions(task_events)
    stim_table_events = select_stim_table_events(all_events)

    # TODO: No need to enumerate once summaries have more metadata
    stim_session_summaries = []
    for i, session in enumerate(sessions):
        # Identify stim and post stim items
        all_session_events = select_session_events(stim_table_events, session)
        all_session_task_events = select_session_events(task_events, session)
        all_session_task_events = select_encoding_events(all_session_task_events)

        stim_item_mask, post_stim_item_mask, stim_param_df = \
            extract_stim_information(all_session_events,
                                     all_session_task_events)

        predicted_probabilities = encoding_classifier_summaries[i].predicted_probabilities
        subject, experiment, session = extract_event_metadata(task_events)
        stim_df = pd.DataFrame(columns=['subject', 'experiment', 'session',
                                        'list', 'item_name','serialpos',
                                        'phase', 'is_stim_list',
                                        'is_post_stim_item', 'it_stim_item',
                                        'recalled', 'thresh',
                                        'classifier_output'])

        stim_df['subject'] = subject
        stim_df['experiment'] = experiment
        stim_df['session'] = all_session_task_events.session
        stim_df['list'] = all_session_task_events.list
        stim_df['item_name'] = all_session_task_events.item_name
        stim_df['serialpos'] = all_session_task_events.serialpos
        stim_df['phase'] = all_session_task_events.phase
        stim_df['is_stim_list'] = stim_item_mask
        stim_df['is_post_stim_list'] = post_stim_item_mask
        stim_df['recalled'] = all_session_task_events.recalled
        stim_df['thresh'] = 0.5 # FIXME: Should this be hardcoded?
        stim_df['classifier_output'] = predicted_probabilities

        # Add in the stim params
        stim_df = stim_df.merge(stim_param_df, on=['session', 'list',
                                                   'item_name'], how='left')

        # TODO: Return the StimSessionSummary objects instead of dataframes
        # once from_datafram is implemented
        # stim_session_summary = StimSessionSummary()
        # stim_session_summary.from_dataframe(stim_df)
        stim_session_summaries.append(stim_df)

    return stim_session_summaries





