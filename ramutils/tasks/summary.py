"""Tasks related to summarizing an experiment. Used primarily in reporting
results.

"""

import numpy as np
import pandas as pd

from ._wrapper import task
from ramutils.events import validate_single_experiment, select_math_events, \
    extract_experiment_from_events, extract_sessions, select_session_events, \
    select_stim_table_events, extract_stim_information, \
    select_encoding_events, extract_event_metadata, dataframe_to_recarray, \
    separate_stim_events
from ramutils.exc import *
from ramutils.log import get_logger
from ramutils.reports.summary import *

logger = get_logger()

__all__ = [
    'summarize_nonstim_sessions',
    'summarize_math',
    'summarize_stim_sessions',
    'summarize_ps_sessions'
]


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
def summarize_nonstim_sessions(all_events, task_events, joint=False,
                               repetition_ratio_dict={}):
    """ Generate a summary by unique session/experiment

    Parameters
    ----------
    all_events: np.recarray
        Full set of events
    task_events : np.recarray
        Event subset used for classifier training
    joint: Bool
        Indicator for if a joint report is being created. This will disable
        checks for single-experiment events
    repetition_ratio_dict: Dict
        Mapping between subject ID and repetition ratio data

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

    """

    if not joint:
        validate_single_experiment(task_events)

    # Since this takes 'cleaned' task events, we know the session numbers
    # have been made unique if cross-experiment events are given
    sessions = extract_sessions(task_events)

    summaries = []
    for session in sessions:
        session_task_events = task_events[task_events.session == session]
        session_all_events = all_events[all_events.session == session]
        experiment = extract_experiment_from_events(session_task_events)[0]

        if experiment in ['FR1']:
            summary = FRSessionSummary()
            summary.populate(session_task_events,
                             raw_events=session_all_events)
        elif experiment in ['catFR1']:
            summary = CatFRSessionSummary()
            summary.populate(session_task_events,
                             raw_events=session_all_events,
                             repetition_ratio_dict=repetition_ratio_dict)
        else:
            raise UnsupportedExperimentError("Unsupported experiment: {}".format(experiment))

        summaries.append(summary)

    return summaries


@task()
def summarize_stim_sessions(all_events, task_events, stim_params,
                            encoding_classifier_summaries,
                            post_stim_predicted_probs,
                            pairs_data):
    """ Construct stim session summaries """
    sessions = extract_sessions(task_events)
    stim_table_events = select_stim_table_events(stim_params)
    location_data = pairs_data[['label', 'location']]
    location_data = location_data.dropna()

    stim_session_summaries = []
    for i, session in enumerate(sessions):
        all_session_events = select_session_events(all_events, session)
        all_session_stim_events = select_session_events(stim_table_events, session)
        all_session_task_events = select_session_events(task_events, session)
        all_session_task_events = select_encoding_events(all_session_task_events)

        stim_item_mask, post_stim_item_mask, stim_param_df = \
            extract_stim_information(all_session_stim_events,
                                     all_session_task_events)

        predicted_probabilities = encoding_classifier_summaries[i].predicted_probabilities
        subject, experiment, session = extract_event_metadata(
            all_session_task_events)
        stim_df = pd.DataFrame(columns=['subject', 'experiment', 'session',
                                        'list', 'mstime', 'item_name', 'type',
                                        'serialpos', 'phase', 'is_stim_item',
                                        'stim_list', 'is_post_stim_item',
                                        'recalled', 'thresh',
                                        'classifier_output'])
        expected_dtypes = [('serialpos', '<i8'),
                           ('session', '<i8'),
                           ('subject', '<U256'),
                           ('experiment', '<U256'),
                           ('mstime', '<i8'),
                           ('type', '<U256'),
                           ('recalled', '<i8'),
                           ('list', '<i8'),
                           ('stim_list', '<i8'),
                           ('phase', '<U256'),
                           ('item_name', '<U256'),
                           ('is_stim_item', '<i8'),
                           ('is_post_stim_item', '<i8'),
                           ('thresh', 'f'),
                           ('classifier_output', 'f'),
                           ('location', '<U256'),
                           ('amplitude', '<U256'),
                           ('pulse_freq', '<U256'),
                           ('stim_duration', '<U256'),
                           ('stimAnodeTag', '<U256'),
                           ('stimCathodeTag', '<U256')]

        stim_df['session'] = all_session_task_events.session
        stim_df['list'] = all_session_task_events.list
        stim_df['mstime'] = all_session_task_events.mstime
        stim_df['type'] = all_session_task_events.type
        stim_df['item_name'] = all_session_task_events.item_name
        stim_df['serialpos'] = all_session_task_events.serialpos
        stim_df['phase'] = all_session_task_events.phase
        stim_df['is_stim_item'] = stim_item_mask
        stim_df['is_post_stim_item'] = post_stim_item_mask
        stim_df['stim_list'] = all_session_task_events.stim_list
        stim_df['recalled'] = all_session_task_events.recalled
        stim_df['thresh'] = 0.5
        stim_df['classifier_output'] = predicted_probabilities
        stim_df['subject'] = subject
        stim_df['experiment'] = experiment

        # Add in the stim params
        stim_df = stim_df.merge(stim_param_df, on=['session', 'list',
                                                   'item_name'], how='left')

        # Add region from pairs_data. TODO: This won't scale to multi-site stim
        stim_df['label'] = (stim_df['stimAnodeTag'] + "-" +
                            stim_df['stimCathodeTag'])
        stim_df = stim_df.merge(location_data, how='left', on=['label'])
        del stim_df['label']

        # TODO: Add some sort of data quality check here potentially. Do the
        # observed stim items match what we expect from classifier output?

        if experiment in ['FR5', 'catFR5']:
            stim_events = dataframe_to_recarray(stim_df, expected_dtypes)
            stim_session_summary = FRStimSessionSummary()
            stim_session_summary.populate(
                stim_events, raw_events=all_session_events,
                post_stim_prob_recall=post_stim_predicted_probs[i])
        else:
            raise UnsupportedExperimentError('Only FR5 and catFR5 currently '
                                             'implemented')

        stim_session_summaries.append(stim_session_summary)

    return stim_session_summaries


@task()
def summarize_ps_sessions(ps_events):
    """ Task for generating summaries of PS session

    Parameters
    ----------
    ps_events

    """
    session_summaries = []
    sessions = extract_sessions(ps_events)
    for session in sessions:
        session_events = select_session_events(ps_events, session)
        summary = PSSessionSummary()
        summary.populate(session_events)
        session_summaries.append(summary)

    return session_summaries






