"""Tasks related to summarizing an experiment. Used primarily in reporting
results.

"""

import numpy as np
import pandas as pd

from ._wrapper import task
from ramutils.events import (
    validate_single_experiment, select_math_events,
    extract_experiment_from_events, extract_sessions, select_session_events,
    select_stim_table_events, extract_stim_information,
    select_encoding_events, extract_event_metadata, dataframe_to_recarray,
    get_encoding_mask, correct_fr2_stim_item_identification,
    extract_biomarker_information)
from ramutils.exc import *
from ramutils.log import get_logger
from ramutils.reports.summary import *
# from ramutils.reports.summary import (FRSessionSummary, MathSummary,
#                                       FRStimSessionSummary, TICLFRSessionSummary,
#                                       LocationSearchSessionSummary, repFRSessionSummary,
#                                       EFRCourierSessionSummary, EFRCourierStimSessionSummary)
from ramutils.tasks.thetamod import get_psd_data
from os.path import basename

logger = get_logger()

__all__ = [
    'summarize_nonstim_sessions',
    'summarize_math',
    'summarize_stim_sessions',
    'summarize_ps_sessions',
    'summarize_location_search_sessions',
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
        pass # need to let repFR pass, and don't know of an instance when this protection is needed
        #raise RuntimeError("No math events found when trying to summarize math "distractor period")

    sessions = extract_sessions(math_events)
    summaries = []
    for session in sessions:
        summary = MathSummary()
        summary.populate(math_events[math_events.session == session])
        summaries.append(summary)

    return summaries


@task()
def summarize_nonstim_sessions(all_events, task_events,
                               bipolar_pairs, excluded_pairs,
                               normalized_powers, joint=False,
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
        session_powers = normalized_powers[(task_events.session == session)]
        experiment = extract_experiment_from_events(session_task_events)[0]

        if experiment in ['FR1', 'IFR1']:
            summary = FRSessionSummary()
            summary.populate(session_task_events,
                             bipolar_pairs,
                             excluded_pairs,
                             session_powers,
                             raw_events=session_all_events)
        elif experiment in ['catFR1', 'ICatFR1']:
            summary = CatFRSessionSummary()
            summary.populate(session_task_events,
                             bipolar_pairs,
                             excluded_pairs,
                             session_powers,
                             raw_events=session_all_events,
                             repetition_ratio_dict=repetition_ratio_dict)
        elif experiment in ['RepFR1']:
            summary = repFRSessionSummary()
            summary.populate(session_task_events,
                             bipolar_pairs,
                             excluded_pairs,
                             session_powers,
                             raw_events=session_all_events)

        elif experiment in ['DBOY1']:
            summary = FRSessionSummary()
            summary.populate(session_task_events,
                             bipolar_pairs,
                             excluded_pairs,
                             session_powers,
                             raw_events=session_all_events)
        
        elif experiment in ['EFRCourierOpenLoop']:
            summary = EFRCourierSessionSummary()
            summary.populate(session_task_events,
                             bipolar_pairs,
                             excluded_pairs,
                             session_powers,
                             raw_events=session_all_events)
        
        elif experiment in ['EFRCourierReadOnly']:
            summary = EFRCourierSessionSummary()
            summary.populate(session_task_events,
                             bipolar_pairs,
                             excluded_pairs,
                             session_powers,
                             raw_events=session_all_events)

        else:
            raise UnsupportedExperimentError(
                "Unsupported experiment: {}".format(experiment))

        summaries.append(summary)

    return summaries


@task()
def summarize_stim_sessions(all_events, task_events, stim_params, pairs_data,
                            bipolar_pairs, excluded_pairs,
                            normalized_powers,
                            encoding_classifier_summaries=None,
                            post_stim_predicted_probs=None,
                            trigger_output=None,
                            post_stim_trigger_output=None,
                            post_stim_eeg=None):
    """ Construct stim session summaries """
    sessions = extract_sessions(task_events)
    stim_table_events = select_stim_table_events(stim_params)
    location_data = pairs_data[['label', 'location']]
    location_data = location_data.dropna()

    stim_session_summaries = []
    for i, session in enumerate(sessions):
        all_session_events = select_session_events(all_events, session)
        all_session_stim_events = select_session_events(
            stim_table_events, session)
        all_session_task_events = select_session_events(task_events, session)
        encoding_mask = get_encoding_mask(all_session_task_events)

        # Careful: Events and powers need to have the same number of entries
        all_session_task_events = select_encoding_events(
            all_session_task_events)
        session_powers = normalized_powers[encoding_mask]
        assert len(all_session_task_events) == len(session_powers)

        stim_item_mask, post_stim_item_mask, stim_param_df = \
            extract_stim_information(all_session_stim_events,
                                     all_session_task_events)

        stim_param_df["stimAnodeTag"] = stim_param_df["stimAnodeTag"].str.rstrip(
            ',')
        stim_param_df["stimCathodeTag"] = stim_param_df["stimCathodeTag"].str.rstrip(
            ',')

        # PS5 sessions do not have classifier summaries, but use the raw
        # power value output for making the stim decision. Open loop stim
        # sessions do not have a classifier, so there is no threshold
        if encoding_classifier_summaries is not None:
            predicted_probabilities = encoding_classifier_summaries[i].predicted_probabilities
            thresh = 0.5
        elif trigger_output is not None:
            # We don't want retrieval powers for the triggering electrode
            predicted_probabilities = trigger_output[encoding_mask]
            event_based_avg = []
            for j in range(len(predicted_probabilities)):
                powers_so_far = predicted_probabilities[:j]
                event_based_avg.append(np.mean(powers_so_far))
            thresh = event_based_avg
        else:
            predicted_probabilities = np.nan
            thresh = np.nan

        subject, experiment, session = extract_event_metadata(
            all_session_task_events)
        stim_df = pd.DataFrame(columns=['subject', 'experiment', 'session',
                                        'list', 'mstime', 'item_name', 'type',
                                        'serialpos', 'phase', 'is_stim_item',
                                        'is_stim_list', 'is_post_stim_item',
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
                           ('is_stim_list', '<i8'),
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
        stim_df['is_stim_list'] = all_session_task_events.stim_list
        stim_df['recalled'] = all_session_task_events.recalled
        stim_df['thresh'] = thresh
        stim_df['classifier_output'] = predicted_probabilities
        stim_df['subject'] = subject
        stim_df['experiment'] = experiment

        # Add in the stim params. This is making the assumption that stim
        # parameters do not change within a list
        stim_param_df = stim_param_df.drop_duplicates(
            subset=['session', 'list'])
        stim_df = stim_df.merge(
            stim_param_df, on=['session', 'list', 'item_name'], how='left')

        # Add region from pairs_data. TODO: This won't scale to multi-site stim
        stim_df['label'] = (stim_df['stimAnodeTag'] + "-" +
                            stim_df['stimCathodeTag'])
        stim_df = stim_df.merge(location_data, how='left', on=['label'])
        del stim_df['label']

        # TODO: Add some sort of data quality check here potentially. Do the
        # observed stim items match what we expect from classifier output?

        if experiment in ['FR3', 'FR5', 'catFR3', 'catFR5', 'FR6', 'catFR6', 'IFR6', 'ICatFR5', 'ICatFR6']:
            stim_events = dataframe_to_recarray(stim_df, expected_dtypes)
            stim_session_summary = FRStimSessionSummary()
            stim_session_summary.populate(
                stim_events, bipolar_pairs, excluded_pairs, session_powers,
                raw_events=all_session_events,
                post_stim_prob_recall=post_stim_predicted_probs[i],
                post_stim_eeg=post_stim_eeg
            )

        elif experiment in ['FR2', 'catFR2']:
            # The usual algorithm for identifying stim events will miss some
            # specifically for FR2
            stim_df = correct_fr2_stim_item_identification(stim_df)
            stim_events = dataframe_to_recarray(stim_df, expected_dtypes)
            stim_session_summary = FRStimSessionSummary()
            stim_session_summary.populate(
                stim_events, bipolar_pairs, excluded_pairs, session_powers,
                raw_events=all_session_events)

        elif experiment in ["PS5_FR", "PS5_catFR"]:
            stim_events = dataframe_to_recarray(stim_df, expected_dtypes)
            stim_session_summary = FRStimSessionSummary()
            stim_session_summary.populate(
                stim_events, bipolar_pairs,
                excluded_pairs, trigger_output, raw_events=all_session_events,
                post_stim_prob_recall=post_stim_trigger_output)
        elif experiment in ["TICL_FR", "TICL_catFR"]:
            biomarker_events = extract_biomarker_information(
                all_session_stim_events)
            stim_events = dataframe_to_recarray(stim_df,expected_dtypes)
            stim_session_summary = TICLFRSessionSummary()
            stim_session_summary.populate(
                stim_events, bipolar_pairs,
                excluded_pairs, session_powers,
                raw_events=all_session_events,
                biomarker_events=biomarker_events,
                post_stim_eeg=post_stim_eeg,
                stim_tstats=pairs_data[['stim_tstats','stim_pvals']].to_records(index=False)
            )

        else:
            raise UnsupportedExperimentError('Experiment not supported')

        stim_session_summaries.append(stim_session_summary)

        # Do a quick quality check here to see that the number of stim items
        # matches the size of the post_stim_prob_recall. We do not calculate
        # post stim prob recall for FR2 or TICL, so do not check in that case
        if experiment not in ['FR2', 'catFR2', 'TICL_FR', 'TICL_catFR']:
            num_stim_items = FRStimSessionSummary.pre_stim_prob_recall([stim_session_summary])
            num_post_stim_prob_recall = FRStimSessionSummary.all_post_stim_prob_recall([stim_session_summary])
            if len(num_stim_items) != len(num_post_stim_prob_recall):
                logger.warning("Number of identified stim items ({}) does not "
                               "match the  number of STIM_OFF events ({}). Confirm "
                               "that the stim item identification algorithm is "
                               "working correctly".format(len(num_stim_items),
                                                          len(num_post_stim_prob_recall)))

    return stim_session_summaries


@task()
def summarize_ps_sessions(ps_events, bipolar_pairs, excluded_pairs):
    """ Task for generating summaries of PS session

    Parameters
    ----------
    ps_events: np.recarray
    bipolar_pairs: dict
    excluded_pairs: dict

    """
    session_summaries = []
    sessions = extract_sessions(ps_events)
    for session in sessions:
        session_events = select_session_events(ps_events, session)
        summary = PSSessionSummary()
        summary.populate(session_events, bipolar_pairs, excluded_pairs, None)
        session_summaries.append(summary)

    return session_summaries


@task()
def summarize_location_search_sessions(all_events, stim_params, pairs_metadata_table, excluded_pairs,
                                       connectivity, post_stim_eeg, rootdir='/'):
    session_summaries = []
    subject, experiment, sessions = extract_event_metadata(all_events)
    bipolar_pairs = {subject: {'pairs': pairs_metadata_table.to_dict(orient='index')}}
    locations = pairs_metadata_table[['location', 'label']]
    locations.index = pairs_metadata_table.channel_1.astype(int)
    locations = pd.DataFrame(locations)
    expected_dtypes = [('subject', '<U256'),
                       ('experiment', '<U256'),
                       ('eegoffset', '<i8'),
                       ('session', '<i8'),
                       ('type', '<U256'),
                       ('mstime', '<i8'),
                       ('amplitude', '<U256'),
                       ('pulse_freq', '<U256'),
                       ('stim_duration', '<U256'),
                       ('stimAnodeTag', '<U256'),
                       ('stimCathodeTag', '<U256'),
                       ('location', '<U256'),
                       ('label', '<U256'),
                       ]
    stim_param_df = pd.DataFrame(stim_params['stim_params'])[['amplitude', 'pulse_freq', 'stim_duration',
                                                              'anode_label', 'cathode_label', 'anode_number']]
    stim_param_df = stim_param_df.merge(locations, how='left', left_on='anode_number', right_index=True)

    stim_param_df.rename(columns={'anode_label': 'stimAnodeTag',
                                  'cathode_label': 'stimCathodeTag', }, inplace=True)

    stim_param_df.drop(columns='anode_number', inplace=True)
    events_df = pd.DataFrame(all_events)

    # need to remove path to load eeg with cmlreaders
    events_df["eegfile"] = events_df["eegfile"].apply(lambda x: basename(x))

    events_df = events_df.loc[events_df['type'] == 'STIM_ON']
    events_df.drop(columns=['phase', ], inplace=True)
    events_df = events_df.merge(stim_param_df, how='left', left_index=True, right_index=True,).reset_index(drop=True)
    for _, target_df in events_df.groupby(['session', 'stimAnodeTag', 'stimCathodeTag']):
        idxs = target_df.index.tolist()
        target_events = dataframe_to_recarray(target_df, expected_dtypes)
        pre_psd, post_psd, emask, cmask = get_psd_data(target_df, rootdir)
        summary = LocationSearchSessionSummary()
        summary.populate(
            target_events, bipolar_pairs, excluded_pairs,
            connectivity, pre_psd, post_psd,
            emask, cmask,
            post_stim_eeg=post_stim_eeg[:, idxs, :],
            stim_tstats=pairs_metadata_table[['stim_tstats', 'stim_pvals']].to_records(index=False)
        )
        session_summaries.append(summary)
    return session_summaries
