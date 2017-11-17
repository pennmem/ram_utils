""" A collection of utility functions for loading, combining, and cleaning
events """

import os
import numpy as np

from ptsa.data.readers import BaseEventReader, JsonIndexReader
from ramutils.utils import extract_subject_montage


def load_events(subject, experiment, sessions=None, rootdir='/'):
    """ Load events for a specific subject and experiment

    Parameters
    ----------
    subject: str
    experiment: str
    sessions: iterable or None
    rootdir: str

    Returns
    -------
    np.recarray
        A numpy recarray containing all events for the requested subject,
        experiment, and session(s)

    """
    subject_id, montage = extract_subject_montage(subject)

    json_reader = JsonIndexReader(os.path.join(rootdir,
                                               "protocols",
                                               "r1.json"))

    sessions_to_load = sessions
    if sessions is None:
        # Find all sessions for the requested experiment
        sessions_to_load = json_reader.aggregate_values('sessions',
                                                        subject=subject_id,
                                                        experiment=experiment)

    event_files = sorted([json_reader.get_value('all_events',
                                                subject=subject,
                                                montage=montage,
                                                experiment=experiment,
                                                session=session)
                          for session in sorted(sessions_to_load)])

    # Update the paths based on the given root directory. This makes it easier
    # to run tests and use a mounted file system
    event_files = [os.path.join(rootdir, event_file) for event_file in
                   event_files]

    events = np.concatenate([BaseEventReader(filename=f).read() for f in
                             event_files]).view(np.recarray)
    return events


def concatenate_events_across_experiments(event_list):
    """
        Concatenate events across different experiment types. To make session
        numbers unique, 100 is added to the second set of events in event_list,
        200 to the next set of events, and so on.

    Parameters:
    -----------
    event_list: iterable
        An iterable containing events to be concatenated

    Returns:
    --------
    np.recarray
        The combined set of events

    """
    # Update sessions to not be in conflict
    session_offset = 0
    final_event_list = []
    for events in event_list:
        events.session += session_offset
        # This won't be necessary if using DataFrames
        events = select_column_subset(events)
        final_event_list.append(events)
        session_offset += 100

    # In order to combine events, we need have the same fields and types, which
    # effectively makes the events appear as though coming from the same
    # experiment
    final_events = concatenate_events_for_single_experiment(final_event_list)

    return final_events


def concatenate_events_for_single_experiment(event_list):
    """ Combine events that are part of the same experiment

    Parameters
    ----------
    event_list

    Returns
    -------
    np.recarray
        The flattened set of events

    """
    final_events = np.concatenate(event_list).view(np.recarray)
    final_events.sort(order=['session', 'list', 'mstime'])

    return final_events


def clean_events(events):
    """
        Peform basic cleaning operations on events such as removing
        incomplete sessions, negative offset events, etc. Any cleaning functions
        called here should be agnostic to the type of experiment.
        Experiment-specific cleaning should occur in a separate function

    Parameters
    ----------
    events: np.recarray of events

    Returns
    -------
    np.recarray
        Cleaned set of events

    """
    events = remove_negative_offsets(events)
    events = remove_practice_lists(events)
    events = remove_incomplete_lists(events)
    events = update_recall_outcome_for_retrieval_events(events)
    return events


def remove_negative_offsets(events):
    """ Remove events with a negative eegoffset """
    pos_offset_events = events[events['eegoffset'] >= 0]
    return pos_offset_events


def remove_incomplete_lists(events):
    """ Remove incomplete lists for every session in the given events

    """
    sessions = np.unique(events.session)
    final_event_list = []
    for session in sessions:
        sess_events = events[(events.session == session)]

        try:
            last_list = sess_events[sess_events.type == 'REC_END'][-1]['list']
            final_event_list.append(sess_events[sess_events.list <= last_list])
        except IndexError:
            final_event_list.append(sess_events)

    final_events = concatenate_events_for_single_experiment(final_event_list)

    return final_events


def update_recall_outcome_for_retrieval_events(events):
    """
        Manually override the recall outcomes for baseline retrieval and word
        retrieval events. All baseline retrieval events should be marked as not
        recalled and all word events in the recall period should be marked as
        recalled. WHY??

    Parameters:
    -----------
    events:

    Returns:
    --------
    np.recarray
        Events containing updated recall outcomes for retrieval events

    """
    events[events.type == 'REC_WORD'].recalled = 1
    events[events.type == 'REC_BASE'].recalled = 0
    return events


def remove_practice_lists(events):
    cleaned_events = events[events.list > -1]
    return cleaned_events


def remove_bad_events():
    # TODO: This should remove any events where the read window would be
    # outside the bounds of the associated EEG file
    raise NotImplementedError


def select_column_subset(events):
    # TODO: Need to update this to also work with PAL events
    columns = [
        'item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment',
        'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion',
        'montage', 'list', 'eegfile', 'msoffset'
    ]
    events = events[columns]
    return events


def insert_baseline_retrieval_events(events, start_time, end_time):
    """
        Match recall events to matching baseline periods of failure to recall.
        Baseline events all begin at least 1000 ms after a vocalization, and end
        at least 1000 ms before a vocalization. Each recall event is matched,
        wherever possible, to a valid baseline period from a different list
        within 3 seconds relative to the onset of the recall period.

    Parameters
    ----------
    events : np.recarray
        The event structure in which to incorporate these baseline periods
    start_time : int
        The amount of time to skip at the beginning of the session (ms)
    end_time : int
        The amount of time within the recall period to consider (ms)

    """

    # TODO: document within code blocks what is actually happening
    # TODO: Finish cleaning this mess up
    all_events = []
    for session in np.unique(events.session):
        sess_events = events[(events.session == session)]
        rec_events = select_retrieval_events(sess_events)
        voc_events = select_vocalization_events(sess_events)

        # Events corresponding to the start of the recall period
        starts = sess_events[(sess_events.type == 'REC_START')]

        # Events corresponding to the end of the recall period
        ends = sess_events[(sess_events.type == 'REC_END')]

        # Times associated with start and stop of recall period
        start_times = starts.mstime.astype(np.int)
        end_times = ends.mstime.astype(np.int)

        rec_lists = tuple(np.unique(starts.list))

        # Get list of vocalization times by list if there were any vocalizations
        # TODO: Pull this into its own function?
        times = [voc_events[(voc_events.list == lst)].mstime if (
            voc_events.list == lst).any() else []
                 for lst in rec_lists]

        # FIXME: Parameterize these values rather than hard-coding them
        epochs = find_free_time_periods(times,
                                        500,
                                        2000,
                                        1000,
                                        start=start_times,
                                        end=end_times)

        # FIXME: Wow... could this be any more confusing? Pull out into a
        # separate function
        rel_times = [(t - i)[(t - i > start_time) & (t - i < end_time)] for
                     (t, i) in
                     zip([rec_events[rec_events.list == lst].mstime for lst in
                          rec_lists], start_times)
                     ]
        rel_epochs = epochs - start_times[:, None]
        full_match_accum = np.zeros(epochs.shape, dtype=np.bool)

        for (i, rec_times_list) in enumerate(rel_times):
            is_match = np.empty(epochs.shape, dtype=np.bool)
            is_match[...] = False
            for t in rec_times_list:
                is_match_tmp = np.abs((rel_epochs - t)) < 3000
                is_match_tmp[i, ...] = False
                good_locs = np.where(is_match_tmp & (~full_match_accum))
                if len(good_locs[0]):
                    choice_position = np.argmin(
                        np.mod(good_locs[0] - i, len(good_locs[0])))
                    choice_inds = (good_locs[0][choice_position],
                                   good_locs[1][choice_position])
                    full_match_accum[choice_inds] = True

        matching_epochs = epochs[full_match_accum]
        new_events = np.zeros(len(matching_epochs),
                              dtype=sess_events.dtype).view(np.recarray)

        for i, _ in enumerate(new_events):
            new_events[i].mstime = matching_epochs[i]
            new_events[i].type = 'REC_BASE'

        new_events.recalled = 0
        merged_events = np.concatenate((sess_events, new_events)).view(
            np.recarray)
        merged_events.sort(order='mstime')

        for (i, event) in enumerate(merged_events):
            if event.type == 'REC_BASE':
                merged_events[i].session = merged_events[i - 1].session
                merged_events[i].list = merged_events[i - 1].list
                merged_events[i].eegfile = merged_events[i - 1].eegfile
                merged_events[i].eegoffset = merged_events[i - 1].eegoffset + (
                    merged_events[i].mstime - merged_events[i - 1].mstime)

        all_events.append(merged_events)

    return np.concatenate(all_events).view(np.recarray)


def find_free_time_periods(times, duration, pre, post, start=None, end=None):
    """
        Given a list of event times, find epochs between them when nothing is
        happening.

    Parameters
    ----------
    times : list or np.ndarray
        An iterable of 1-d numpy arrays, each of which indicates event times
    duration : int
        The length of the desired empty epochs
    pre : int
        the time before each event to exclude
    post: int
        The time after each event to exclude

    Returns
    -------
    epoch_array : np.ndarray

    """

    # TODO: Clean this up and add some explanation about what is happening
    n_trials = len(times)
    epoch_times = []
    for i in range(n_trials):
        ext_times = times[i]
        if start is not None:
            ext_times = np.append([start[i]], ext_times)
        if end is not None:
            ext_times = np.append(ext_times, [end[i]])
        pre_times = ext_times - pre
        post_times = ext_times + post
        interval_durations = pre_times[1:] - post_times[:-1]
        free_intervals = np.where(interval_durations > duration)[0]
        trial_epoch_times = []
        for interval in free_intervals:
            begin = post_times[interval]
            finish = pre_times[interval + 1] - duration
            interval_epoch_times = range(int(begin), int(finish), int(duration))
            trial_epoch_times.extend(interval_epoch_times)
        epoch_times.append(np.array(trial_epoch_times))

    epoch_array = np.empty((n_trials, max([len(x) for x in epoch_times])))
    epoch_array[...] = -np.inf
    for i, epoch in enumerate(epoch_times):
        epoch_array[i, :len(epoch)] = epoch

    return epoch_array


def select_word_events(events, include_retrieval=True):
    """
        Filter out non-WORD events. Assumes that baseline events are in the
        input events.

    :param np.recarray events:
    :param bool include_retrieval: Include REC_WORD and REC_BASE events
    :return: filtered events recarray

    """
    encoding_events_mask = get_encoding_mask(events)
    baseline_retrieval_event_mask = get_baseline_retrieval_mask(events)
    retrieval_event_mask = get_retrieval_events_mask(events)

    if include_retrieval:
        mask = (encoding_events_mask |
                baseline_retrieval_event_mask |
                retrieval_event_mask)
    else:
        mask = encoding_events_mask

    filtered_events = events[mask]

    events = filtered_events.view(np.recarray)
    return events


def get_time_between_events(events):
    """ Calculate the time between successive events"""
    inter_event_times = np.append([0], np.diff(events.mstime))
    return inter_event_times


def select_encoding_events(events):
    """ Select only encoding events """
    encoding_mask = get_encoding_mask(events)
    encoding_events = events[encoding_mask]
    return encoding_events


def get_encoding_mask(events):
    """ Create encoding event mask """
    encoding_mask = (events.type == "WORD")
    return encoding_mask


def select_vocalization_events(events):
    """ Select all vocalization events """
    vocalization_mask = get_vocalization_mask(events)
    vocalization_events = events[vocalization_mask]
    return vocalization_events


def get_vocalization_mask(events):
    """ Create mask for vocalization events"""
    vocalization_mask = ((events.type == 'REC_WORD') |
                         (events.type == 'REC_WORD_VV'))
    return vocalization_mask


def select_baseline_retrieval_events(events):
    """ Select baseline retrieval events """
    baseline_retrieval_mask = get_baseline_retrieval_mask(events)
    baseline_retrieval_events = events[baseline_retrieval_mask]
    return baseline_retrieval_events


def get_baseline_retrieval_mask(events):
    """ Create a boolean mask for baseline retrieval events """
    mask = (events.type == 'REC_BASE')
    if max(mask) is False:
        raise RuntimeError("No baseline retrieval events found. Create "
                           "baseline retrieval events first.")
    return mask


def select_retrieval_events(events):
    """ Select retrieval events """
    retrieval_events_mask = get_retrieval_events_mask(events)
    retrieval_events = events[retrieval_events_mask]
    return retrieval_events


def get_retrieval_events_mask(events):
    """
        Create boolean mask identifying retrieval events that were not
        intrusions and occured at least 1 second after the previous event.
        The idea here is that if retrieval takes less than 1 seond, there is
        likely not cognitive recall process going on, so we don't want to
        include these events as information to a classifier that uses both
        encoding and retrieval events in training
    """
    # FIXME: Parametrize the inter-event threshold
    # TODO: Why don't we actually study this 1000ms threshold to optimize it
    # or find out if it even matters?
    inter_event_times = get_time_between_events(events)
    retrieval_mask = ((events.type == 'REC_WORD') &
                      (events.intrusion == 0) &
                      (inter_event_times > 1000))
    return retrieval_mask


def select_all_retrieval_events(events):
    """ Select both baseline and actual retrieval events """
    retrieval_mask = get_all_retrieval_events_mask(events)
    retrieval_events = events[retrieval_mask]
    return retrieval_events


def get_all_retrieval_events_mask(events):
    """ Create a boolean bask for any retrieval event """
    all_retrieval_mask = ((events.type == 'REC_WORD') |
                          (events.type == 'REC_BASE'))
    return all_retrieval_mask



