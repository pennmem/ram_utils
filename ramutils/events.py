"""

    A collection of utility functions for loading, cleaning, normalizing, and
    combining events. There are also a smattering of other helper function for
    selecting specific types of events. In general, the following steps must be
    taken to go from raw (on-disk) events to events that can be analyzed:

        1. Load: Load events from disk into memory
        2. Clean: Perform standard sets of cleaning operations
        3. Normalize: Modify fields and values so that events from different
           experiment can be easily combined

"""

import os
import numpy as np

from itertools import groupby
from numpy.lib.recfunctions import rename_fields

from ptsa.data.readers import BaseEventReader, JsonIndexReader, EEGReader
from ramutils.utils import extract_subject_montage


def preprocess_events(subject, experiment, start_time,
                      end_time, duration, pre, post, combine_events=True,
                      encoding_only=False, root='/'):
    """High-level pre-processing function for combining/cleaning record only
    events to be used in config generation and reports

    Parameters
    ----------
    subject : str
        Subject ID.
    experiment: str
        Experiment that events are being combined for
    start_time: int
    end_time: int
    duration: int
    pre: int
    post: int
    combine_events: bool
        Indicates if all record-only sessions should be combined for
        classifier training.
    encoding_only: bool
        Flag for if only encoding events should be used (default is False,
        i.e. encoding and retrieval events will be returned)
    root: str
        Base path for finding event files etc.

    Returns
    -------
    np.recarray
        Full set of cleaned task events

    Notes
    -----
    See insert_baseline_events for details on start_time, end_time, duration,
    pre, and post parameters

    """
    if ("FR" not in experiment) and ("PAL" not in experiment):
        raise RuntimeError("Only PAL, FR, and catFR experiments are currently"
                           "implemented")

    if "PAL" in experiment:
        pal_events = load_events(subject, "PAL1", rootdir=root)
        pal_events = clean_events("PAL1", pal_events)
        pal_events = normalize_pal_events(pal_events)

    if ("FR" in experiment) or combine_events:
        fr_events = load_events(subject, 'FR1', rootdir=root)
        fr_events = clean_events("FR1",
                                 fr_events,
                                 start_time=start_time,
                                 end_time=end_time,
                                 duration=duration,
                                 pre=pre,
                                 post=post)
        fr_events = normalize_fr_events(fr_events)

        catfr_events = load_events(subject, 'catFR1', rootdir=root)
        catfr_events = clean_events("catFR1",
                                    catfr_events,
                                    start_time=start_time,
                                    end_time=end_time,
                                    pre=pre,
                                    post=post,
                                    duration=duration)
        catfr_events = normalize_fr_events(catfr_events)

        # Free recall events are always combined
        free_recall_events = concatenate_events_across_experiments(
            [fr_events, catfr_events])

    if ("PAL" in experiment) and combine_events:
        all_events = concatenate_events_across_experiments([
            free_recall_events, pal_events])

    elif ("PAL" in experiment) and not combine_events:
        all_events = pal_events

    else:
        all_events = free_recall_events

    final_events = select_word_events(all_events,
                                      encoding_only=encoding_only)

    return final_events


def load_events(subject, experiment, sessions=None, rootdir='/'):
    """ Load events for a specific subject and experiment. If no events are
    found, an empty recarray with the correct datatypes are returned

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

    if len(event_files) == 0:
        empty_recarray = initialize_empty_event_reccarray()
        return empty_recarray

    # TODO: Make this less ugly to look at
    events = np.concatenate([
        BaseEventReader(filename=f, eliminate_events_with_no_eeg=True).read()
        for f in event_files]).view(np.recarray)

    return events


def clean_events(experiment, events, start_time=None, end_time=None,
                 duration=None, pre=None, post=None):
    """Peform basic cleaning operations on events such as removing incomplete
    sessions, negative offset events, and incomplete lists. For FR events,
    baseline events needs to be found.

    Parameters
    ----------
    experiment: str
    events: np.recarray of events
    start_time:
    end_time:
    duration:
    pre:
    post:

    Returns
    -------
    np.recarray
        Cleaned set of events

    Notes
    -----
    This function should be called on an experiment by experiment basis and
    should not be used to clean cross-experiment datasets

    """
    events = remove_negative_offsets(events)
    events = remove_practice_lists(events)
    events = remove_incomplete_lists(events)

    if "FR" in experiment:
        events = insert_baseline_retrieval_events(events,
                                                  start_time,
                                                  end_time,
                                                  duration,
                                                  pre,
                                                  post)

        events = remove_intrusions(events)
        events = update_recall_outcome_for_retrieval_events(events)

    if "PAL" in experiment:
        events = subset_pal_events(events)
        events = update_pal_retrieval_events(events)
        events = remove_nonresponses(events)

    return events


def normalize_fr_events(events):
    events = combine_retrieval_events(events)
    return events


def normalize_pal_events(events):
    """Perform any normalization to PAL event so make the homogeneous enough so
    that it is trivial to combine with other experiment events.

    """
    events = rename_correct_to_recalled(events)
    events = coerce_study_pair_to_word_event(events)
    return events


def rename_correct_to_recalled(events):
    """Normalizes PAL "recall" event names to match those of FR experiments.

    Parameters
    ----------
    events: np.recarray

    Returns
    -------
    np.recarray
        Events with a 'recalled' field added to mirror the 'correct' field

    """
    events = rename_fields(events, {'correct': 'recalled'})

    return events


def remove_negative_offsets(events):
    """ Remove events with a negative eegoffset """
    pos_offset_events = events[events['eegoffset'] >= 0]
    return pos_offset_events


def remove_incomplete_lists(events):
    """Remove incomplete lists for every session in the given events. Note,
    there are two ways that this is done in the reporting code, so it is an
    outstanding item to determine which method is better

    """
    # TODO: This needs to be cleaned up and tested
    sessions = np.unique(events.session)
    final_event_list = []
    for session in sessions:
        sess_events = events[(events.session == session)]

        # partition events into math and task
        math_mask = np.in1d(sess_events.type, ['START', 'STOP', 'PROB'])
        task_events = sess_events[~math_mask]
        math_events = sess_events[math_mask]
        final_sess_events = task_events
        final_sess_events.sort(order=['session','list','mstime'])

        # Remove all task events for lists that don't have a "REC_END" event
        events_by_list = (np.array([l for l in list_group]) for listno,
                                                                list_group in
                          groupby(final_sess_events, lambda x: x.list))
        list_has_end = [any([l['type'] == 'REC_END' for l in list_group]) or
                        listno == -999 for listno, list_group in groupby(
            final_sess_events, lambda x:x.list)]
        final_sess_events = np.concatenate([e for (e, a) in zip(
            events_by_list, list_has_end) if a])

        # Re-combine math and task events
        final_sess_events = np.concatenate([final_sess_events,
                                            math_events]).view(np.recarray)
        final_sess_events.sort(order=['session', 'list', 'mstime'])
        final_event_list.append(final_sess_events)

        # METHOD #2 (perhaps less accurate?) We need to figure out which one
        # should be used. Don't delete for now
        # try:
        #     last_list = sess_events[sess_events.type == 'REC_END'][-1]['list']
        #     final_event_list.append(sess_events[sess_events.list <= last_list])
        # except IndexError:
        #     final_event_list.append(sess_events)

    final_events = concatenate_events_for_single_experiment(final_event_list)

    return final_events


def remove_nonresponses(events):
    """ Selects only events that were listed as recalled or not recalled """
    events = events[(events.correct == 0) | (events.correct == 1)]
    return events


def subset_pal_events(events):
    """ Only a subset of event types are needed for PAL experiments """
    events = events[(events.type == 'STUDY_PAIR') |
                    (events.type == 'TEST_PROBE') |
                    (events.type == 'PROBE_START')]
    return events


def update_recall_outcome_for_retrieval_events(events):
    """Manually override the recall outcomes for baseline retrieval and word
    retrieval events. All baseline retrieval events should be marked as not
    recalled and all word events in the recall period should be marked as
    recalled. This assumes that intrusions have already been removed from the
    given set of events. It exists merely to serve as an extra check on what
    should already be true in the raw events data.

    Parameters
    ----------
    events: np.recarray

    Returns
    -------
    np.recarray
        Events containing updated recall outcomes for retrieval events

    """
    events[events.type == 'REC_WORD'].recalled = 1
    events[events.type == 'REC_BASE'].recalled = 0
    return events


def update_pal_retrieval_events(events):
    """Create surrogate responses for retrieval period based on PS4/PAL5 design
    doc. Surrogate responses are created by identifying trials without any
    responses. For these trials, a new response time is created based on a
    random draw from the set of response times from actual responses.

    """
    # Identify the sample rate
    samplerate = 1000 #extract_sample_rate(events)

    # Separate retrieval and non-retrieval events
    retrieval_mask = get_pal_retrieval_events_mask(events)
    retrieval_events = events[retrieval_mask]
    nonretrieval_events = events[~retrieval_mask]

    incorrect_no_response_mask = (retrieval_events.RT == -999)
    correct_mask = (retrieval_events.correct == 1)

    correct_response_times = retrieval_events[correct_mask].RT
    response_time_rand_indices = np.random.randint(0,
                                                   len(correct_response_times),
                                                   sum(incorrect_no_response_mask))
    retrieval_events.RT[incorrect_no_response_mask] = correct_response_times[
        response_time_rand_indices]
    retrieval_events.type = 'REC_EVENT'
    retrieval_events.eegoffset = retrieval_events.eegoffset + (
        retrieval_events.RT * (samplerate/1000.0)).astype(np.int64)

    # Staple everything back together
    cleaned_events = concatenate_events_for_single_experiment([
        retrieval_events, nonretrieval_events])

    return cleaned_events


def combine_retrieval_events(events):
    """
        Combine baseline retrieval and actual retrieval events into a single
        event type.

    """
    events.type[(events.type == 'REC_WORD') |
                (events.type == 'REC_BASE')] = 'REC_EVENT'
    return events


def coerce_study_pair_to_word_event(events):
    """Update STUDY_PAIR events to be WORD events. These are the same event
    type, but PAL calls them STUDY_PAIR and FR/catFR call them WORD. In the
    future, it may make more sense to make an update to event creation instead
    of coercing the event types here.

    """
    events.type[(events.type == 'STUDY_PAIR')] = 'WORD'
    return events


def remove_practice_lists(events):
    """ Remove practice lists from the set of events """
    cleaned_events = events[events.list > -1]
    return cleaned_events


def remove_bad_events(events):
    """Remove events whose offset values would result in trying to read data
    that is out of bounds in the EEG file. Currently, this is done automatically
    in PTSA when you load the EEG, but to avoid having to catch updated events
    when reading the EEG, it should be done ahead of time.

    """
    raise NotImplementedError


def select_column_subset(events):
    columns = [
        'serialpos', 'session', 'subject', 'rectime', 'experiment',
        'mstime', 'type', 'eegoffset', 'recalled', 'intrusion',
        'montage', 'list', 'eegfile', 'msoffset'
    ]
    events = events[columns]
    return events


def initialize_empty_event_reccarray():
    """Utility function for generating a recarray that looks normalized,
    but is empty.

    """
    empty_recarray = np.recarray((0, ), dtype=[('serialpos', '<i8'),
                                               ('session', '<i8'),
                                               ('subject', '|U6'),
                                               ('rectime', '<i8'),
                                               ('experiment', '|U256'),
                                               ('mstime', '<i8'),
                                               ('type', '|U256'),
                                               ('eegoffset', '<i8'),
                                               ('recalled', '<i8'),
                                               ('intrusion', '<i8'),
                                               ('montage', '<i8'),
                                               ('list', '<i8'),
                                               ('eegfile', '|U256'),
                                               ('msoffset', '<i8')])
    return empty_recarray


def insert_baseline_retrieval_events(events, start_time, end_time, duration,
                                     pre, post):
    """Match recall events to matching baseline periods of failure to recall.
    This is required for all free recall events, but is not necessary for
    PAL events, which have a natural baseline/comparison group. Baseline
    events all begin at least 1000 ms after a vocalization, and end
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
    duration: int
        The length of desired empty epochs
    pre: int
        The time before each event to exclude
    post: int
        The time after each event to exclude

    Returns
    -------
    np.reccarray
        Events with REC_BASE event types inserted

    """

    if len(events) == 0:
        return events

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

        epochs = find_free_time_periods(times,
                                        duration,
                                        pre,
                                        post,
                                        start=start_times,
                                        end=end_times)

        # FIXME: Wow... could this be any more confusing? Pull out into a
        # separate function. Times relative to recall start
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
                # TODO: possibly parametrize this
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

    return np.recarray(np.concatenate(all_events))


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


def concatenate_events_across_experiments(event_list):
    """
    Concatenate events across different experiment types. To make session
    numbers unique, 100 is added to the second set of events in event_list,
    200 to the next set of events, and so on.

    Parameters
    ----------
    event_list: iterable
        An iterable containing events to be concatenated

    Returns
    -------
    np.recarray
        The combined set of events

    """
    # Update sessions to not be in conflict
    session_offset = 0
    final_event_list = []
    for events in event_list:
        if len(events) == 0:
            continue # we don't want to be incrementing if we dont have to
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
    event_sizes = [len(events) for events in event_list]
    if sum(event_sizes) == 0:
        empty_events = initialize_empty_event_reccarray()
        return empty_events
    final_events = np.recarray(np.concatenate(event_list))
    final_events.sort(order=['session', 'list', 'mstime'])

    return final_events


def remove_intrusions(events):
    """
    Select all encoding events that were part of the encoding period or
    were non-intrusion retrieval events.

    """
    encoding_events_mask = get_encoding_mask(events)
    retrieval_event_mask = get_fr_retrieval_events_mask(events)
    baseline_retrieval_event_mask = get_baseline_retrieval_mask(events)

    mask = (encoding_events_mask |
            retrieval_event_mask |
            baseline_retrieval_event_mask)

    filtered_events = events[mask]
    events = np.recarray(filtered_events)
    return events


def select_word_events(events, encoding_only=True):
    """ Filter out any non-word events

    Parameters
    ----------
    events: np.recarray
    encoding_only: bool
        Flag for whether retrieval events should be included

    """
    encoding_events_mask = get_encoding_mask(events)
    retrieval_event_mask = get_all_retrieval_events_mask(events)

    if encoding_only:
        mask = encoding_events_mask
    else:
        mask = (encoding_events_mask | retrieval_event_mask)

    filtered_events = events[mask]
    events = np.recarray(filtered_events)

    return events


def extract_sample_rate(events):
    """ Extract the samplerate used for the given set of events"""
    eeg_reader = EEGReader(events=events[:2], start_time=0.0, end_time=1.0)
    eeg = eeg_reader.read()
    samplerate = float(eeg['samplerate'])
    return samplerate


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

    # No events to mask
    if len(mask) == 0:
        return mask

    if max(mask) is False:
        raise RuntimeError("No baseline retrieval events found. Create "
                           "baseline retrieval events first.")
    return mask


def select_retrieval_events(events):
    """
    Select retrieval events. Uses the experiment field in the events to
    determine how selection should be done since selection differes for PAL
    and FR/catFR

    Parameters
    ----------
    events: np.recarray
        Events to mask

    """
    experiments = np.unique(events.experiment)
    if len(experiments) > 1:
        raise RuntimeError("Retrieval event selection only supports "
                           "single-experiment datasets")
    experiment = experiments[0]

    if "FR" in experiment:
        mask = get_fr_retrieval_events_mask(events)

    elif "PAL"in experiment:
        mask = get_pal_retrieval_events_mask(events)

    retrieval_events = events[mask]

    return retrieval_events


def get_fr_retrieval_events_mask(events):
    """ Identify actual retrieval events for FR/catFR experiments"""
    # FIXME: Parametrize the inter-event threshold
    # TODO: Why don't we actually study this 1000ms threshold to optimize it?
    inter_event_times = get_time_between_events(events)
    retrieval_mask = ((events.type == 'REC_WORD') &
                      (events.intrusion == 0) &
                      (inter_event_times > 1000))
    return retrieval_mask


def get_pal_retrieval_events_mask(events):
    """ Identify retrieval events for PAL experiments """
    retrieval_mask = ((events.type == 'TEST_PROBE') |
                      (events.type == 'PROBE_START'))
    return retrieval_mask


def select_all_retrieval_events(events):
    """ Select both baseline and actual retrieval events """
    retrieval_mask = get_all_retrieval_events_mask(events)
    retrieval_events = events[retrieval_mask]
    return retrieval_events


def get_all_retrieval_events_mask(events):
    """ Create a boolean bask for any retrieval event """
    all_retrieval_mask = ((events.type == 'REC_WORD') |
                          (events.type == 'REC_BASE') |
                          (events.type == 'REC_EVENT'))
    return all_retrieval_mask


def partition_events(events):
    """
    Split a given set of events into partitions by experiment class (
    FR/PAL) and encoding/retrieval

    Parameters
    ----------
    events: np.recarray
        Set of events to partition

    Returns
    -------
    list
        A list containing all identified partitions to the data

    """

    retrieval_mask = get_all_retrieval_events_mask(events)
    pal_mask = (events.experiment == "PAL1")

    fr_encoding = events[(~retrieval_mask & ~pal_mask)]
    fr_retrieval = events[(retrieval_mask & ~pal_mask)]
    pal_encoding = events[(~retrieval_mask & pal_mask)]
    pal_retrieval = events[(retrieval_mask & pal_mask)]

    # Only add partitions with actual events
    final_partitions = {
        'fr_encoding': fr_encoding,
        'fr_retrieval': fr_retrieval,
        'pal_encoding': pal_encoding,
        'pal_retrieval': pal_retrieval
    }
    return final_partitions
