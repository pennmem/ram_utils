import numpy as np
from ptsa.data.readers import BaseEventReader

from . import task

__all__ = [
    'read_fr_events',
    'concatenate_events',
    'combine_events',
    'remove_negative_offsets',
    'remove_incomplete_lists',
    'insert_baseline_retrieval_events',
    'get_time_between_events',
    'select_word_events',
    'get_encoding_mask',
    'get_all_retrieval_events_mask',
    'select_encoding_events',
    'select_all_retrieval_events',
    'select_baseline_retrieval_events',
    'select_retrieval_events',
    'select_vocalization_events',
]


@task()
def read_fr_events(index, subject, sessions=None, cat=False):
    """Read FR events.

    :param JsonIndexReader index:
    :param str subject:
    :param list sessions: Sessions to read events from (all sessions if None)
    :param bool cat: True when reading CatFR events.
    :returns: list of events for each session

    """
    # FIXME: We really shouldn't need to be extracting montage from the
    # subject ID
    tmp = subject.split('_')
    montage = 0 if len(tmp) == 1 else int(tmp[1])
    subj_code = tmp[0]
    exp = 'catFR1' if cat else 'FR1'

    if sessions is not None:
        # Sessions are offset by 100 for CatFR sessions in order to distinguish
        # from regular FR sessions.
        # FIXME: see if there's a better way to handle this when we are concatenating FR with CatFR
        offset = 100 if cat else 0
        use_sessions = [s - offset for s in sessions if s < 100]

        files = [
            index.get_value('all_events', subject=subj_code, montage=montage,
                            experiment=exp, session=s)
            for s in sorted(use_sessions)
            ]

        # TODO: Don't do any event cleaning as part of a read() function...
        events = [remove_incomplete_lists(BaseEventReader(filename=event_path).read())
                  for event_path in files]
    else:
        files = sorted(
            list(index.aggregate_values('all_events', subject=subj_code,
                                        montage=montage, experiment=exp))
        )
        events = [remove_incomplete_lists(BaseEventReader(filename=f).read()) for f in
                  files]

    return events


@task()
def concatenate_events(fr_events, catfr_events):
    """Concatenate FR and CatFR events. To make session numbers unique, 100 is
    added to all CatFR sessions, otherwise this function could be made much
    simpler.

    :param list fr_events:
    :param list catfr_events:
    :returns: events recarray

    """
    types = [
        'item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment',
        'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion',
        'montage', 'list', 'eegfile', 'msoffset'
    ]

    if len(fr_events):
        fr_events = np.concatenate(fr_events).view(np.recarray)
    if len(catfr_events):
        catfr_events = np.concatenate(catfr_events).view(np.recarray)
        catfr_events = catfr_events[types].copy()
        catfr_events.session += 100

        if len(fr_events):
            fr_events = fr_events[types].copy()

    if len(fr_events) and len(catfr_events):
        events = np.append(fr_events, catfr_events).view(np.recarray)
    else:
        events = fr_events if len(fr_events) else catfr_events

    events = events[events.list > -1]
    return events


@task()
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


@task()
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


def remove_incomplete_lists(sess_events):
    """

        Identify any incomplete lists and remove those events from the set of
        session events

    """
    try:
        last_list = sess_events[sess_events.type == 'REC_END'][-1]['list']
        return sess_events[sess_events.list <= last_list]
    except IndexError:
        return sess_events


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
    if max(mask) == False:
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
    """ Select all retrieval events """
    retrieval_mask = get_all_retrieval_events_mask(events)
    retrieval_events = events[retrieval_mask]
    return retrieval_events


def get_all_retrieval_events_mask(events):
    """ Create a boolean bask for any retrieval event """
    all_retrieval_mask = ((events.type == 'REC_WORD') |
                          (events.type == 'REC_BASE'))
    return all_retrieval_mask


@task()
def combine_events(event_list):
    """ Combines a list of events into single recarray """
    events = np.concatenate(event_list).view(np.recarray)
    events.sort(order=['session', 'list', 'mstime'])
    return events


def remove_negative_offsets(events):
    """ Remove events with a negative eegoffset """
    pos_offset_events = events[events['eegoffset'] >= 0]
    return pos_offset_events


def remove_bad_events():
    raise NotImplementedError
