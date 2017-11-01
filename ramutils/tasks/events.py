import numpy as np
from dask import delayed

from ptsa.data.readers import BaseEventReader

from ReportTasks.RamTaskMethods import create_baseline_events  # FIXME
from . import memory as mem


def _filter_session(sess_events):
    try:
        last_list = sess_events[sess_events.type == 'REC_END'][-1]['list']  # drop any incomplete lists
        return sess_events[sess_events.list <= last_list]
    except IndexError:
        return sess_events


@delayed
@mem.cache
def read_fr_events(index, subject, sessions=None, cat=False):
    """Read FR events.

    :param JsonIndexReader index:
    :param str subject:
    :param list sessions: Sessions to read events from (all sessions if None)
    :param bool cat: True when reading CatFR events.
    :returns: list of events for each session

    """
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
            index.get_value('task_events', subject=subj_code, montage=montage,
                            experiment=exp, session=s)
            for s in sorted(use_sessions)
        ]
        events = [_filter_session(BaseEventReader(filename=event_path).read()) for event_path in files]
    else:
        files = sorted(
            list(index.aggregate_values('task_events', subject=subj_code,
                                        montage=montage, experiment=exp))
        )
        events = [_filter_session(BaseEventReader(filename=f).read()) for f in files]

    return events


@delayed
@mem.cache
def concatenate_events(fr_events, catfr_events):
    """Concatenate FR and CatFR events.

    :param list fr_events:
    :param list catfr_events:
    :returns: events recarray

    """
    types = [
        'item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment',
        'mstime', 'type', 'eegoffset',  'recalled', 'item_name', 'intrusion',
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


# FIXME: better name?
# FIXME: convert to task
def free_epochs(times, duration, pre, post, start=None, end=None):
    """Given a list of event times, find epochs between them when nothing is
    happening.

    Parameters
    ----------
    times : np.ndarray
        An iterable of 1-d numpy arrays, each of which indicates event times
    duration : int
        The length of the desired empty epochs
    pre : int
        the time before each event to exclude
    post: int
        The time after each event to exclude

    """
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


@delayed
@mem.cache
def create_baseline_events(events, start_time, end_time):
    """Match recall events to matching baseline periods of failure to recall.
    Baseline events all begin at least 1000 ms after a vocalization, and end at
    least 1000 ms before a vocalization. Each recall event is matched, wherever
    possible, to a valid baseline period from a different list within 3 seconds
    relative to the onset of the recall period.

    Parameters
    ----------
    events : np.recarray
        The event structure in which to incorporate these baseline periods
    start_time : int
        The amount of time to skip at the beginning of the session (ms)
    end_time : int
        The amount of time within the recall period to consider (ms)

    """
    # TODO: clean this mess up
    # TODO: document within code blocks what is actually happening

    all_events = []
    for session in np.unique(events.session):
        sess_events = events[(events.session == session)]
        irts = np.append([0], np.diff(sess_events.mstime))
        rec_events = sess_events[(sess_events.type == 'REC_WORD') & (sess_events.intrusion == 0) & (irts > 1000)]
        voc_events = sess_events[((sess_events.type == 'REC_WORD') | (sess_events.type == 'REC_WORD_VV'))]
        starts = sess_events[(sess_events.type == 'REC_START')]
        ends = sess_events[(sess_events.type == 'REC_END')]
        rec_lists = tuple(np.unique(starts.list))

        times = [voc_events[(voc_events.list == lst)].mstime if (voc_events.list==lst).any() else []
                 for lst in rec_lists]
        start_times = starts.mstime.astype(np.int)
        end_times = ends.mstime.astype(np.int)

        # FIXME: make epochs an input
        epochs = free_epochs(times, 500, 2000, 1000, start=start_times, end=end_times)
        rel_times = [(t - i)[(t - i > start_time) & (t - i < end_time)] for (t, i) in
                     zip([rec_events[rec_events.list == lst].mstime for lst in rec_lists ], start_times)
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
                    choice_position = np.argmin(np.mod(good_locs[0] - i, len(good_locs[0])))
                    choice_inds = (good_locs[0][choice_position], good_locs[1][choice_position])
                    full_match_accum[choice_inds] = True

        matching_epochs = epochs[full_match_accum]
        new_events = np.zeros(len(matching_epochs), dtype=sess_events.dtype).view(np.recarray)

        for i, _ in enumerate(new_events):
            new_events[i].mstime = matching_epochs[i]
            new_events[i].type = 'REC_BASE'

        new_events.recalled = 0
        merged_events = np.concatenate((sess_events, new_events)).view(np.recarray)
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


@delayed
@mem.cache
def select_word_events(events, include_retrieval=True):
    """Filter out non-WORD events.

    :param np.recarray events:
    :param bool include_retrieval: Include REC_WORD and REC_BASE events
    :return: filtered events recarray

    """
    # FIXME: note that input here should be after calling create_baseline_events
    # events = create_baseline_events(events, 1000, 29000)

    # FIXME: document what is going on here
    irts = np.append([0], np.diff(events.mstime))
    encoding_events_mask = events.type == 'WORD'
    retrieval_events_mask = (events.type == 'REC_WORD') | (events.type == 'REC_BASE')
    retrieval_events_mask_0s = retrieval_events_mask & (events.type == 'REC_BASE')
    retrieval_events_mask_1s = retrieval_events_mask & (events.type == 'REC_WORD') & (events.intrusion == 0) & (irts > 1000)

    # FIXME: is this necessary?
    if include_retrieval:
        mask = encoding_events_mask | retrieval_events_mask_0s | retrieval_events_mask_1s
    else:
        mask = encoding_events_mask

    filtered_events = events[mask]

    events = filtered_events.view(np.recarray)
    return events
