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


@delayed
@mem.cache
def select_word_events(events, include_retrieval=True):
    """Filter out non-WORD events.

    :param np.recarray events:
    :param bool include_retrieval: Include REC_WORD and REC_BASE events
    :return: filtered events recarray

    """
    events = create_baseline_events(events,1000,29000)
    irts = np.append([0],np.diff(events.mstime))

    encoding_events_mask = events.type == 'WORD'
    retrieval_events_mask = (events.type == 'REC_WORD') | (events.type == 'REC_BASE')
    retrieval_events_mask_0s = retrieval_events_mask & (events.type == 'REC_BASE')
    retrieval_events_mask_1s = retrieval_events_mask & (events.type == 'REC_WORD') & (events.intrusion == 0)  & (irts > 1000)

    # FIXME: is this necessary?
    if include_retrieval:
        mask = encoding_events_mask | retrieval_events_mask_0s | retrieval_events_mask_1s
    else:
        mask = encoding_events_mask

    filtered_events = events[mask]

    events = filtered_events.view(np.recarray)
    return events
