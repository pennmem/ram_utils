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
import json
import numpy as np
import pandas as pd

from itertools import groupby
from numpy.lib.recfunctions import rename_fields

from ptsa.data.readers import BaseEventReader, JsonIndexReader, EEGReader
from ramutils.utils import extract_subject_montage, get_completed_sessions, extract_experiment_series
from ramutils.exc import *
from ramutils.retrieval import create_matched_events,append_fields


def load_events(subject, experiment, file_type='all_events',
                sessions=None, rootdir='/'):
    """ Load events for a specific subject and experiment. If no events are
    found, an empty recarray with the correct datatypes are returned

    Parameters
    ----------
    subject: str
    experiment: str
    file_type: str
        The name of the event file to load, i.e. all_events, task_events,
        math_events, ps4_events. Default is 'all_events'
    sessions: iterable or None
    rootdir: str

    Returns
    -------
    np.rec.array
        A numpy recarray containing all events for the requested subject,
        experiment, and session(s)

    """
    rootdir = "/home1/jbruska/Code/ram_utils/" # TODO: JPB: Remove this

    print(f"load_events({subject}, {experiment}, {file_type}, {sessions}, {rootdir})")
    json_reader = JsonIndexReader(os.path.join(rootdir,
                                               "protocols",
                                               "r1.json"))

    if sessions is None:
        sessions = get_completed_sessions(subject, experiment,
                                          rootdir=rootdir)

    # Make sure sessions are integers since this is required to check the max
    sessions = [int(s) for s in sessions]

    # If the given sessions have offsets, then remove, otherwise leave them alone
    sessions_to_load = sessions
    if len(sessions_to_load) > 0:
        if max(sessions) >= 100:
            sessions_to_load = remove_session_number_offsets(
                experiment, sessions)

    event_files = []
    for session in sorted(sessions_to_load):
        try:
            event_file = json_reader.get_value(file_type,
                                               subject_alias=subject,
                                               experiment=experiment,
                                               session=session)
            event_files.append(event_file)

        # If an event file cannot be found for a session, skip that session
        except ValueError:
            continue

    event_files = sorted(event_files)

    # Update the paths based on the given root directory. This makes it easier
    # to run tests and use a mounted file system
    # 10.9.20: cmlreaders makes this obsolete, as
    # event_files = [os.path.join(rootdir, event_file) for event_file in
                   #event_files]

    if len(event_files) == 0:
        empty_recarray = initialize_empty_event_reccarray()
        return empty_recarray
    try:
        events = np.rec.array(np.concatenate([
            BaseEventReader(filename=f,
                            eliminate_events_with_no_eeg=True).read()
            for f in event_files]))
    except Exception:
        raise DataLoadingError(
            'Could not load events for %s, %s' % (subject, experiment))

    return events


def clean_events(events, start_time=None, end_time=None, duration=None,
                 pre=None, post=None, return_stim_events=False,
                 all_events=False):
    """
        Perform basic cleaning operations on events such as removing incomplete
        sessions, negative offset events, and incomplete lists. For FR events,
        baseline events needs to be found. Events are then normalized so that
        cross-experiment events can be merged.

    Parameters
    ----------
    events: np.recarray
        Raw events
    start_time: int
    end_time: int
    duration: int
    pre: int
    post: int
    return_stim_events: bool
        Indicator for if stim parameters should be returned in addition to the
        cleaned events
    all_events: bool
        Indicates if the data to be cleaned is the all_event.json file. These
        require a different set of cleaning procedures

    Returns
    -------
    np.recarray
        Cleaned set of events

    Notes
    -----
    This function should be called on an experiment by experiment basis and
    should not be used to clean cross-experiment datasets
    """

    experiments = extract_experiment_from_events(events)
    series_num = extract_experiment_series(experiments[0])

    if all_events:
        all_fields = list(events.dtype.names)
        if (series_num != "1") and (series_num is not None):
            if "stim_params" in all_fields:
                all_fields.remove('stim_params')
        if "test" in all_fields:
            all_fields.remove('test')
        all_events = events[all_fields].copy()
        return all_events

    # If you clean 'all_events' for joint reports, there will be multiple
    # experiments, so only check this after determining if you are cleaning
    # combined events
    if len(experiments) > 1:
        raise RuntimeError('Event cleaning can only happen on single-experiment'
                           ' datasets')
    experiment = experiments[0]

    events = remove_negative_offsets(events)

    events = remove_voice_detection(events)

    # needed for DBOY 1, normalization should be improved to
    # standardaze events without these ad hoc corrections
    events = rename_fields(events, {"trial": "list"})
    events = change_field_type(events, {"mstime": np.int32, "rectime": np.int32, "eegoffset": np.int32})

    # Only for PS5 do we want to keep the practice list around so we can know
    # what the baseline mean power was for the session, but we still need to get
    # rid of the events with -999 for list
    if all(['PS5' not in experiment for experiment in experiments]):
        events = remove_practice_lists(events)

    else:
        events = events[events.list >= -1]
    
    events = remove_incomplete_lists(events)

    # FIXME: simplify logic so subsetting only happens if combining experiments
    events = select_column_subset(events, all_relevant=True)

    # separate_stim_events is called within the task-specific functions
    # because the columns to subset differs by task
    if "FR" in experiment or 'DBOY' in experiment:
        events, stim_params = separate_stim_events(events)
        if "TICL" not in experiment:
            events = insert_baseline_retrieval_events(events,
                                                      start_time,
                                                      end_time,
                                                      duration,
                                                      pre,
                                                      post)
        events = remove_intrusions(events)
        events = update_recall_outcome_for_retrieval_events(events)
        events = normalize_fr_events(events)

    elif "PAL" in experiment:
        events, stim_params = separate_stim_events(events, pal=True)
        events = subset_pal_events(events)
        events = update_pal_retrieval_events(events)
        events = remove_nonresponses(events)
        events = normalize_pal_events(events)

    else:
        stim_params = initialize_empty_stim_reccarray()

    events = update_subject(events)

    if return_stim_events:
        return events, stim_params

    return events


def remove_session_number_offsets(experiment, sessions):
    """
        Given a list of sessions to include, undo the offsets for catFR and
        PAL so the sessions can be looked up correctly in the r1.json file
    """
    if sessions is None:
        return sessions

    elif experiment.find("PAL") != -1:
        relevant_sessions = [(sess - 200) for sess in sessions if sess >= 200]

    elif experiment.find("cat") != -1:
        relevant_sessions = [(sess - 100)
                             for sess in sessions if (sess >= 100 and sess < 200)]

    elif experiment.find("FR") != -1:
        relevant_sessions = [sess for sess in sessions if sess < 100]

    elif experiment.find("PS") != -1:
        relevant_sessions = [sess for sess in sessions if sess < 100]

    else:
        raise RuntimeError(
            "Only Fr/catFR/PAL session numbering with offsets is supported")
    return relevant_sessions


def update_subject(events):
    """ Ensure subject field is populated for all events """
    subject = extract_subject(events)
    events.subject = subject
    return events


def normalize_fr_events(events):
    events = combine_retrieval_events(events)

    if 'category_num' not in events.dtype.names:
        events = add_field(events, 'category_num', 999, '<i8')

    if 'phase' not in events.dtype.names:
        events = add_field(events, 'phase', '', '<U256')

    events = select_column_subset(events, cat=True)

    return events


def normalize_pal_events(events):
    """
        Perform any normalization to PAL event so make the homogeneous enough so
        that it is trivial to combine with other experiment events.
    """
    events = rename_correct_to_recalled(events)
    events = coerce_study_pair_to_word_event(events)

    if 'phase' not in events.dtype.names:
        events = add_field(events, 'phase', '', '<U256')
    if 'matched' not in events.dtype.names:
        events = add_field(events, 'matched', True, np.bool_)
    events = add_field(events, 'item_name', 'X', '<U256')
    events = add_field(events, 'category_num', 999, '<i8')

    return events


def separate_stim_events(events, pal=False, stim=True, cat=False):
    """ Separate stim params contained within events structure from the 1-D
        events. The returned events and stim_params are both 1-dimensional

    Parameters
    ----------
    pal
    stim
    cat
    events: np.recarray
        Event structure

    Return
    ------
    events: np.reccary
        1D event structure with stim params removed
    stim_params: np.recarray
        2D stim params strsucture

    """
    # Short-circuit if no stim params field (non stim experiment) or no events
    if (len(events) == 0) or ('stim_params' not in events.dtype.names):
        stim_params = initialize_empty_stim_reccarray()
        return events, stim_params

    stim_cols = get_required_columns(pal=pal, stim=stim, cat=cat)
    all_cols = get_required_columns(pal)
    all_fields = list(events.dtype.names)

    # Historically, some event files do not have a phase field, but we need
    # it if it is there
    stim_cols = [col for col in stim_cols if col in all_fields]

    all_fields.remove('stim_params')

    stim_params = events[stim_cols]
    events = events[all_fields]

    return events, stim_params


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

def change_field_type(events, mapping_dict):
    return events.astype([(d[0], mapping_dict[d[0]]) if d[0] in mapping_dict else d for d in events.dtype.descr])


def add_field(events, field_name, default_val, dtype):
    """ Add field to the recarray

    Notes
    -----
    Converting to a dataframe, adding the field, and reconverting to a
    recarray because the rec_append_fields function in numpy doesn't seem to
    work

    """
    events_df = pd.DataFrame(events)
    events_df[field_name] = default_val
    orig_dtypes = build_dtype_list(events.dtype)

    # Add the given field and type to dtype list
    orig_dtypes.append((field_name, dtype))
    events = dataframe_to_recarray(events_df, orig_dtypes)
    return events


def build_dtype_list(dtypes):
    """
        Given a numpy.dtype object, return a list of tuples in the form
        (field_name, field_type_string)
    """
    names = dtypes.names
    dtype_list = []
    for i in range(len(dtypes)):
        dtype_list.append((names[i], dtypes[i].str))

    return dtype_list


def dataframe_to_recarray(dataframe, dtypes):
    """
        Convert from dataframe to recarray maintaining the original datatypes
    """
    names = [dt[0] for dt in dtypes]
    events = dataframe.to_records(index=False)
    # Make sure that all the columns are in the correct order
    events = events[names].astype(dtypes)
    events.dtype.names = [str(name) for name in events.dtype.names]
    return events


def remove_negative_offsets(events):
    """ Remove events with a negative eegoffset """
    pos_offset_events = events[events['eegoffset'] >= 0]
    return pos_offset_events

def remove_voice_detection(events):
    """ Remove events
    """
    return events[['VOCALIZATION' not in ev['type'] for ev in events]]


def lookup_sample_rate(subject, experiment, session, rootdir="/"):
    """ Identify the sample rate used for a session """
    base_path = os.path.join(rootdir,
                             'protocols/r1/subjects/{subject}/experiments/{experiment}/sessions/{session}/ephys/current_processed/sources.json')

    with open(base_path.format(subject=subject, experiment=experiment,
                               session=session)) as f:
        contents = json.load(f)
        first_key = list(contents.keys())[0]
        sample_rate = contents[first_key]['sample_rate']

    return sample_rate


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
        final_sess_events.sort(order=['session', 'list', 'eegoffset'])
        print("math", math_events)

        # Remove all task events for lists that don't have a "REC_END" event
        events_by_list = (np.array([l for l in list_group]) for listno,
                          list_group in
                          groupby(final_sess_events, lambda x: x.list))
        list_has_end = [any([l['type'] in ['REC_START'] for l in list_group]) or # TODO: JPB: Remove this
        #list_has_end = [any([l['type'] in ['REC_END', 'REC_STOP'] for l in list_group]) or
                        listno == -999 for listno, list_group in groupby(
            final_sess_events, lambda x:x.list)]

        final_sess_events = np.concatenate([e for (e, a) in zip(
            events_by_list, list_has_end) if a])

        # Re-combine math and task events
        final_sess_events = np.rec.array(np.concatenate([final_sess_events,
                                                         math_events]))
        final_sess_events.sort(order=['session', 'list', 'eegoffset'])
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
    samplerate = 1000  # extract_sample_rate(events)

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
    events.recalled[events.type == 'REC_WORD'] = 1
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


def select_column_subset(events, all_relevant=False, pal=False, stim=False,
                         cat=False):
    """ Select only the necessary subset of the fields

    Parameters
    ----------
    events: np.recaarray
        The set of events to subset from

    Keyword Arguments
    -----------------
    all_relevant: bool
        A subset that includes all fields that are subsequently used by any
        of the experiments
    pal: bool
        Fields specific to PAL experiments
    stim: bool
        Fields specific to stim experiments
    cat: bool
        Fields specific to categorical free recall experiments
    """
    columns = get_required_columns(all_relevant=all_relevant, pal=pal,
                                   stim=stim, cat=cat)

    # Not all columns will always be available. This in handled during event
    # normalization, so column selection should allow for the non-existence
    # of a desired column
    final_columns = []
    for col in columns:
        if col in events.dtype.names:
            final_columns.append(col)

    # Explicitly ask for a copy since a view is returned in numpy 1.13 and later
    events = events[final_columns].copy()

    return events


def get_required_columns(all_relevant=False, pal=False, stim=False, cat=False):
    """ Return baseline mandatory columns based on experiment type

    Keyword Arguments
    -----------------
    all_relevant: bool
        A subset that includes all fields that are subsequently used by any
        of the experiments
    pal: bool
        Fields specific to PAL experiments
    stim: bool
        Fields specific to stim experiments
    cat: bool
        Fields specific to categorical free recall experiments
    """

    # FIXME: This would probably be better as just a dictionary
    if all_relevant and any([pal, stim, cat]):
        raise RuntimeError('all cannot be chosen in conjunction with other '
                           'options')

    if cat & pal:
        raise RuntimeError('cat and pal cannot be selected at the same time')

    # FIXME: this hack isn't a substitute for doing this right in event_filter
    columns = [
        'serialpos', 'session', 'subject', 'rectime', 'experiment',
        'mstime', 'type', 'eegoffset', 'recalled', 'intrusion',
        'montage', 'list', 'stim_list', 'eegfile', 'msoffset', 'item_name',
        'iscorrect', 'phase', 'matched', 'is_repeat', 'repeats', 'item', 'trial'
    ]

    if all_relevant:
        columns.append('stim_params')
        columns.append('correct')
        columns.append('category_num')
        columns.append('RT')
        return columns

    if stim:
        columns = ['subject', 'experiment', 'session', 'list',
                   'stim_list', 'mstime', 'eegoffset', 'item_name',
                   'serialpos', 'type', 'phase', 'stim_params', 'recalled']

    if cat:
        columns.append('category_num')

    if pal:
        columns.remove('item_name')
        columns.remove('recalled')
        columns.append('correct')

    return columns


def initialize_empty_event_reccarray():
    """Utility function for generating a recarray that looks normalized,
    but is empty.

    """
    empty_recarray = np.recarray((0, ), dtype=[('serialpos', '<i8'),
                                               ('session', '<i8'),
                                               ('subject', '<U256'),
                                               ('rectime', '<i8'),
                                               ('experiment', '<U256'),
                                               ('mstime', '<i8'),
                                               ('type', '<U256'),
                                               ('eegoffset', '<i8'),
                                               ('recalled', '<i8'),
                                               ('intrusion', '<i8'),
                                               ('montage', '<i8'),
                                               ('list', '<i8'),
                                               ('stim_list', '<i8'),
                                               ('phase', '<U256'),
                                               ('eegfile', '<U256'),
                                               ('msoffset', '<i8'),
                                               ('item_name', '<U256'),
                                               ('iscorrect', '<i8')])
    return empty_recarray


def initialize_empty_stim_reccarray():
    """ Generate empty recarray that mirrors fields in stim_params """
    empty_recarray = np.recarray((0, ), dtype=[('serialpos', '<i8'),
                                               ('session', '<i8'),
                                               ('subject', '<U256'),
                                               ('experiment', '<U256'),
                                               ('mstime', '<i8'),
                                               ('eegoffset', '<i8'),
                                               ('type', '<U256'),
                                               ('recalled', '<i8'),
                                               ('list', '<i8'),
                                               ('stim_list', '<i8'),
                                               ('phase', '<U256'),
                                               ('item_name', '<U256'),
                                               ('stim_params', '<U256')])
    return empty_recarray


def insert_baseline_retrieval_events(events, start_time, end_time, duration,
                                     pre, post,
                                     use_deprecated=False):
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
    if use_deprecated:
        return insert_baseline_retrieval_events_deprecated(
            events, start_time, end_time, duration, pre, post
        )
    else:
        return insert_baseline_retrieval_events_logan(events,
                                                      duration,
                                                      pre,
                                                      post)

def insert_baseline_retrieval_events_logan(events,duration,pre,post):
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

    all_events = []
    for experiment in extract_experiment_from_events(events):
        exp_events = events[events['experiment'] == experiment]
        for session in extract_sessions(exp_events):
            sess_events = select_session_events(events,session)
            samplerate = extract_sample_rate_from_eeg(sess_events)
            new_events = create_matched_events(
                sess_events,
                samplerate=samplerate,
                rec_inclusion_before=1000,
                rec_inclusion_after=1000,
                recall_eeg_start=-1*duration,
                recall_eeg_end=0,
                remove_before_recall=pre,
                remove_after_recall=post,
            )
            event_fields = list(sess_events.dtype.names)
            new_events = new_events[event_fields][:]
            is_matched_rec_word = np.in1d(
                sess_events[sess_events.type == 'REC_WORD'],
                new_events[new_events.type == 'REC_WORD'])
            new_events = append_fields(new_events, [('matched',np.bool_)])
            new_events['matched'] = True
            sess_events = append_fields(sess_events, [('matched', np.bool_)])
            sess_events['matched'] = False
            rec_events = sess_events[sess_events.type == 'REC_WORD']
            rec_events['matched'] = is_matched_rec_word
            sess_events[sess_events.type == 'REC_WORD'] = rec_events
            all_events.append(
                concatenate_events_for_single_experiment(
                    [sess_events,new_events[new_events.type == 'REC_BASE']])
            )
    return concatenate_events_for_single_experiment(all_events)


def insert_baseline_retrieval_events_deprecated(
        events, start_time, end_time, duration, pre, post):
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

    # We need to know the sample rate in order to create the new REC_BASE
    # events. Rather than load a file, we can just load a snippet of eeg and
    # back out the sample rate
    samplerate = extract_sample_rate_from_eeg(events)

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
                # For each recall event, reject everything that is more than
                # three seconds away
                is_match_tmp = np.abs((rel_epochs - t)) < 3000
                is_match_tmp[i, ...] = False
                good_locs = np.where(is_match_tmp & (~full_match_accum))
                if len(good_locs[0]):
                    # Find next closest list with a valid deliberation period
                    choice_position = np.argmin(
                        np.mod(good_locs[0] - i, len(good_locs[0])))
                    choice_inds = (good_locs[0][choice_position],
                                   good_locs[1][choice_position])
                    full_match_accum[choice_inds] = True

        matching_epochs = epochs[full_match_accum]
        new_events = np.rec.array(np.zeros(len(matching_epochs),
                                           dtype=sess_events.dtype))

        for i, _ in enumerate(new_events):
            new_events[i].mstime = matching_epochs[i]
            new_events[i].type = 'REC_BASE'

        new_events.recalled = 0
        merged_events = np.rec.array(np.concatenate((sess_events,
                                                     new_events)))
        merged_events.sort(order='mstime')

        for (i, event) in enumerate(merged_events):
            if event.type == 'REC_BASE':
                merged_events[i].experiment = merged_events[i - 1].experiment
                merged_events[i].session = merged_events[i - 1].session
                merged_events[i].list = merged_events[i - 1].list
                merged_events[i].eegfile = merged_events[i - 1].eegfile
                elapsed_time_sec = (merged_events[i].mstime -
                                    merged_events[i - 1].mstime) / 1000.0
                samples_elapsed = samplerate * elapsed_time_sec
                merged_events[i].eegoffset = (merged_events[i - 1].eegoffset +
                                              samples_elapsed)
        merged_events = append_fields(merged_events,[('matched',np.bool_)])
        merged_events['matched']=False
        merged_events[(merged_events['type']=='REC_WORD') |
                      (merged_events['type']=='REC_BASE')]['matched']=True
        all_events.append(merged_events)

    return np.rec.array(np.concatenate(all_events))


def find_free_time_periods(times, duration, pre, post, start=None, end=None):
    """
    Given a list of event times, find epochs between them when nothing is
    happening.

    Parameters
    ----------
    times : list where elements are lists
        An iterable of 1-d numpy arrays, each of which is a list that
        indicates the starting times of all vocalization events. We do not
        want to include these as candidate time periods
    duration : int
        The length of the desired empty epochs
    pre : int
        the time before each event to exclude
    post: int
        The time after each event to exclude
    start: array_like
        List of a recall period start times
    end: array_like
        List of recall period end times

    Returns
    -------
    epoch_array : np.ndarray

    """
    # TODO: Do not allow start and end to be optional because bad stuff will
    # happen
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

        # FIXME: Is this backwards?
        interval_durations = pre_times[1:] - post_times[:-1]
        free_intervals = np.where(interval_durations > duration)[0]
        # For each word event, attempt to find a set of possible deliberation
        # periods in the recall phase
        trial_epoch_times = []
        for interval in free_intervals:
            begin = post_times[interval]
            finish = pre_times[interval + 1] - duration
            interval_epoch_times = range(
                int(begin), int(finish), int(duration))
            trial_epoch_times.extend(interval_epoch_times)
        epoch_times.append(np.array(trial_epoch_times))

    epoch_array = np.empty((n_trials, max([len(x) for x in epoch_times])))
    epoch_array[...] = -np.inf
    for i, epoch in enumerate(epoch_times):
        epoch_array[i, :len(epoch)] = epoch

    return epoch_array



def concatenate_events_across_experiments(event_list, pal=False, stim=False,
                                          cat=False):
    """
    Concatenate events across different experiment types. To make session
    numbers unique, 100 is added to the second set of events in event_list,
    200 to the next set of events, and so on.

    Parameters
    ----------
    event_list: iterable
        An iterable containing events to be concatenated
    pal: Bool
        Indicator for if PAL sessions are included in event_list. This will
        alter which columns are kept for merging events
    stim: Bool
        Indicator for if event_list contains stim sessions. If True,
        then stim_params field will be kept

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
            continue  # we don't want to be incrementing if we dont have to
        events.session += session_offset
        events = select_column_subset(events, pal=pal, stim=stim, cat=cat)
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
    final_events = np.rec.array(np.concatenate(event_list))
    final_events.sort(order=['subject', 'session', 'list',
                             'eegoffset'])

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
    events = np.rec.array(filtered_events)
    return events


def select_word_events(events, encoding_only=True):
    """ Filter out any non-word events

    Parameters
    ----------
    events: np.recarray
    encoding_only: bool
        Flag for whether retrieval events should be included

    """
    mask = get_word_event_mask(events, encoding_only=encoding_only)
    filtered_events = events[mask]
    events = np.rec.array(filtered_events)

    return events


def get_word_event_mask(events, encoding_only):
    """ Get a mask identify word events. If encoding_only, then retrieval
    events will not be counted """
    encoding_events_mask = get_encoding_mask(events)
    retrieval_event_mask = get_all_retrieval_events_mask(events)

    if encoding_only:
        mask = encoding_events_mask
    else:
        mask = (encoding_events_mask | retrieval_event_mask)

    return mask


def extract_event_metadata(events):
    """ Extract the subject, experiment(s), and session(s) associated with an
    event structure """
    subject = extract_subject(events)
    experiments = extract_experiment_from_events(events)
    experiment = ",".join(experiments)
    sessions = extract_sessions(events)
    return subject, experiment, sessions


def extract_subject(events, add_localization=False):
    """ Extract subject identifier from events """
    subjects = np.unique(events[events.subject != u''].subject).tolist()
    if len(subjects) > 1:
        raise RuntimeError('There should only be one subject in an event '
                           'recarray')
    if len(subjects) == 0:
        subject = ''

    else:
        subject = subjects[0]

    if add_localization:
        montage = np.unique(events[events.montage != ''].montage).tolist()
        if montage[0] != '0.0':
            montage_id = montage[0][2]
            subject = "_".join([subject, montage_id]) # this follows our naming convention
    return subject


def extract_experiment_from_events(events):
    """ Given a set of events, return a list of unique experiments contained
        within
    """
    # Experiment field can be blank, so make sure to not include that in the
    # final list
    experiments = np.unique(
        events[events.experiment != ''].experiment).tolist()

    # Handle the case of empty events being passed
    if len(events) == 0:
        experiments = ['']

    return experiments


def extract_sessions(events):
    """ Return a list of sessions contained within the events structure"""
    sessions = np.unique(events.session)
    sessions = [int(sess) for sess in sessions]
    return sessions


def extract_lists(events):
    """ Return a list of lists contained within the events structure """
    lists = np.unique(events.list)
    return lists


def select_session_events(events, session):
    """ Select events corresponding to a particular session """
    sessions = extract_sessions(events)
    if session not in sessions:
        raise RuntimeError('Session {} not in event structure'.format(session))

    session_event_mask = get_session_mask(events, session)
    session_events = events[session_event_mask]

    return session_events


def get_session_mask(events, session):
    """ Return a mask for if an event belongs to the given session """
    session_mask = (events.session == session)
    return session_mask


def select_stim_table_events(events):
    """ Return the events needed to build stim session summaries """
    events = remove_practice_lists(events)
    mask = get_stim_table_event_mask(events)
    stim_table_events = events[mask]
    return stim_table_events


def get_stim_table_event_mask(events):
    """
        Return a mask of events to be included for building stim session
        summaries
    """
    excluded_event_types = ['START', 'STOP', 'PROB']
    event_type_mask = [
        event.type not in excluded_event_types for event in events]

    return event_type_mask


def get_stim_list_mask(events):
    """ Return boolean mask identifying stim lists

    Notes
    -----
    Not all items in a stim list will be stimulated. Stimulation will depend
    on the biomarker at the time of encoding

    """
    stim_list_mask = (events.phase == 'STIM')
    return stim_list_mask


def add_list_phase_info(events):
    """
    Adds a list_phase field to an event structure that says which
    phase of the list (ENCODING, DISTRACT, RETRIEVAL,...) that event is
    part of.

    Parameters
    ----------
    events: np.recarray
        All event or task events

    """
    if 'list_phase' in events.dtype.names:
        return events

    dtype_desc = events.type.dtype.str

    phases = np.empty(len(events), dtype=dtype_desc)

    lstphases = []
    for ev in events:
        if ev['type'].endswith('_START'):
            lstphases.append(ev['type'].rpartition('_')[0])
        elif 'TRIAL' in ev['type']:
            lstphases.append(ev['type'].replace('TRIAL','ENCODING'))
        elif lstphases:
            lstphases.append(lstphases[-1])
        else:
            lstphases.append('')
    phases[:] = lstphases

    new_events = append_fields(events, [('list_phase', dtype_desc)])
    new_events['list_phase'] = phases

    # Certain special cases for older experiments

    word_events = new_events[(new_events.type == 'WORD')
                             | (new_events.type == 'PRACTICE_WORD')]
    word_events.list_phase = [t.replace('WORD', 'ENCODING')
                              for t in word_events.type]
    new_events[(new_events.type == 'WORD')
               | (new_events.type == 'PRACTICE_WORD')] = word_events

    retrieval_events = new_events[(new_events.list_phase == 'REC')
                                  | (new_events.type == 'REC_WORD') ]
    retrieval_events.list_phase = 'RETRIEVAL'
    new_events[(new_events.list_phase == 'REC')
               | (new_events.type == 'REC_WORD')] = retrieval_events
    return new_events


def extract_stim_information(all_events, task_events):
    """ Identify stim items, post stim items, and stimulation parameters

    Parameters
    ----------
    all_events: np.recarray
        All events with stim_params field
    task_events: np.recarray
        Task events used for classifier training/evaluation

    Returns
    -------
    is_stim_item: list
        Boolean array matching the length of task events indicating if a
        word was stimulated
    is_post_stim_item: list
        Boolean array matching the length of task_events indicating if a word
        occured after a stimulated word
    stim_df: pd.DataFrame
        Stim parameters used for each stimulation event


    Notes
    -----
    This is a rather convoluted set of logic. The goal is to match all word
    encoding events with their associated STIM_ON events, which occur as
    separate entries in the json event structures.

    """
    n_events = len(task_events)
    is_stim_item = np.zeros(n_events, dtype=np.bool)
    is_post_stim_item = np.zeros(n_events, dtype=np.bool)

    stim_param_data = {
        'item_name': [],
        'session': [],
        'list': [],
        'amplitude': [],
        'pulse_freq': [],
        'stim_duration': [],
        'stimAnodeTag': [],
        'stimCathodeTag': [],
    }

    lists = extract_lists(all_events)
    for lst in lists:
        lst_events = all_events[all_events.list == lst]
        lst_stim_words = np.zeros(len(lst_events[lst_events.type == 'WORD']))
        lst_post_stim_words = np.zeros(
            len(lst_events[lst_events.type == 'WORD']))

        # j will track word (task) events, while i tracks all events
        j = 0
        for i, event in enumerate(lst_events):
            if event.type == 'WORD':
                # Messy logic to find stim items
                if ((lst_events[i + 1].type == 'STIM_ON')
                        or (lst_events[i + 1].type == 'WORD_OFF' and
                            (lst_events[i + 2].type == 'STIM_ON' or (
                                lst_events[i + 2].type == 'DISTRACT_START'
                                and lst_events[i + 3].type == 'STIM_ON')))):
                    lst_stim_words[j] = True
                    # Identify which post 'WORD' event was the 'STIM_ON'
                    # event and use the stored stim params for that event to
                    # update the stim table
                    for offset in range(1, 4):
                        if lst_events[i + offset].type == 'STIM_ON':
                            # Assign stim params
                            loc = i + offset

                            # Single-site stimulation will have stim_param
                            # field as a record, while multi-site will be
                            # ndarray. Coerce everything to ndarray for
                            # consistency
                            stim_params = lst_events[loc].stim_params
                            if type(stim_params) != np.ndarray:
                                stim_params = np.array([stim_params])

                            # TODO: Add location field to stim params by
                            # looking up the contacts in the pairs metadata
                            # table, which would need to be passed to this
                            # function

                            stim_param_data['item_name'].append(
                                lst_events[loc].item_name)
                            stim_param_data['session'].append(
                                lst_events[loc].session)
                            stim_param_data['list'].append(
                                lst_events[loc].list)
                            stim_param_data['amplitude'].append(",".join(
                                [str(stim_params[k].amplitude / 1000.0) for k in range(len(stim_params))]))
                            stim_param_data['pulse_freq'].append(
                                ",".join([str(stim_params[k].pulse_freq) for k in range(len(stim_params))]))
                            stim_param_data['stim_duration'].append(
                                ",".join([str(stim_params[k].stim_duration) for k in range(len(stim_params))]))
                            stim_param_data['stimAnodeTag'].append(
                                ",".join([str(stim_params[k].anode_label) for k in range(len(stim_params))]))
                            stim_param_data['stimCathodeTag'].append(
                                ",".join([str(stim_params[k].cathode_label) for k in range(len(stim_params))]))
                            break

                # Post stim words are always the word after a stim word,
                # so just shift to find them
                if j > 0:
                    lst_post_stim_words[j] = lst_stim_words[j - 1]
                j += 1

        # FYI: It should always be the case that the number of word events
        # from all_events.json is equal to the number of events from
        # task_events.json. However, when reading the eeg as part of
        # computing powers, the PTSA EEGReader can elect to remove some
        # events. If it happens to remove a 'WORD' event, then these two
        # values could differ.
        lst_mask = (task_events.list == lst)
        if sum(lst_mask) != len(lst_stim_words):
            new_mask = np.in1d(lst_events[lst_events.type == 'WORD'].item_name,
                               task_events[lst_mask].item_name)

            lst_stim_words = lst_stim_words[new_mask]
            lst_post_stim_words = lst_post_stim_words[new_mask]
            # TODO: Do we need to do this correction for the stim param data
            # as well?

        is_stim_item[lst_mask] = lst_stim_words
        is_post_stim_item[lst_mask] = lst_post_stim_words

    stim_df = pd.DataFrame.from_dict(stim_param_data)

    return is_stim_item, is_post_stim_item, stim_df

def extract_biomarker_information(events):
    biomarker_events = events[events['type']=='BIOMARKER']
    biomarker_df = pd.DataFrame(columns=['position', 'phase',
                                        'biomarker_value', 'id'])
    biomarker_df['phase'] = biomarker_events['phase']
    biomarker_df['position'] = biomarker_events['stim_params']['position']
    biomarker_df['biomarker_value'] = biomarker_events['stim_params']['biomarker_value']
    biomarker_df['id'] = biomarker_events['stim_params']['id']
    biomarker_dtypes = [('position', 'U64'),
                        ('phase', 'U64'),
                        ('biomarker_value', float),
                        ('id', 'U64')]
    return dataframe_to_recarray(biomarker_df,biomarker_dtypes)


def correct_fr2_stim_item_identification(stim_param_df):
    """ Update the stim_item and post_stim_item masks for FR2 stim experiments

    The FR2 experiment is a special bird in that stimulation occurs
    across two items at a time, and therefore only a single STIM_ON
    event is recorded. This causes the stim item identification
    algorithm to miss those second items and therefore they must be corrected
    separately

    Parameters:
    -----------
    stim_param_df: `pd.DataFrame`
        Table containing the fully processed encoding events

    Returns
    -------
    stim_param_df: `pd.DataFrame`
        DataFrame with corrected is_stim_item and is_post_stim_item fields

    """
    updated_is_stim_item = [0] * len(stim_param_df)
    updated_is_post_stim_item = [0] * len(stim_param_df)

    for index, row in stim_param_df.iterrows():
        if row['is_stim_item'] == 1:
            updated_is_stim_item[index] = 1

        if row['is_post_stim_item'] == 1:
            updated_is_post_stim_item[index] = 1

        if ((row['experiment'] == 'FR2') or (
                row['experiment'] == 'catFR2')) and (row['is_stim_item'] == 1):
            # Only items from the same list should be counted as stim items or
            # post stim items. Ensure this by checking serial position
            updated_is_stim_item[index] = 1

            if row['serialpos'] < 12:
                updated_is_stim_item[index + 1] = 1
                updated_is_post_stim_item[index + 1] = 1
            if row['serialpos'] < 11:
                updated_is_post_stim_item[index + 2] = 1

    stim_param_df['is_stim_item'] = updated_is_stim_item
    stim_param_df['is_post_stim_item'] = updated_is_post_stim_item

    return stim_param_df


def validate_single_experiment(events):
    """ Raises an error if more than one experiment is present in the events """
    experiments = extract_experiment_from_events(events)
    if len(experiments) > 1:
        raise TooManyExperimentsError('Expected single experiment in events')
    return


def validate_single_session(events):
    """ Raises an error if more than one session is present in the events """
    sessions = np.unique(events.session)
    if len(sessions) > 1:
        raise TooManySessionsError("Expected single session events")
    return


def extract_sample_rate_from_eeg(events):
    """ Extract the samplerate used for the given set of events by loading EEG """

    eeg_reader = EEGReader(events=events[:2], start_time=0.0, end_time=1.0)
    eeg = eeg_reader.read()
    samplerate = float(eeg['samplerate'])
    return samplerate


def select_math_events(events):
    """ Select math events from a set of events """
    math_event_mask = get_math_events_mask(events)
    math_events = events[math_event_mask]
    return math_events


def get_math_events_mask(events):
    """ Get a boolean array identifying math events """
    math_event_mask = (events.type == 'PROB')
    return math_event_mask


def get_nonstim_events_mask(events):
    """ Get a mask of any non-stim WORD events

    Notes
    -----
    These events are what is used in post-hoc classifier evaluation
    """
    non_stim_mask = (events.type == 'WORD') & (events.phase != 'STIM')
    return non_stim_mask


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

    if 'matched' not in events.dtype.names:
        return all_retrieval_mask
    matched_mask = events['matched']
    return matched_mask & all_retrieval_mask



def get_recall_events_mask(events):
    """ Create a boolean mask for any recall events """
    recall_mask = (events.recalled == 1) #& (events.type == 'WORD') #& (events.is_repeat == False)
    return recall_mask

def get_non_recall_events_mask(events):
    """ Create a boolean mask for any recall events """
    recall_mask = (events.recalled != 1) #& (events.type == 'WORD') #& (events.is_repeat == False)
    return recall_mask


def get_post_stim_events_mask(events):
    # In general, there are no stim events during practice lists, but when artifact detection
    # is enabled, the STIM_OFF events have -999 for the list number, so this will be sure
    # to exclude those stim events
    post_stim_events_mask = ((events.type == 'STIM_OFF') & (events.list > -1))
    return post_stim_events_mask


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
    post_stim_mask = get_post_stim_events_mask(events)

    fr_encoding = events[(~retrieval_mask & ~pal_mask & ~post_stim_mask)]
    fr_retrieval = events[(retrieval_mask & ~pal_mask)]
    pal_encoding = events[(~retrieval_mask & pal_mask)]
    pal_retrieval = events[(retrieval_mask & pal_mask)]
    post_stim = events[post_stim_mask]

    # Only add partitions with actual events
    final_partitions = {
        'fr_encoding': fr_encoding,
        'fr_retrieval': fr_retrieval,
        'pal_encoding': pal_encoding,
        'pal_retrieval': pal_retrieval,
        'post_stim': post_stim
    }
    return final_partitions


def get_partition_masks(events):
    """
        Return a set of masks corresponding to the partitions present in the
        events

    """
    retrieval_mask = get_all_retrieval_events_mask(events)
    pal_mask = (events.experiment == "PAL1")
    post_stim_mask = get_post_stim_events_mask(events)

    fr_encoding = (~retrieval_mask & ~pal_mask)
    fr_retrieval = (retrieval_mask & ~pal_mask)
    pal_encoding = (~retrieval_mask & pal_mask)
    pal_retrieval = (retrieval_mask & pal_mask)
    post_stim = post_stim_mask

    # Only add partitions with actual events
    partition_masks = {
        'fr_encoding': fr_encoding,
        'fr_retrieval': fr_retrieval,
        'pal_encoding': pal_encoding,
        'pal_retrieval': pal_retrieval,
        'post_stim': post_stim
    }

    return partition_masks


def get_repetition_ratio_dict(rootdir="/"):
    all_repetition_rates = {}
    all_catfr1_subjects = find_subjects("catFR1", rootdir=rootdir)
    for i, subject in enumerate(all_catfr1_subjects):
        #if (subject not in ["R1617S", 'R1640T']):
        # TODO: JPB: Remove this
        if (subject not in ["R1642J"]):
            continue
        events = load_events(subject, "catFR1", file_type='task_events',
                             rootdir=rootdir)
        print(f"{i+1}/{len(all_catfr1_subjects)} {round(100*(i+1)/len(all_catfr1_subjects), 2):.2f}% {subject}" )

        recall_events = events[events.recalled == 1]
        sessions = np.unique(recall_events.session)
        lists = np.unique(recall_events.list)

        # Initialize single subject repetition rates of shape n_sessions X
        # n_lists
        repetition_rates = np.empty([len(sessions), len(lists)])

        for i, r in enumerate(repetition_rates.flat):
            repetition_rates.flat[i] = np.nan

        for i, session in enumerate(sessions):
            sess_recalls = recall_events[recall_events.session == session]
            lists = np.unique(sess_recalls.list)
            repetition_rates[i][:len(lists)] = [
                calculate_repetition_ratio(sess_recalls[sess_recalls.list ==
                                                        l]) for l in lists]
        all_repetition_rates[subject] = repetition_rates.copy()

    return all_repetition_rates


def find_subjects(experiment, rootdir="/"):
    """ Identify subjects who completed a given experiment """
    json_reader = JsonIndexReader(os.path.join(rootdir,
                                               "protocols",
                                               "r1.json"))
    subjects = json_reader.aggregate_values(
        'subject_alias', experiment=experiment)
    return subjects


def calculate_repetition_ratio(recall_events):
    """
        Determine the repetition ratio for a given list based on the recalled
        events for that list
    """
    is_repetition = np.diff(recall_events.category_num) == 0
    repetition_ratio = np.sum(is_repetition)/float(len(recall_events) - 1)

    return repetition_ratio
