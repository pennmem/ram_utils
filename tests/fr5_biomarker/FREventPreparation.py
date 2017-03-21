__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import RamTask

import hashlib


class FREventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        super(FREventPreparation, self).__init__(mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        fr1_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))
        for fname in fr1_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        catfr1_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR1')))
        for fname in catfr1_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        fr3_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR3')))
        for fname in fr3_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        catfr3_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR3')))
        for fname in catfr3_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        fr1_events_fname = os.path.abspath(
            os.path.join(self.pipeline.mount_point, 'scratch','jkragel','events_FR5','RAM_FR1', subj_code + '_events.mat'))

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        # event_files = sorted(
        #     list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))

        e_reader = BaseEventReader(filename=fr1_events_fname, eliminate_events_with_no_eeg=True,common_root='scratch')
        fr1_events = e_reader.read()
        print

        encoding_events_mask = fr1_events.type == 'WORD'
        retrieval_events_mask = (fr1_events.type == 'REC_WORD') | (fr1_events.type == 'REC_BASE')

        retrieval_events_mask_0s = retrieval_events_mask & (fr1_events.type == 'REC_BASE')
        retrieval_events_mask_1s = retrieval_events_mask & (fr1_events.type == 'REC_WORD') & \
                                   (fr1_events['repeat'] == 0) & (fr1_events.pirt > 1000) & (fr1_events.intrusion == 0)

        filtered_events = fr1_events[encoding_events_mask | retrieval_events_mask_0s | retrieval_events_mask_1s]

        print

        events = filtered_events.view(np.recarray)

        print len(events), 'WORD events'

        self.pass_object('FR_events', events)
        # self.pass_object('encoding_events_mask',encoding_events_mask)
        # self.pass_object('retrieval_events_mask_0s',retrieval_events_mask_0s)
        # self.pass_object('retrieval_events_mask_1s',retrieval_events_mask_1s)


def free_epochs(times, duration, pre, post):
    # (list(vector(int))*int*int*int) -> list(vector(int))
    """
    Given a list of event times, find epochs between them when nothing is happening

    Parameters:
    -----------

    times:
        An iterable of 1-d numpy arrays, each of which indicates event times

    duration: int
        The length of the desired empty epochs

    pre: int
        the time before each event to exclude

    post: int
        The time after each event to exclude

    """
    n_trials = len(times)
    epoch_times = []
    for i in range(n_trials):
        pre_times = times[i] - pre
        post_times = times[i] + post
        interval_durations = pre_times[1:] - post_times[:-1]
        free_intervals = np.where(interval_durations > duration)[0]
        trial_epoch_times = []
        for interval in free_intervals:
            start = post_times[interval]
            finish = pre_times[interval + 1] - duration
            interval_epoch_times = range(start, finish, duration)
            trial_epoch_times.extend(interval_epoch_times)
        epoch_times.append(np.array(trial_epoch_times))
    epoch_array = np.empty((n_trials, max([len(x) for x in epoch_times])))
    epoch_array[...] = -np.inf
    for i, epoch in enumerate(epoch_times):
        epoch_array[i, :len(epoch)] = epoch
    return epoch_array


def create_baseline_events(events,filter):
    sess_events=filter(events)
    times = [sess_events[sess_events.list == lst].mstime for lst in np.unique(sess_events.list)]
    epochs = free_epochs(times, 500, 1000, 1000)
    starts = events[(events.type == 'REC_START') & (events.session == 1)]
    start_times = starts[np.in1d(starts.list, sess_events.list)].mstime
    rel_times = [t - i for (t, i) in zip(times, start_times)]
    rel_epochs = epochs - start_times[:, None]
    full_match_accum = np.empty(epochs.shape, dtype=np.bool)
    full_match_accum[...] = False
    for (i, rec_times_list) in enumerate(rel_times):
        is_match = np.empty(epochs.shape, dtype=np.bool)
        is_match[...] = False
        for t in rec_times_list:
            is_match_tmp = np.abs((rel_epochs - t)) < 3000
            is_match_tmp[i, ...] = False
            is_match |= is_match_tmp
        full_match_accum |= is_match
    print full_match_accum.shape
    print epochs.shape
    matching_epochs = epochs[full_match_accum]
    matching_epochs = np.random.choice(matching_epochs,sess_events.size)
    new_events = np.zeros(sess_events.shape,dtype=sess_events.dtype).view(np.recarray)
    for i,_ in enumerate(new_events):
        new_events[i].mstime=matching_epochs[i]
        new_events[i].type='REC_BASE'
    new_events.recalled=0
    merged_events=np.concatenate((events,new_events)).view(np.recarray)
    merged_events.sort(order='mstime')
    for (i,event) in enumerate(merged_events):
        if event.type=='REC_BASE':
            merged_events[i].session=merged_events[i-1].session
            merged_events[i].list=merged_events[i-1].list
            merged_events[i].eegfile=merged_events[i-1].eegfile
            merged_events[i].eegoffset=merged_events[i-1].eegoffset + (merged_events[i].mstime-merged_events[i-1].mstime)
    return new_events


