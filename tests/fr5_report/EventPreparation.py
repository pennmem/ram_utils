__author__ = 'm'

import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import hashlib
from copy import deepcopy

class FR1EventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(FR1EventPreparation,self).__init__(mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        fr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='FR1')))
        for fname in fr1_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='FR1')))
        fr1_events = np.concatenate([BaseEventReader(filename=f,eliminate_events_with_no_eeg=True) for f in event_files])
        events=fr1_events.view(np.recarray)

        events = self.make_base_events(events)

        self.pass_object('all_events', events)

        math_events = events[events.type == 'PROB']

        rec_events = events[events.type == 'REC_WORD']

        base_events= events[events.type=='REC_BASE']

        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]

        events = events[events.type == 'WORD']

        print len(events), 'WORD events'


        self.pass_object('FR_baseline_events',base_events)
        self.pass_object('FR_events', events)
        self.pass_object('FR_math_events', math_events)
        self.pass_object('FR_intr_events', intr_events)
        self.pass_object('FR_rec_events', rec_events)


    def make_base_events(self,events):
        all_events = []
        for session in np.unique(events.session):
            sess_events = events[events.session==session]
            rec_events = sess_events[(sess_events.type == 'REC_WORD') & (sess_events.intrusion == 0)]
            voc_events = sess_events[((sess_events.type == 'REC_WORD') | (sess_events.type == 'REC_WORD_VV'))]
            starts = sess_events[(sess_events.type == 'REC_START')]
            ends = sess_events[(sess_events.type == 'REC_END')]
            rec_lists = tuple(np.unique(starts.list))
            times = [voc_events[(voc_events.list == lst)].mstime for lst in rec_lists]
            start_times = starts.mstime
            end_times = ends.mstime
            epochs = free_epochs(times, 500, 1000, 1000, start=start_times, end=end_times)
            rel_times = [t - i for (t, i) in
                         zip([rec_events[rec_events.list == lst].mstime for lst in rec_lists], start_times)]
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

class FR5EventPreparation(ReportRamTask):
    def __init__(self):
        super(FR5EventPreparation,self).__init__(mark_as_completed=False)
    def run(self):
        jr = JsonIndexReader(os.path.join(self.pipeline.mount_point,'protocols','r1.json'))
        temp=self.pipeline.subject.split('_')
        subject= temp[0]
        montage = 0 if len(temp)==1 else temp[1]
        task = self.pipeline.task

        events = [ BaseEventReader(filename=event_path).read() for event_path in
                                jr.aggregate_values('task_events',subject=subject,montage=montage,experiment=task)]
        if events:
            events = np.concatenate(events).view(np.recarray)
        self.pass_object('fr5_events',events)




def free_epochs(times, duration, pre, post, start=None, end=None):
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
            interval_epoch_times = range(begin, finish, duration)
            trial_epoch_times.extend(interval_epoch_times)
        epoch_times.append(np.array(trial_epoch_times))
    epoch_array = np.empty((n_trials, max([len(x) for x in epoch_times])))
    epoch_array[...] = -np.inf
    for i, epoch in enumerate(epoch_times):
        epoch_array[i, :len(epoch)] = epoch
    return epoch_array

class PSEventPreparation(ReportRamTask):
    def __init__(self):
        super(PSEventPreparation,self).__init__(mark_as_completed=False)

    def run(self):
        jr = JsonIndexReader(os.path.join(self.pipeline.mount_point,'protocols','r1.json'))
        temp=self.pipeline.subject.split('_')
        subject= temp[0]
        montage = 0 if len(temp)==1 else temp[1]

        events = np.concatenate([ BaseEventReader(filename=event_path).read() for event_path in
                                jr.aggregate_values('task_events',subject=subject,montage=montage,experiment='PS4')])
        events = events.view(np.recarray)
        self.pass_object('ps_events',events)


