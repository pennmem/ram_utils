__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import hashlib
from copy import deepcopy

class FREventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(FREventPreparation,self).__init__(mark_as_completed)

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

        self.pass_object('all_events', events)

        math_events = events[events.type == 'PROB']

        rec_events = events[events.type == 'REC_WORD']

        base_events= self.make_base_events(events)

        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]

        events = events[events.type == 'WORD']

        print len(events), 'WORD events'


        self.pass_object('FR_baseline_events',base_events)
        self.pass_object('FR_events', events)
        self.pass_object('FR_math_events', math_events)
        self.pass_object('FR_intr_events', intr_events)
        self.pass_object('FR_rec_events', rec_events)


    def make_base_events(self,events):
        time_gaps = np.append([0], np.diff(events.mstime))
        vocalizations = (events.type == 'REC_WORD') | (events.type == 'REC_WORD_VV')
        is_rec_break_type = vocalizations | (events.type == 'REC_END')
        n_periods = ((time_gaps-2000)/525)
        break_starts = events[n_periods.astype(np.bool) & is_rec_break_type].mstime
        break_durations = time_gaps[n_periods.astype(np.bool) & is_rec_break_type]
        baseline_durations = break_durations -1000 # The second before recall is invalid
        baseline_mstimes = [ range(start+1000,start+duration,525)[:-1] for start,duration in zip(break_starts,baseline_durations)]
        baseline_mstimes = np.concatenate(baseline_mstimes)
        closest_times = []
        baseline_events = []
        for event in events[events.type=='REC_WORD']:
            valid_times= baseline_mstimes[np.array([time not in closest_times for time in baseline_mstimes])
                                         & (np.abs(baseline_mstimes-event.mstime)>=1000)]
            closest_times.append(valid_times[np.abs(valid_times-event.mstime).argmin()])
            new_event=deepcopy(event)
            new_event.mstime=closest_times[-1]
            new_event.eegoffset = event.eegoffset + (event.mstime-new_event.mstime)
            new_event.recalled=0
            baseline_events.append(new_event)
        return np.array(baseline_events).view(np.recarray)



