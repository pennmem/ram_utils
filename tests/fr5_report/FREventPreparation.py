__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import hashlib


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

        base_events= events[events.type == 'REC_BASE']

        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]

        events = events[events.type == 'WORD']

        print len(events), 'WORD events'

        self.pass_object('FR_events', events)
        self.pass_object('FR_math_events', math_events)
        self.pass_object('FR_intr_events', intr_events)
        self.pass_object('FR_rec_events', rec_events)
