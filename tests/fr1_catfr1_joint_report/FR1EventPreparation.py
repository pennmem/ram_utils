__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import hashlib
from ReportTasks.RamTaskMethods import load_events

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

        catfr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='catFR1')))
        for fname in catfr1_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        task=self.pipeline.task,
        sessions=self.pipeline.sessions

        fr1_events = load_events(subject,experiment='FR1',mount_point=self.pipeline.mount_point,*[s for s in sessions if s <100])
        assert len(fr1_events)>0
        catfr1_events = load_events(subject,experiment='catFR1',mount_point=self.pipeline.mount_point,*[s-100 for s in sessions if s>=100])
        assert len(catfr1_events)>0
        self.pass_object('cat_events',catfr1_events)
        catfr1_events = catfr1_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime', 'type', 'eegoffset', 'iscorrect', 'answer', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
        catfr1_events.session += 100
        events = np.hstack((fr1_events,catfr1_events)).view(np.recarray)
        self.pass_object('all_events', events)


        math_events = events[events.type == 'PROB']

        rec_events = events[events.type == 'REC_WORD']

        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]

        events = events[events.type == 'WORD']

        print len(events), 'WORD events'

        self.pass_object('events', events)
        self.pass_object('math_events', math_events)
        self.pass_object('intr_events', intr_events)
        self.pass_object('rec_events', rec_events)
