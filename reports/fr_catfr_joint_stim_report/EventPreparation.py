import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *

from ReportUtils import ReportRamTask

import hashlib


class EventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(EventPreparation, self).__init__(mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()
        for task in [self.pipeline.task,'cat'+self.pipeline.task]:
            event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
            for fname in event_files:
                with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        events = None

        if self.pipeline.sessions is None or not self.pipeline.sessions:
            event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        else:
            fr_sessions = [s for s in self.pipeline.sessions if s <100]
            print 'fr sessions: ',fr_sessions
            event_files = [json_reader.get_value('all_events',subject=subj_code,montage=montage,experiment=task,session=s)
                           for s in fr_sessions]

        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, sess_file)
            print e_path
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()

            if events is None:
                events = sess_events
            else:
                events = np.hstack((events,sess_events))
        fr_event_fields=list(events.dtype.names)

        if self.pipeline.sessions is None or not self.pipeline.sessions:
            cat_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='cat'+task)))
        else:
            catfr_sessions = [s-100 for s in self.pipeline.sessions if s>=100]
            print 'catFR3 sessions: ', catfr_sessions
            cat_event_files = [json_reader.get_value('all_events',subject=subj_code,montage=montage,experiment='cat'+task,session=s)
                               for s in catfr_sessions]

        cat_events = None
        for sess_file in cat_event_files:
            e_path = os.path.join(self.pipeline.mount_point, sess_file)
            print e_path
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()

            if cat_events is None:
                cat_events = sess_events
            else:
                cat_events = np.hstack((events,sess_events))

        self.pass_object('cat_events',cat_events[cat_events.type=='WORD'].view(np.recarray))

        cat_events.session += 100
        events = np.hstack((events, cat_events[fr_event_fields].copy()))
        events = events.view(np.recarray)

        self.pass_object(task+'_all_events', events)

        math_events = events[events.type == 'PROB']

        rec_events = events[events.type == 'REC_WORD']

        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]

        events = events[events.type == 'WORD']

        print len(events), task, 'WORD events'

        self.pass_object(task+'_events', events)
        self.pass_object(task+'_math_events', math_events)
        self.pass_object(task+'_intr_events', intr_events)
        self.pass_object(task+'_rec_events', rec_events)
