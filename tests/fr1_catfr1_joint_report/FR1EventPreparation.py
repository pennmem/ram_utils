__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import hashlib
from ReportTasks.RamTaskMethods import create_baseline_events
from ReportTasks.RamTaskMethods import filter_session


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
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        if self.pipeline.sessions is None or not self.pipeline.sessions:
            event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='FR1')))
        else:
            fr1_sessions = [s for s in self.pipeline.sessions if s<100]
            event_files = [json_reader.get_value('all_events',subject=subj_code,montage=montage,experiment='FR1', session=s)
                           for s in fr1_sessions]
        assert any(event_files)
        fr1_events = None
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            print e_path
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = filter_session(e_reader.read())[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime', 'type', 'eegoffset', 'iscorrect', 'answer', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
            if fr1_events is None:
                fr1_events = sess_events
            else:
                fr1_events = np.hstack((fr1_events,sess_events))

        if self.pipeline.sessions is None or not self.pipeline.sessions:
            event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='catFR1')))
        else:
            catfr1_sessions =[s-100 for s in self.pipeline.sessions if s>=100]
            event_files = [json_reader.get_value('all_events',subject=subj_code,montage=montage,experiment='catFR1',session=s)
                           for s in catfr1_sessions]
        catfr1_events = None
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            print e_path
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = filter_session(e_reader.read())
            sess_events.session += 100
            if catfr1_events is None:
                catfr1_events = sess_events
            else:
                catfr1_events = np.hstack((catfr1_events,sess_events))

        self.pass_object('cat_events',catfr1_events.view(np.recarray))

        catfr1_events = catfr1_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime', 'type', 'eegoffset', 'iscorrect', 'answer', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]

        events = np.hstack((fr1_events,catfr1_events)).view(np.recarray)

        events = events[events.list>-1]

        events = create_baseline_events(events, start_time=1000, end_time=29000)


        self.pass_object('all_events', events)


        math_events = events[events.type == 'PROB']

        rec_events = events[events.type == 'REC_WORD']

        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]

        events = events[(events.type == 'WORD') | (events.type=='REC_BASE') | (events.intrusion==0)]

        print len(events), 'task events'

        for session in np.unique(events.session):
            print np.unique(events[events.session==session].list)

        self.pass_object('events', events)
        self.pass_object('math_events', math_events)
        self.pass_object('intr_events', intr_events)
        self.pass_object('rec_events', rec_events)
