__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import RamTask

import hashlib


class THREventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        super(THREventPreparation,self).__init__(mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        thr_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='THR')))
        for fname in thr_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        thr3_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='THR3')))
        for fname in thr3_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        print 'subject=',subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        if self.pipeline.sessions is None:
            event_files = sorted(
                list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='THR')))
        else:
            event_files = [json_reader.get_value('task_events',subject=subj_code,
                                                 montage=montage,experiment='THR',session=sess)
                           for sess in sorted(self.pipeline.sessions)]
        events = None
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            print e_path
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()
            sess_events = sess_events[sess_events.type=='CHEST']

            if events is None:
                events = sess_events
            else:
                events = np.hstack((events,sess_events))

        event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='THR3')))
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            print e_path
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()
            sess_events = sess_events[(sess_events.stim_list==0) & (sess_events.type=='CHEST')]
            sess_events.session += 200
            sess_events = sess_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]

            if events is None:
                events = sess_events
            else:
                events = np.hstack((events,sess_events))

        events = events.view(np.recarray)

        print len(events), 'Presentation events'

        self.pass_object('THR_events', events)
