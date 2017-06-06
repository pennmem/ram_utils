__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import hashlib


class THREventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(THREventPreparation,self).__init__(mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

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

        if self.pipeline.sessions is None:
            event_files = sorted(
                list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment=task)))
        else:
            event_files = [json_reader.get_value('task_events',subject=subj_code,
                                                 montage=montage,experiment=task,session=sess)
                           for sess in sorted(self.pipeline.sessions)]

        events = None
        print event_files
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()

            if events is None:
                events = sess_events
            else:
                events = np.hstack((events,sess_events))

        events = events.view(np.recarray)

        self.pass_object(task+'_all_events', events)

        probe_events = events[events.type == 'PROBE']
        events = events[events.type == 'CHEST']

        print len(events), task, 'presentation events'

        self.pass_object(task+'_events', events)
        self.pass_object(task+'_probe_events', probe_events)
