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
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols','r1.json'))

        hash_md5 = hashlib.md5()

        event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        evs_field_list = ['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime', 'type', 'eegoffset', 'iscorrect', 'answer', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']
        if task=='catFR1':
            evs_field_list += ['category', 'category_num']

        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols','r1.json'))

        if self.pipeline.sessions is None or not self.pipeline.sessions:
            event_files = sorted(
                list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        else:
            event_files = [json_reader.get_value('all_events',subject=subj_code,
                                                                             montage=montage,experiment=task,session=sess)
                                          for sess in sorted(self.pipeline.sessions)]

        events = None
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = filter_session(e_reader.read()[evs_field_list])

            if events is None:
                events = sess_events
            else:
                events = np.hstack((events,sess_events))

        events = events.view(np.recarray)
        events = events[events.list>-1]



        self.pass_object(task+'_all_events', events)

        events = create_baseline_events(events, start_time=1000, end_time=29000)

        math_events = events[events.type == 'PROB']

        rec_events = events[events.type == 'REC_WORD']

        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]
        irts = np.append([0],np.diff(events.mstime))

        events = events[(events.type == 'WORD')| ((events.intrusion==0) & (irts>1000)) | (events.type=='REC_BASE')]

        print len(events), task, 'WORD events'

        self.pass_object(task+'_events', events)
        self.pass_object(task+'_math_events', math_events)
        self.pass_object(task+'_intr_events', intr_events)
        self.pass_object(task+'_rec_events', rec_events)
