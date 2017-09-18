__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import hashlib


class PALEventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(PALEventPreparation,self).__init__(mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        pal1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='PAL1')))
        for fname in pal1_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run(self):
        evs_field_list = ['session','list','serialpos','type','probepos','study_1',
                          'study_2','cue_direction','probe_word','expecting_word',
                          'resp_word','correct','intrusion','resp_pass','vocalization',
                          'RT','mstime','msoffset','eegoffset','eegfile','iscorrect'
                          ]

        subject = self.pipeline.subject

        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        if self.pipeline.sessions is None or not self.pipeline.sessions:
            event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='PAL1')))
        else:
            event_files = [json_reader.get_value('all_events',subject=subj_code,montage=montage,experiment='PAL1',session=s)
                           for s in self.pipeline.sessions]
        events = None
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()
            sess_events = sess_events[evs_field_list].copy()

            if events is None:
                events = sess_events
            else:
                events = np.hstack((events,sess_events))

        events = events.view(np.recarray)

        self.pass_object('PAL1_all_events', events)

        math_events = events[events.type == 'PROB']

        rec_events = events[(events.type == 'REC_EVENT') & (events.vocalization!=1)]

        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.correct==0)]

        test_probe_events = events[events.type == 'TEST_PROBE']

        events = events[(events.type == 'STUDY_PAIR') & (events.correct!=-999)]

        print len(events), 'PAL STUDY_PAIR events'

        self.pass_object('PAL_events', events)
        self.pass_object('PAL_math_events', math_events)
        self.pass_object('PAL_intr_events', intr_events)
        self.pass_object('PAL_rec_events', rec_events)
        self.pass_object('PAL_test_probe_events', test_probe_events)
