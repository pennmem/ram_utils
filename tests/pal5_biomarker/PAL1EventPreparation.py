__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import RamTask

import hashlib
import warnings

class PAL1EventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        super(PAL1EventPreparation,self).__init__(mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        pal1_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='PAL1')))
        for fname in pal1_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        pal3_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='PAL3')))
        for fname in pal3_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run(self):
        evs_field_list = ['session','list','serialpos','type','probepos','study_1',
                          'study_2','cue_direction','probe_word','expecting_word',
                          'resp_word','correct','intrusion','resp_pass','vocalization',
                          'RT','mstime','msoffset','eegoffset','eegfile'
                          ]

        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='PAL1')))
        events = None
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))


            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
            try:
                sess_events = e_reader.read()[evs_field_list]
            except IOError:
                warnings.warn('Could not process %s. Please make sure that the event file exist'%e_path, RuntimeWarning)
                continue


            sess_events = sess_events[(sess_events.type == 'STUDY_PAIR') & (sess_events.correct!=-999)]

            if events is None:
                events = sess_events
            else:
                events = np.hstack((events,sess_events))

        # event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='PAL3')))
        # for sess_file in event_files:
        #     e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
        #     print e_path
        #     e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
        #
        #     sess_events = e_reader.read()
        #     sess_events = sess_events[(sess_events.stim_list==0) & (sess_events.type == 'STUDY_PAIR') & (sess_events.correct!=-999)]
        #     sess_events.session += 200
        #     sess_events = sess_events[evs_field_list]
        #
        #     if events is None:
        #         events = sess_events
        #     else:
        #         events = np.hstack((events,sess_events))

        events = events.view(np.recarray)

        print len(events), 'STUDY_PAIR events'

        self.pass_object('PAL1_events', events)
