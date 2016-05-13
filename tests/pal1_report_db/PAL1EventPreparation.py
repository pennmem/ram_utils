__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader

from RamPipeline import *

from ReportUtils import ReportRamTask

class PAL1EventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(PAL1EventPreparation,self).__init__(mark_as_completed)

    def run(self):
        task = self.pipeline.task

        e_path = os.path.join(self.pipeline.mount_point , 'data', 'events', task, self.pipeline.subject + '_events.mat')
        e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

        events = e_reader.read()

        self.pass_object(self.pipeline.task+'_all_events', events)

        intr_events = events[(events.intrusion!=-999) & (events.correct==0) & (events.vocalization!=1)]

        rec_events = events[(events.type == 'REC_EVENT') & (events.vocalization!=1)]

        test_probe_events = events[events.type == 'TEST_PROBE']

        events = events[(events.type == 'STUDY_PAIR') & (events.correct!=-999)]
        ev_order = np.argsort(events, order=('session','list','mstime'))
        events = events[ev_order]

        print len(events), task, 'STUDY_PAIR events'

        self.pass_object(task+'_events', events)
        self.pass_object(self.pipeline.task+'_intr_events', intr_events)
        self.pass_object(self.pipeline.task+'_rec_events', rec_events)
        self.pass_object(self.pipeline.task+'_test_probe_events', test_probe_events)
