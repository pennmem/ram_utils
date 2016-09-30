__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import json


class FR1EventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(FR1EventPreparation,self).__init__(mark_as_completed)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = None
        events_per_session_files = json.load(open(os.path.join(self.pipeline.mount_point, '/data/eeg/protocols/r1.json')))['protocols']['r1']['subjects'][subject]['experiments'][task]['sessions']
        for sess in sorted(events_per_session_files.keys()):
            print 'Session', sess, 'events found'
            e_path = str(os.path.join(self.pipeline.mount_point, 'data/eeg', events_per_session_files[sess]['all_events']))
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()
            print sess_events.dtype

            # ev_order = np.argsort(events, order=('list','mstime'))
            # sess_events = sess_events[ev_order]

            if events is None:
                events = sess_events
            else:
                events = np.hstack((events,sess_events))

        events = events.view(np.recarray)

        self.pass_object(self.pipeline.task+'_all_events', events)

        math_events = events[events.type == 'PROB']

        rec_events = events[events.type == 'REC_WORD']

        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]

        events = events[events.type == 'WORD']

        print len(events), task, 'WORD events'

        self.pass_object(task+'_events', events)
        self.pass_object(self.pipeline.task+'_math_events', math_events)
        self.pass_object(self.pipeline.task+'_intr_events', intr_events)
        self.pass_object(self.pipeline.task+'_rec_events', rec_events)
