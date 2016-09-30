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

        events_per_session_files = json.load(open(os.path.join(self.pipeline.mount_point, '/data/eeg/protocols/r1.json')))['protocols']['r1']['subjects'][subject]['experiments']['FR1']['sessions']
        fr1_events = None
        for sess in sorted(events_per_session_files.keys()):
            print 'FR1 session', sess, 'events found'
            e_path = str(os.path.join(self.pipeline.mount_point, 'data/eeg', events_per_session_files[sess]['all_events']))
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()[['wordno', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'test', 'mstime', 'type', 'eegoffset', 'iscorrect', 'answer', 'recalled', 'word', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
            print sess_events.dtype

            if fr1_events is None:
                fr1_events = sess_events
            else:
                fr1_events = np.hstack((fr1_events,sess_events))

        events_per_session_files = json.load(open(os.path.join(self.pipeline.mount_point, '/data/eeg/protocols/r1.json')))['protocols']['r1']['subjects'][subject]['experiments']['catFR1']['sessions']
        catfr1_events = None
        for sess in sorted(events_per_session_files.keys()):
            print 'catFR1 session', sess, 'events found'
            e_path = str(os.path.join(self.pipeline.mount_point, 'data/eeg', events_per_session_files[sess]['all_events']))
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()
            sess_events.session += 100
            sess_events = sess_events[['wordno', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'test', 'mstime', 'type', 'eegoffset', 'iscorrect', 'answer', 'recalled', 'word', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
            print sess_events.dtype

            if catfr1_events is None:
                catfr1_events = sess_events
            else:
                catfr1_events = np.hstack((catfr1_events,sess_events))

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
