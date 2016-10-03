__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask


class FR1EventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(FR1EventPreparation,self).__init__(mark_as_completed)

    def run(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'data/eeg/protocols/r1.json'))

        event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='FR1')))
        fr1_events = None
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            print e_path
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()[['wordno', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime', 'type', 'eegoffset', 'iscorrect', 'answer', 'recalled', 'word', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]

            if fr1_events is None:
                fr1_events = sess_events
            else:
                fr1_events = np.hstack((fr1_events,sess_events))

        event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='catFR1')))
        catfr1_events = None
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            print e_path
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()
            sess_events.session += 100
            sess_events = sess_events[['wordno', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime', 'type', 'eegoffset', 'iscorrect', 'answer', 'recalled', 'word', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]

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
