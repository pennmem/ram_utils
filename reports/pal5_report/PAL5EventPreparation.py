import numpy as np
from ReportUtils import ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader
from ptsa.data.readers import BaseEventReader
import os


class PAL5EventPreparation(ReportRamTask):

    def run(self):
        subject_list=self.pipeline.subject.split('_')
        montage = 0 if len(subject_list)==1 else int(subject_list[1])
        subject= subject_list[0]

        jr = JsonIndexReader(os.path.join(self.pipeline.mount_point,'protocols/r1.json'))
        events = np.concatenate([BaseEventReader(filename=pf).read() for pf in
                                 jr.aggregate_values('all_events',subject=subject,montage=montage,experiment='PAL5')]
                                ).view(np.recarray)

        self.pass_object('all_events',events)

        math_events = events[events.type=='PROB']

        rec_events = events[events.type=='REC_EVENT']

        intr_events = events[(events.intrusion != -999) & (events.intrusion != 0)]

        self.pass_object('math_events',math_events)
        self.pass_object('rec_events',rec_events)
        self.pass_object('intr_events',intr_events)

        events = events[(events.type=='STUDY_PAIR') | (events.type=='PRACTICE_PAIR')]

        print '%s STUDY_PAIR events'%len(events)

        self.pass_object('events',events)

