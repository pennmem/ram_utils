import os
import os.path

from ptsa.data.readers import BaseEventReader

from RamPipeline import *

from ReportUtils import ReportRamTask

class EventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(EventPreparation, self).__init__(mark_as_completed)

    def run(self):
        task = self.pipeline.task

        e_path = os.path.join(self.pipeline.mount_point, 'data/events', task, self.pipeline.subject + '_events.mat')
        e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=False)

        events = e_reader.read()

        self.pass_object(self.pipeline.task+'_all_events', events)

        intr_events = events[(events.intrusion!=-999) & (events.intrusion!=0)]

        rec_events = events[events.type == 'REC_WORD']

        events = events[events.type == 'WORD']

        print len(events), task, 'WORD events'

        self.pass_object(task+'_events', events)
        self.pass_object(self.pipeline.task+'_intr_events', intr_events)
        self.pass_object(self.pipeline.task+'_rec_events', rec_events)
