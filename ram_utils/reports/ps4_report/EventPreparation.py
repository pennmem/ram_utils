from ...ReportUtils import ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader
from ptsa.data.readers import BaseEventReader
from os.path import join
import numpy as np

class EventPreparation(ReportRamTask):
    def __init__(self,mark_as_completed):
        super(EventPreparation, self).__init__(mark_as_completed)

    def run(self):
        subject=self.pipeline.subject
        temp = subject.split('_')
        subject=temp[0]
        montage = 0 if len(temp)==1 else temp[1]
        jr = JsonIndexReader(join(self.pipeline.mount_point,'protocols','r1.json'))

        events = [BaseEventReader(filename=f).read() for f in jr.aggregate_values('ps4_events',subject=subject,montage=montage)]
        events = np.concatenate(events).view(np.recarray)
        self.pass_object('ps_events',events)


