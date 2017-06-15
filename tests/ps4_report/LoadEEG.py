from ptsa.data.readers import IndexReader,EEGReader,TalReader
import os
from ReportUtils import ReportRamTask


class LoadEEG(ReportRamTask):
    def run(self):
        events = self.get_passed_object('ps_events')
        post_stim_events = events[(events.type=='BIOMARKER') & (events.position=='POST')]

        eeg = self.load_eeg(post_stim_events)



    def load_eeg(self,events):

        jr = IndexReader.JsonIndexReader(os.path.join(self.pipeline.mount_point,'protocols','r1.json'))

        subject = self.pipeline.subject
        temp = subject.split('_')
        if len(temp)>1:
            montage=temp[1]
        else:
            montage = 0
        subject= temp[0]

        pair_file = jr.get_value('pairs',subject=subject,experiment=self.pipeline.task,montage=montage)


        channels = self.get_passed_object()
