from ptsa.data.readers import EEGReader,IndexReader
from ptsa.data.filters import MonopolarToBipolarMapper
import numpy as np
from ReportUtils import ReportRamTask
from sklearn.externals import joblib
import os
import hashlib

class LoadEEG(ReportRamTask):
    def __init__(self,params,**kwargs):
        super(LoadEEG, self).__init__(**kwargs)
        self.params=  params

    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = IndexReader.JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()
        hash_md5.update(__file__)

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()



    def run(self):
        events = self.get_passed_object('ps_events')
        post_stim_events = events[(events.type=='BIOMARKER') & (events.position=='POST')]

        channels = self.get_passed_object('monopolar_channels')
        pairs = self.get_passed_object('bipolar_pairs')

        eeg = EEGReader(events=post_stim_events,channels=channels,start_time=self.params.start_time,
                        end_time=self.params.end_time).read()
        eeg = eeg.filtered([58.,62.])
        eeg = MonopolarToBipolarMapper(time_series=eeg,bipolar_pairs=pairs).filter()
        eeg = eeg.mean(dim='events').data
        eeg[np.abs(eeg)<5]=np.nan


        self.pass_object('eeg',eeg)
        joblib.dump(eeg,self.get_path_to_resource_in_workspace(self.pipeline.subject+'-eeg.pkl'))


    def restore(self):
        eeg = joblib.load(self.get_path_to_resource_in_workspace(self.pipeline.subject+'-eeg.pkl'))
        self.pass_object('eeg',eeg)







