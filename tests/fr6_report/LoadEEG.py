import os
import hashlib
import numpy as np

from sklearn.externals import joblib

from ReportUtils import ReportRamTask
from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper
from ptsa.data.readers  import JsonIndexReader

class LoadPostStimEEG(ReportRamTask):
    def __init__(self,params,**kwargs):
        super(LoadPostStimEEG, self).__init__(**kwargs)
        self.params=params

    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

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
        post_stim_eeg = {}
        events = self.get_passed_object('all_events')
        sessions = np.unique(events.session)
        # We want to see saturation in the post-stim period across all electrodes
        for session in sessions:
            sess_events = events[(events.type =='STIM_OFF') & (events.session == session)]
            eeg = EEGReader(events=sess_events,
                            start_time=self.params.post_stim_start_time,
                            end_time=self.params.post_stim_end_time + 0.25).read()
            samplerate = eeg['samplerate']
            eeg = eeg.filtered([58.,62.])
            eeg['samplerate'] = samplerate
            eeg = eeg.mean(dim='events').data
            eeg[np.abs(eeg)<5] = np.nan
            post_stim_eeg[session] = eeg

        self.pass_object('post_stim_eeg', post_stim_eeg)
        joblib.dump(post_stim_eeg, self.get_path_to_resource_in_workspace('post_stim_eeg.pkl'))

    def restore(self):
        eeg = joblib.load(self.get_path_to_resource_in_workspace('post_stim_eeg.pkl'))
        self.pass_object('post_stim_eeg',eeg)
