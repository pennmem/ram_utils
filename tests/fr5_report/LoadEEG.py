from ReportUtils import ReportRamTask
from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper
from sklearn.externals import joblib
from ptsa.data.readers.IndexReader import JsonIndexReader
import hashlib,os
import numpy as np
from matplotlib.colorbar import colorbar_doc

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
        events = self.get_passed_object('all_events')
        events = events[events.type=='STIM_OFF']

        channels = self.get_passed_object('monopolar_channels')
        pairs = self.get_passed_object('bipolar_pairs')
        try:
            eeg = EEGReader(events=events,channels=channels,
                            start_time=self.params.post_stim_start_time,
                            end_time=self.params.post_stim_end_time+0.25,).read()
        except IndexError:
            eeg = EEGReader(events=events, channels=np.array([]),
                            start_time=self.params.post_stim_start_time,
                            end_time=self.params.post_stim_end_time + 0.25, ).read()
        samplerate = eeg['samplerate']
        eeg = eeg.filtered([58.,62.])
        eeg['samplerate']=samplerate
        if 'channels' in eeg.coords:
            eeg = MonopolarToBipolarMapper(time_series=eeg,bipolar_pairs=pairs).filter()
        eeg = eeg.mean(dim='events').data
        eeg[np.abs(eeg)<5]=np.nan


        self.pass_object('post_stim_eeg',eeg)
        joblib.dump(eeg,self.get_path_to_resource_in_workspace('post_stim_eeg.pkl'))

    def restore(self):
        eeg = joblib.load(self.get_path_to_resource_in_workspace('post_stim_eeg.pkl'))
        self.pass_object('post_stim_eeg',eeg)



