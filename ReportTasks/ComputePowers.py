from ReportUtils import ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader
import os
import numpy as np
from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper,MorletWaveletFilterCpp
from scipy.stats.mstats import zscore



class ComputePowers(ReportRamTask):
    def __init__(self,params,mark_as_completed=True):
        super(ComputePowers,self).__init__(mark_as_completed=mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.events = None

    def input_hashsum(self):
        task = self.params.task
        subject = self.pipeline.subject.split('_')
        subj_code =subject[0]
        montage = subject[1]
        jr = JsonIndexReader(os.path.join(self.pipeline.mount_point,'/protocols/r1.json'))

        event_paths = jr.aggregate_values('task_events',subject=subj_code,montage=montage,experiment=task)
        for path in event_paths:
            with open(os.path.join(self.pipeline.mount_point,path)) as event:
                self.hash.update(event.read())

    def run(self):
        events = self.get_passed_object('%s_events'%(self.params.task))
        for session in np.unique(events.session):
            sess_events = self.prepare_eeg(events[events.session == session])
            sess_pow_mat = self.compute_powers()
            self.events = sess_events if self.events is None else np.concatenate(self.events,sess_events)
            self.pow_mat = sess_pow_mat if self.pow_mat is None else np.concatenate(self.pow_mat,sess_pow_mat)

    def post(self):
        self.pow_mat = self.pow_mat.reshape((len(self.events),-1))
        self.pass_object('pow_mat',self.pow_mat)
        self.pass_object('%s_events'%(self.params.task),self.events)

    def prepare_eeg(self, events):
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        eeg_reader = EEGReader(events=events,channels =monopolar_channels,
                        start_time=self.params.start, end_time=self.params.end)
        eeg = eeg_reader.read()
        if eeg_reader.removed_bad_data():
            print 'REMOVED SOME BAD EVENTS !!!'
        eeg = eeg.add_mirror_buffer(self.params.buf).filtered(self.params.butter_freqs,filt_type='stop',
                                                              order=self.params.order)
        self.eeg = MonopolarToBipolarMapper(time_series = eeg,bipolar_pairs=bipolar_pairs).filter()
        return self.eeg.events.data

    def compute_powers(self):
        pow_mat,_ = MorletWaveletFilterCpp(time_series=self.eeg,freqs=self.params.freqs,
                                           output='power',cpus=self.params.cpus).filter()
        pow_mat = pow_mat.transpose('events','bipolar_pairs','frequency','time').remove_buffer(self.params.buf)
        pow_mat = pow_mat.data
        if self.params.log10:
            np.log10(pow_mat,out=pow_mat)
        if self.params.zscore:
            pow_mat = zscore(pow_mat,0,ddof=1)
        return np.nanmean(pow_mat,-1)






