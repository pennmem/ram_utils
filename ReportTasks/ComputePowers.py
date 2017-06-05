from ReportUtils import ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader
import os
import numpy as np
from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper,MorletWaveletFilterCpp
from scipy.stats.mstats import zscore
from ptsa.data.readers import BaseRawReader
from ptsa.data.TimeSeriesX import TimeSeriesX
import xarray

class ComputePowers(ReportRamTask):
    def __init__(self,params,**kwargs):
        super(ComputePowers,self).__init__(**kwargs)
        self.params = params
        self.pow_mat = None

    def input_hashsum(self):
        task = self.params.task
        subject = self.pipeline.subject.split('_')
        subj_code =subject[0]
        montage = 0 if len(subject)==1 else int(subject[1])
        jr = JsonIndexReader(os.path.join(self.pipeline.mount_point,'protocols/r1.json'))

        event_paths = jr.aggregate_values('task_events',subject=subj_code,montage=montage,experiment=task)
        for path in event_paths:
            with open(os.path.join(self.pipeline.mount_point,path)) as event:
                self.hash.update(event.read())

    def run(self):
        events = self.get_passed_object('%s_events'%(self.params.task))
        for session in np.unique(events.session):
            events = self.prepare_eeg(events[events.session == session])
            sess_pow_mat = self.compute_powers()
            self.events = events if self.events is None else np.concatenate(self.events,events)
            self.pow_mat = sess_pow_mat if self.pow_mat is None else np.concatenate(self.pow_mat,sess_pow_mat)

        self.pow_mat = np.nanmean(self.pow_mat,-1)
        self.pow_mat = self.pow_mat.reshape((len(self.events),-1))
        self.pass_object(self.params.name,self.pow_mat)
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
                                                              order=self.params.filt_order)
        self.eeg = MonopolarToBipolarMapper(time_series = eeg,bipolar_pairs=bipolar_pairs).filter()
        return self.eeg['events'].values.view(np.recarray)

    def compute_powers(self):
        pow_mat,_ = MorletWaveletFilterCpp(time_series=self.eeg,freqs=self.params.freqs,
                                           output='power',cpus=self.params.cpus).filter()
        pow_mat = pow_mat.transpose('events','bipolar_pairs','frequency','time').remove_buffer(self.params.buf)
        pow_mat = pow_mat.data
        if self.params.log10:
            np.log10(pow_mat,out=pow_mat)
        if self.params.zscore:
            pow_mat = zscore(pow_mat,0,ddof=1)
        return pow_mat


class ComputePSPowers(ComputePowers):

    def prepare_eeg(self, events):
        n_events = len(events)
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        pre_start_time = self.params.start_time - self.params.offset
        pre_end_time = self.params.end_time - self.params.offset

        eeg_pre_reader = EEGReader(events=events, channels=monopolar_channels,
                                   start_time=pre_start_time,
                                   end_time=pre_end_time, buffer_time=0)

        eegs_pre = eeg_pre_reader.read()
        if eeg_pre_reader.removed_bad_data():
            print 'REMOVED SOME BAD EVENTS !!!'

        events = eegs_pre['events'].values.view(np.recarray)
        eegs_pre.add_mirror_buffer(self.params.buf)

        post_start_time = self.params.ps_offset
        post_end_time = self.params.ps_offset + (self.params.ps_end_time - self.params.ps_start_time)

        post_start_time = self.params.ps_offset
        post_end_time = self.params.ps_offset + (self.params.ps_end_time - self.params.ps_start_time)

        post_start_offsets = np.copy(events.eegoffset)

        for i_ev in xrange(n_events):
            ev_offset = events[i_ev].stim_duration
            if ev_offset > 0:
                ev_offset *= 0.001
            else:
                ev_offset = 0.0

            post_start_offsets[i_ev] += (ev_offset + post_start_time - self.params.ps_buf) * float(eegs_pre['samplerate'])

        read_size = eegs_pre['time'].shape[0]
        dataroot = events[0].eegfile
        brr = BaseRawReader(dataroot=dataroot, start_offsets=post_start_offsets,
                            channels=np.array(monopolar_channels), read_size=read_size)

        eegs_post, read_ok_mask = brr.read()

        # #removing bad events from both pre and post eegs
        if np.any(~read_ok_mask):
            # print 'YES'
            read_mask_ok_events = np.all(read_ok_mask, axis=0)
            eegs_post = eegs_post[:, read_mask_ok_events, :]
            # events = events[read_mask_ok_events]
            eegs_pre = eegs_pre[:, read_mask_ok_events, :]

            # FIXING ARRAY ALL EVENTS - MAKE IT A FUNCTION!
            events = eegs_pre['events'].values.view(np.recarray)

        eegs_post = eegs_post.rename({'offsets': 'time', 'start_offsets': 'events'})
        eegs_post['events'] = events
        eegs_post['time'] = eegs_pre['time'].data
        eegs_post = TimeSeriesX(eegs_post)
        self.eeg = xarray.concat((eegs_pre,eegs_post),dim='events')
        self.n_events=len(events)
        return events

    def compute_powers(self):
        combined_eeg=self.eeg
        self.params.zscore=False
        self.eeg=combined_eeg[:,:self.n_events,:]
        pow_mat_pre = super(ComputePSPowers,self).compute_powers()
        self.eeg=combined_eeg[:,self.n_events:,:]
        pow_mat_post = super(ComputePSPowers,self).compute_powers()
        return zscore(np.concatenate(pow_mat_pre,pow_mat_post))















