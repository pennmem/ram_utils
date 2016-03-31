__author__ = 'm'

from RamPipeline import *

import numpy as np
from scipy.stats.mstats import zscore
from morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.readers import EEGReader


from ReportUtils import ReportRamTask

class ComputePAL1HFPowers(ReportRamTask):
    def __init__(self, params,mark_as_completed=True):
        super(ComputePAL1HFPowers,self).__init__(mark_as_completed)

        self.params = params
        self.pow_mat = None
        self.samplerate = None
        self.wavelet_transform = MorletWaveletTransform()

    def initialize(self):
        task = self.pipeline.task
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name=task+'_events',
                                        access_path = ['experiments','pal1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-hf_pow_mat.pkl'))
        self.pass_object('hf_pow_mat', self.pow_mat)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task+'_events')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')

        self.compute_powers(events, sessions, monopolar_channels, bipolar_pairs)

        self.pass_object('hf_pow_mat', self.pow_mat)

        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-' + task + '-hf_pow_mat.pkl'))

    def compute_powers(self, events, sessions,monopolar_channels , bipolar_pairs ):
        n_hfs = len(self.params.hfs)
        n_bps = len(bipolar_pairs)

        self.pow_mat = None

        pow_ev = None
        winsize = bufsize = None
        for sess in sessions:
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels,
                                   start_time=self.params.hfs_start_time,
                                   end_time=self.params.hfs_end_time, buffer_time=self.params.hfs_buf)

            eegs = eeg_reader.read()
            if eeg_reader.removed_bad_data():
                print 'REMOVED SOME BAD EVENTS !!!'
                sess_events = eegs['events'].values.view(np.recarray)
                n_events = len(sess_events)
                events = np.hstack((events[events.session!=sess],sess_events)).view(np.recarray)
                ev_order = np.argsort(events, order=('session','list','mstime'))
                events = events[ev_order]
                self.pass_object(self.pipeline.task+'_events', events)

            #eegs = eegs.add_mirror_buffer(duration=self.params.hfs_buf)

            if self.samplerate is None:
                self.samplerate = float(eegs.samplerate)
                winsize = int(round(self.samplerate*(self.params.hfs_end_time-self.params.hfs_start_time+2*self.params.hfs_buf)))
                bufsize = int(round(self.samplerate*self.params.hfs_buf))
                print 'samplerate =', self.samplerate, 'winsize =', winsize, 'bufsize =', bufsize
                pow_ev = np.empty(shape=n_hfs*winsize, dtype=float)
                self.wavelet_transform.init(self.params.width, self.params.hfs[0], self.params.hfs[-1], n_hfs, self.samplerate, winsize)

            print 'Computing PAL1 powers'

            sess_pow_mat = np.empty(shape=(n_events, n_bps, n_hfs), dtype=np.float)

            for i,ti in enumerate(bipolar_pairs):
                bp = ti['channel_str']
                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]

                bp_data = eegs[elec1] - eegs[elec2]
                bp_data.attrs['samplerate'] = self.samplerate

                bp_data = bp_data.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    self.wavelet_transform.multiphasevec(bp_data[ev][0:winsize], pow_ev)
                    pow_ev_stripped = np.reshape(pow_ev, (n_hfs,winsize))[:,bufsize:winsize-bufsize]
                    if self.params.log_powers:
                        np.log10(pow_ev_stripped, out=pow_ev_stripped)
                    sess_pow_mat[ev,i,:] = np.nanmean(pow_ev_stripped, axis=1)

            sess_pow_mat = zscore(sess_pow_mat, axis=0, ddof=1)
            sess_pow_mat = np.nanmean(sess_pow_mat, axis=2)

            self.pow_mat = np.vstack((self.pow_mat,sess_pow_mat)) if self.pow_mat is not None else sess_pow_mat
