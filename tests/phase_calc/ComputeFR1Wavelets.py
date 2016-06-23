__author__ = 'm'

from RamPipeline import *

import numpy as np
from morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.readers import EEGReader
from ReportUtils import ReportRamTask

import pycircstat.descriptive


class ComputeFR1Wavelets(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeFR1Wavelets,self).__init__(mark_as_completed)
        self.params = params
        self.phase_mat = None
        self.phase_diff_mat = None
        self.samplerate = None
        self.wavelet_transform = MorletWaveletTransform()

    def initialize(self):
        task_prefix = 'cat' if self.pipeline.task == 'RAM_CatFR1' else ''
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name=task_prefix+'fr1_events',
                                        access_path = ['experiments',task_prefix+'fr1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.phase_diff_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-phase_diff_mat.pkl'))
        self.samplerate = joblib.load(self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

        self.pass_object('phase_diff_mat', self.phase_diff_mat)
        self.pass_object('samplerate', self.samplerate)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task+'_events')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        bipolar_pair_pairs = self.get_passed_object('bipolar_pair_pairs')

        self.compute_wavelets(events, sessions, monopolar_channels, bipolar_pairs)
        self.compute_phase_differences(bipolar_pair_pairs)

        self.pass_object('phase_diff_mat', self.phase_diff_mat)
        self.pass_object('samplerate', self.samplerate)

        joblib.dump(self.phase_diff_mat, self.get_path_to_resource_in_workspace(subject + '-' + task + '-phase_diff_mat.pkl'))
        joblib.dump(self.samplerate, self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

    def compute_wavelets(self, events, sessions, monopolar_channels, bipolar_pairs):
        n_freqs = len(self.params.freqs)
        n_bps = len(bipolar_pairs)

        self.phase_mat = None

        phase_ev = None
        winsize = bufsize = tsize = None
        for sess in sessions:
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels,
                                   start_time=self.params.fr1_start_time,
                                   end_time=self.params.fr1_end_time, buffer_time=self.params.fr1_buf)

            eegs = eeg_reader.read()
            if eeg_reader.removed_bad_data():
                print 'REMOVED SOME BAD EVENTS !!!'
                sess_events = eegs['events'].values.view(np.recarray)
                n_events = len(sess_events)
                events = np.hstack((events[events.session!=sess],sess_events)).view(np.recarray)
                ev_order = np.argsort(events, order=('session','list','mstime'))
                events = events[ev_order]
                self.pass_object(self.pipeline.task+'_events', events)


            #eegs = eegs.add_mirror_buffer(duration=self.params.fr1_buf)

            if self.samplerate is None:
                self.samplerate = float(eegs.samplerate)
                winsize = int(round(self.samplerate*(self.params.fr1_end_time-self.params.fr1_start_time+2*self.params.fr1_buf)))
                bufsize = int(round(self.samplerate*self.params.fr1_buf))
                tsize = winsize - 2*bufsize
                print 'samplerate =', self.samplerate, 'winsize =', winsize, 'bufsize =', bufsize
                pow_ev = np.empty(shape=n_freqs*winsize, dtype=float)
                phase_ev = np.empty(shape=n_freqs*winsize, dtype=float)
                self.wavelet_transform.init(self.params.width, self.params.freqs[0], self.params.freqs[-1], n_freqs, self.samplerate, winsize)

            print 'Computing FR1 wavelets'

            sess_phase_mat = np.empty(shape=(n_events, n_bps, n_freqs, tsize), dtype=np.float)

            for i,bp in enumerate(bipolar_pairs):
                print 'Computing wavelets for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]

                bp_data = eegs[elec1] - eegs[elec2]
                bp_data.attrs['samplerate'] = self.samplerate

                for ev in xrange(n_events):
                    self.wavelet_transform.multiphasevec(bp_data[ev][0:winsize], pow_ev, phase_ev)
                    sess_phase_mat[ev,i,...] = np.reshape(phase_ev, (n_freqs,winsize))[:,bufsize:winsize-bufsize]

            self.phase_mat = np.concatenate((self.phase_mat,sess_phase_mat), axis=0) if self.phase_mat is not None else sess_phase_mat

    def compute_phase_differences(self, bipolar_pair_pairs):
        time_bin_len = int(round(self.samplerate * self.params.fr1_bin))
        n_bp_pairs = len(bipolar_pair_pairs)

        n_events,n_bps,n_freqs,tsize = self.phase_mat.shape
        n_bins = tsize // time_bin_len

        self.phase_diff_mat = np.empty(shape=(n_events,n_bp_pairs,n_freqs,n_bins), dtype=np.float)

        for j,bpp in enumerate(bipolar_pair_pairs):
            print "Computing phase differences for bp pair", bpp
            for i in xrange(n_events):
                bp1,bp2 = bpp
                phase_diff = self.phase_mat[i,bp1,...] - self.phase_mat[i,bp2,...]
                #phase_diff = np.where(phase_diff>=0.0, phase_diff, phase_diff+2*np.pi)
                for k in xrange(n_bins):
                    self.phase_diff_mat[i,j,:,k] = pycircstat.descriptive.mean(phase_diff[:,k*time_bin_len:(k+1)*time_bin_len], axis=1)
