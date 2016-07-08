__author__ = 'm'

from RamPipeline import *

import numpy as np
from morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.readers import EEGReader
from ReportUtils import ReportRamTask

# from ptsa.data.TimeSeriesX import TimeSeriesX
# from scipy.signal import resample


class ComputeFR1Wavelets(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeFR1Wavelets,self).__init__(mark_as_completed)
        self.params = params
        self.wavelets = None
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

        self.wavelets = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-wavelets.pkl'))
        self.samplerate = joblib.load(self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

        self.pass_object('wavelets', self.wavelets)
        self.pass_object('samplerate', self.samplerate)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task+'_events')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')

        self.compute_wavelets(events, sessions, monopolar_channels, bipolar_pairs)

        self.pass_object('wavelets', self.wavelets)
        self.pass_object('samplerate', self.samplerate)

        joblib.dump(self.wavelets, self.get_path_to_resource_in_workspace(subject + '-' + task + '-wavelets.pkl'))
        joblib.dump(self.samplerate, self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

    def compute_wavelets(self, events, sessions, monopolar_channels, bipolar_pairs):
        n_events = len(events)
        n_freqs = len(self.params.freqs)
        n_bps = len(bipolar_pairs)

        self.wavelets = None
        cur_ev = 0

        wav_ev = None
        winsize = bufsize = tsize = None
        for sess in sessions:
            sess_events = events[events.session == sess]
            n_sess_events = len(sess_events)

            print 'Loading EEG for', n_sess_events, 'events of session', sess

            eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels,
                                   start_time=self.params.fr1_start_time,
                                   end_time=self.params.fr1_end_time, buffer_time=self.params.fr1_buf)

            eegs = eeg_reader.read()
            if eeg_reader.removed_bad_data():
                # NB: this is not supported yet in this pipeline
                print 'REMOVED SOME BAD EVENTS !!!'
                import sys
                sys.exit(0)
                # sess_events = eegs['events'].values.view(np.recarray)
                # n_sess_events = len(sess_events)
                # events = np.hstack((events[events.session!=sess],sess_events)).view(np.recarray)
                # ev_order = np.argsort(events, order=('session','list','mstime'))
                # events = events[ev_order]
                # self.pass_object(self.pipeline.task+'_events', events)


            #eegs = eegs.add_mirror_buffer(duration=self.params.fr1_buf)

            if self.samplerate is None:
                self.samplerate = float(eegs.samplerate)
                winsize = int(round(self.samplerate*(self.params.fr1_end_time-self.params.fr1_start_time+2*self.params.fr1_buf)))
                bufsize = int(round(self.samplerate*self.params.fr1_buf))
                tsize = winsize - 2*bufsize
                print 'samplerate =', self.samplerate, 'winsize =', winsize, 'bufsize =', bufsize
                self.wavelet_transform.init(self.params.width, self.params.freqs[0], self.params.freqs[-1], n_freqs, self.samplerate, winsize)
                wav_ev = np.empty(shape=n_freqs*winsize, dtype=np.complex)
                self.wavelets = np.empty(shape=(n_freqs, n_bps, n_events, tsize), dtype=np.complex)

            print 'Computing FR1 wavelets'

            eegs = eegs.filtered([58,62], filt_type='stop', order=self.params.filt_order)

            for i,bp in enumerate(bipolar_pairs):
                print 'Computing wavelets for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]

                bp_data = eegs[elec1] - eegs[elec2]

                for ev in xrange(n_sess_events):
                    self.wavelet_transform.multiphasevec_complex(bp_data[ev][0:winsize], wav_ev)
                    self.wavelets[:,i,cur_ev+ev,:] = np.reshape(wav_ev, (n_freqs,winsize))[:,bufsize:winsize-bufsize]

            cur_ev += n_sess_events
