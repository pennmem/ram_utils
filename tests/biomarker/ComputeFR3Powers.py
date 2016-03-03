__author__ = 'm'

from RamPipeline import *

import numpy as np
#from scipy.stats import describe
#from ptsa.wavelet import phase_pow_multi
from morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.events import Events


class ComputeFR3Powers(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.wavelet_transform = MorletWaveletTransform()

    def restore(self):
        subject = self.pipeline.subject
        task3 = self.pipeline.task3

        self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task3 + '-pow_mat.pkl'))
        self.pass_object('pow_mat', self.pow_mat)

    def run(self):
        subject = self.pipeline.subject
        task3 = self.pipeline.task3

        events = self.get_passed_object(task3 + '_events')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        channels = self.get_passed_object('channels')
        tal_info = self.get_passed_object('tal_info')
        self.compute_powers(events, sessions, channels, tal_info)

        self.pass_object('pow_mat', self.pow_mat)
        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-' + task3 + '-pow_mat.pkl'))

    def compute_powers(self, events, sessions, channels, tal_info):
        n_freqs = len(self.params.freqs)
        n_bps = len(tal_info)

        self.wavelet_transform.init(5, self.params.freqs[0], self.params.freqs[-1], n_freqs, 1000.0, 4096)

        self.pow_mat = None

        pow_ev = np.empty(shape=n_freqs*4096, dtype=float)
        for sess in sessions:
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            eegs = Events(sess_events).get_data(channels=channels, start_time=self.params.fr1_start_time, end_time=self.params.fr1_end_time,
                                        buffer_time=self.params.fr1_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)

            #print describe(eegs)

            # mirroring
            eegs[...,:1365] = eegs[...,2730:1365:-1]
            eegs[...,2731:4096] = eegs[...,2729:1364:-1]

            print 'Computing', self.pipeline.task3, 'powers'

            sess_pow_mat = np.empty(shape=(n_events, n_bps, n_freqs), dtype=np.float)

            for i,ti in enumerate(tal_info):
                bp = ti['channel_str']
                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(channels == bp[0])[0][0]
                elec2 = np.where(channels == bp[1])[0][0]
                bp_data = eegs[elec1] - eegs[elec2]
                bp_data = bp_data.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    self.wavelet_transform.multiphasevec(bp_data[ev][0:4096], pow_ev)
                    if np.min(pow_ev) < 0.0:
                        print ev, events[ev]
                        joblib.dump(bp_data[ev], 'bad_bp_ev%d'%ev)
                        joblib.dump(eegs[elec1][ev], 'bad_elec1_ev%d'%ev)
                        joblib.dump(eegs[elec2][ev], 'bad_elec2_ev%d'%ev)
                        print 'Negative powers detected'
                        import sys
                        sys.exit(1)

                    if self.params.log_powers:
                        np.log10(pow_ev, out=pow_ev)

                    pow_ev_stripped = np.reshape(pow_ev, (n_freqs,4096))[:,1365:1365+1366]
                    sess_pow_mat[ev,i,:] = np.nanmean(pow_ev_stripped, axis=1)

            self.pow_mat = np.concatenate((self.pow_mat,sess_pow_mat), axis=0) if self.pow_mat is not None else sess_pow_mat

        self.pow_mat = np.reshape(self.pow_mat, (len(events), n_bps*n_freqs))
