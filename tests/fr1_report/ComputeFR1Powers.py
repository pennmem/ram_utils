__author__ = 'm'

from RamPipeline import *

import numpy as np
from scipy.signal import resample
from ptsa.wavelet import phase_pow_multi
from sklearn.externals import joblib


class ComputeFR1Powers(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.pow_mat = None

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-pow_mat.pkl'))
        self.pass_object('pow_mat', self.pow_mat)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task+'_events')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        channels = self.get_passed_object('channels')
        tal_info = self.get_passed_object('tal_info')
        self.compute_powers(events, sessions, channels, tal_info)

        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-' + task + '-pow_mat.pkl'))
        self.pass_object('pow_mat', self.pow_mat)

    def compute_powers(self, events, sessions, channels, tal_info):
        n_freqs = len(self.params.freqs)
        n_bps = len(tal_info)
        nt = int((self.params.fr1_end_time-self.params.fr1_start_time+2*self.params.fr1_buf+1e-5) * 50)
        nb = int((self.params.fr1_buf+1e-5) * 50)
        n_times = nt - 2*nb

        self.pow_mat = None

        for sess in sessions:
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            eegs = sess_events.get_data(channels=channels, start_time=self.params.fr1_start_time, end_time=self.params.fr1_end_time,
                                        buffer_time=self.params.fr1_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)

            print 'Computing FR1 powers'

            sess_pow_mat = np.empty(shape=(n_events, n_bps, n_freqs, n_times), dtype=np.float)

            for i,ti in enumerate(tal_info):
                bp = ti['channel_str']
                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(channels == bp[0])[0][0]
                elec2 = np.where(channels == bp[1])[0][0]
                bp_data = eegs[elec1] - eegs[elec2]
                bp_data = bp_data.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    pow_ev = phase_pow_multi(self.params.freqs, bp_data[ev], to_return='power')
                    #
                    # if not np.all(pow_ev > 0.):
                    #     raise RuntimeError('BEFORE LOG: negative power for electrodes'+str(bp)+' event=%s'%str((ev.session,ev.list,ev.serialpos)))

                    if self.params.log_powers:
                        pow_ev[pow_ev<1.0] = 1.0 # we do not allow power values to be less than 1 to avoid potential logarithmic instability
                        np.log10(pow_ev, out=pow_ev)

                    # if not np.all(np.isfinite(pow_ev)):
                    #     raise RuntimeError('AFTER LOG: NAN power for electrodes'+str(bp)+' event=%s'%str((ev.session,ev.list,ev.serialpos)))

                    pow_ev = resample(pow_ev, num=nt, axis=1)

                    # if not np.all(np.isfinite(pow_ev)):
                    #     raise RuntimeError('AFTER RESAMPLE: NAN power for electrodes'+str(bp)+' event=%s'%str((ev.session,ev.list,ev.serialpos)))

                    sess_pow_mat[ev,i,:,:] = pow_ev[:,nb:-nb]

            self.pow_mat = np.concatenate((self.pow_mat,sess_pow_mat), axis=0) if self.pow_mat is not None else sess_pow_mat
