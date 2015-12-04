__author__ = 'm'


from RamPipeline import *

import numpy as np
from scipy.signal import resample
from scipy.stats.mstats import zscore
from ptsa.wavelet import phase_pow_multi
from sklearn.externals import joblib


class ComputePSPowers(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    def restore(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.experiment

        ps_pow_mat_pre = joblib.load(self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_pow_mat_pre.pkl'))
        ps_pow_mat_post = joblib.load(self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_pow_mat_post.pkl'))

        self.pass_object('ps_pow_mat_pre',ps_pow_mat_pre)
        self.pass_object('ps_pow_mat_post',ps_pow_mat_post)


    def run(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.experiment

        #fetching objects from other tasks
        events = self.get_passed_object(self.pipeline.experiment+'_events')
        channels = self.get_passed_object('channels')
        tal_info = self.get_passed_object('tal_info')

        sessions = np.unique(events.session)
        print experiment, 'sessions:', sessions

        ps_pow_mat_pre, ps_pow_mat_post = self.compute_ps_powers(events, sessions, channels, tal_info, experiment)

        joblib.dump(ps_pow_mat_pre, self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_pow_mat_pre.pkl'))
        joblib.dump(ps_pow_mat_post, self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_pow_mat_post.pkl'))

        self.pass_object('ps_pow_mat_pre',ps_pow_mat_pre)
        self.pass_object('ps_pow_mat_post',ps_pow_mat_post)

    def compute_ps_powers(self, events, sessions, channels, tal_info, experiment):
        n_freqs = len(self.params.freqs)
        n_bps = len(tal_info)
        nt = int((self.params.ps_end_time-self.params.ps_start_time+2*self.params.ps_buf+1e-5) * 50)
        nb = int((self.params.ps_buf+1e-5) * 50)

        pow_mat_pre = pow_mat_post = None

        for sess in sessions:
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            pre_start_time = self.params.ps_start_time - self.params.ps_offset
            pre_end_time = self.params.ps_end_time - self.params.ps_offset
            eegs_pre = sess_events.get_data(channels=channels, start_time=pre_start_time, end_time=pre_end_time,
                        buffer_time=self.params.ps_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)

            eegs_post = np.empty_like(eegs_pre)
            post_start_time = self.params.ps_offset
            post_end_time = self.params.ps_offset + (self.params.ps_end_time - self.params.ps_start_time)
            for i_ev in xrange(n_events):
                ev_offset = sess_events[i_ev].pulse_duration
                if ev_offset > 0:
                    if experiment == 'PS3' and sess_events[i_ev].nBursts > 0:
                        ev_offset *= sess_events[i_ev].nBursts + 1
                    ev_offset *= 0.001
                else:
                    ev_offset = 0.0
                eegs_post[:,i_ev:i_ev+1,:] = sess_events[i_ev:i_ev+1].get_data(channels=channels, start_time=post_start_time+ev_offset,
                            end_time=post_end_time+ev_offset, buffer_time=self.params.ps_buf,
                            eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)

            print 'Computing', experiment, 'powers'

            sess_pow_mat_pre = np.empty(shape=(n_events, n_bps, n_freqs), dtype=np.float)
            sess_pow_mat_post = np.empty_like(sess_pow_mat_pre)

            for i,ti in enumerate(tal_info):
                bp = ti['channel_str']
                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(channels == bp[0])[0][0]
                elec2 = np.where(channels == bp[1])[0][0]

                bp_data_pre = eegs_pre[elec1] - eegs_pre[elec2]
                bp_data_pre = bp_data_pre.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    pow_pre_ev = phase_pow_multi(self.params.freqs, bp_data_pre[ev], to_return='power')
                    if self.params.log_powers:
                        np.log10(pow_pre_ev, out=pow_pre_ev)
                    pow_pre_ev = resample(pow_pre_ev, num=nt, axis=1)
                    sess_pow_mat_pre[ev,i,:] = np.mean(pow_pre_ev[:,nb:-nb], axis=1)

                bp_data_post = eegs_post[elec1] - eegs_post[elec2]
                bp_data_post = bp_data_post.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    pow_post_ev = phase_pow_multi(self.params.freqs, bp_data_post[ev], to_return='power')
                    if self.params.log_powers:
                        np.log10(pow_post_ev, out=pow_post_ev)
                    pow_post_ev = resample(pow_post_ev, num=nt, axis=1)
                    sess_pow_mat_post[ev,i,:] = np.mean(pow_post_ev[:,nb:-nb], axis=1)

            sess_pow_mat_pre = sess_pow_mat_pre.reshape((n_events, n_bps*n_freqs))
            sess_pow_mat_pre = zscore(sess_pow_mat_pre, axis=0, ddof=1)

            sess_pow_mat_post = sess_pow_mat_post.reshape((n_events, n_bps*n_freqs))
            sess_pow_mat_post = zscore(sess_pow_mat_post, axis=0, ddof=1)

            pow_mat_pre = np.vstack((pow_mat_pre,sess_pow_mat_pre)) if pow_mat_pre is not None else sess_pow_mat_pre
            pow_mat_post = np.vstack((pow_mat_post,sess_pow_mat_post)) if pow_mat_post is not None else sess_pow_mat_post

        return pow_mat_pre, pow_mat_post
