__author__ = 'm'

from RamPipeline import *
import numpy as np

import os
import os.path
import re
import numpy as np
from scipy.io import loadmat
from scipy.stats.mstats import zscore

from ptsa.data.events import Events
from ptsa.data.rawbinwrapper import RawBinWrapper

from get_bipolar_subj_elecs import get_bipolar_subj_elecs

from ptsa.wavelet import phase_pow_multi

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib


class ComputeFR1Powers(RamTask):
    def __init__(self, params, task, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.task = task
        self.params = params

    def restore(self):
        pow_mat = joblib.load( self.get_path_to_resource_in_workspace(self.pipeline.subject_id+'_pow_mat.pkl'))
        recalls = joblib.load(self.get_path_to_resource_in_workspace(self.pipeline.subject_id+'_recalls.pkl'))

        self.pass_object('pow_mat',pow_mat)
        self.pass_object('recalls',recalls)

    def run(self):


        events = self.get_passed_object(self.task+'_events')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        channels = self.get_passed_object('channels')
        tal_info = self.get_passed_object('tal_info')
        pow_mat, recalls = self.compute_powers(events, sessions, channels, tal_info)


        joblib.dump(pow_mat, self.get_path_to_resource_in_workspace(self.pipeline.subject_id+'_pow_mat.pkl'))
        joblib.dump(recalls, self.get_path_to_resource_in_workspace(self.pipeline.subject_id+'_recalls.pkl'))

        self.pass_object('pow_mat',pow_mat)
        self.pass_object('recalls',recalls)



    def compute_powers(self, events, sessions, channels, tal_info):
        n_freqs = len(self.params.freqs)
        n_bps = len(tal_info)

        pow_mat = None
        recalls = None

        for sess in sessions:
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            sess_recalls = sess_events.recalled

            print 'Loading EEG for', n_events, 'events of session', sess

            eegs = sess_events.get_data(channels=channels, start_time=self.params.fr1_start_time, end_time=self.params.fr1_end_time,
                                        buffer_time=self.params.fr1_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)

            print 'Computing FR1 powers'

            sess_pow_mat = np.empty(shape=(n_events, n_bps, n_freqs), dtype=np.float)

            for i,ti in enumerate(tal_info):
                bp = ti['channel_str']
                print 'Computing Power for bipolar pair = ', bp
                elec1 = np.where(channels == bp[0])[0][0]
                elec2 = np.where(channels == bp[1])[0][0]
                bp_data = eegs[elec1] - eegs[elec2]
                bp_data = bp_data.filtered([58,62], filt_type='stop', order=1)
                for ev in xrange(n_events):
                    pow_ev = phase_pow_multi(self.params.freqs, bp_data[ev], to_return='power')
                    pow_ev = pow_ev.remove_buffer(self.params.fr1_buf)
                    pow_ev = np.nanmean(pow_ev, 1)
                    sess_pow_mat[ev,i,:] = pow_ev

            sess_pow_mat = sess_pow_mat.reshape((n_events, n_bps*n_freqs))
            sess_pow_mat = zscore(sess_pow_mat, axis=0, ddof=1)

            pow_mat = np.vstack((pow_mat,sess_pow_mat)) if pow_mat is not None else sess_pow_mat
            recalls = np.hstack((recalls,sess_recalls)) if recalls is not None else sess_recalls

        return pow_mat, recalls
