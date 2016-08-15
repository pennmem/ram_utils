from RamPipeline import *

from random import shuffle
import numpy as np
from scipy.stats import zmap
import circular_stat

from sklearn.externals import joblib

from ReportUtils import ReportRamTask


class ComputePPCFeatures(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputePPCFeatures,self).__init__(mark_as_completed)
        self.params = params
        self.ppc_features = None
        self.theta_sum_recalls = None
        self.theta_sum_non_recalls = None

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.ppc_features = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-ppc_features.pkl'))
        self.theta_sum_recalls = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-theta_sum_recalls.pkl'))
        self.theta_sum_non_recalls = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-theta_sum_non_recalls.pkl'))

        self.pass_object('ppc_features', self.ppc_features)
        self.pass_object('theta_sum_recalls', self.theta_sum_recalls)
        self.pass_object('theta_sum_non_recalls', self.theta_sum_non_recalls)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        print "Computing PPC features"

        wavelets = self.get_passed_object('wavelets')
        events = self.get_passed_object(task+'_events')

        n_freqs, n_bps, n_events, t_size = wavelets.shape
        n_features = n_freqs * n_bps * (n_bps-1) / 2

        self.theta_sum_recalls = dict()
        self.theta_sum_non_recalls = dict()

        sessions = np.unique(events.session)

        for sess in sessions:
            print 'Session', sess

            sess_sel = (events.session==sess)
            sess_events = events[sess_sel]
            n_events = len(sess_events)

            sess_wavelets = wavelets[:,:,sess_sel,:].ravel()

            sess_recalls = np.array(sess_events.recalled, dtype=np.bool).ravel()
            sess_ppc_features = np.empty(n_features*n_events, dtype=float)
            sess_theta_sum_recalls = np.zeros(n_features*t_size, dtype=np.complex)
            sess_theta_sum_non_recalls = np.zeros(n_features*t_size, dtype=np.complex)

            circular_stat.single_trial_ppc_all_features(sess_recalls, sess_wavelets, sess_ppc_features, sess_theta_sum_recalls, sess_theta_sum_non_recalls, n_freqs, n_bps, 40)

            sess_ppc_features = sess_ppc_features.reshape((n_features,n_events)).transpose()

            self.ppc_features = np.concatenate((self.ppc_features,sess_ppc_features), axis=0) if self.ppc_features is not None else sess_ppc_features
            self.theta_sum_recalls[sess] = sess_theta_sum_recalls
            self.theta_sum_non_recalls[sess] = sess_theta_sum_non_recalls

        self.pass_object('ppc_features', self.ppc_features)
        self.pass_object('theta_sum_recalls', self.theta_sum_recalls)
        self.pass_object('theta_sum_non_recalls', self.theta_sum_non_recalls)

        joblib.dump(self.ppc_features, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ppc_features.pkl'))
        joblib.dump(self.theta_sum_recalls, self.get_path_to_resource_in_workspace(subject + '-' + task + '-theta_sum_recalls.pkl'))
        joblib.dump(self.theta_sum_non_recalls, self.get_path_to_resource_in_workspace(subject + '-' + task + '-theta_sum_non_recalls.pkl'))
