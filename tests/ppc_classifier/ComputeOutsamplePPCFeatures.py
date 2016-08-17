from RamPipeline import *

import numpy as np
import circular_stat

from sklearn.externals import joblib

from ReportUtils import ReportRamTask

from scipy.stats import describe


class ComputeOutsamplePPCFeatures(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeOutsamplePPCFeatures,self).__init__(mark_as_completed)
        self.params = params
        self.outsample_ppc_features = None
        self.theta_avg_recalls = dict()
        self.theta_avg_non_recalls = dict()

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.outsample_ppc_features = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-outsample_ppc_features.pkl'))

        self.pass_object('outsample_ppc_features', self.outsample_ppc_features)

    def prepare_theta_avgs(self, events, theta_sum_recalls, theta_sum_non_recalls, n_features, t_size):
        sessions = np.unique(events.session)
        for sess in sessions:
            outsess_theta_avg_recalls = np.zeros(n_features*t_size, dtype=np.complex)
            outsess_theta_avg_non_recalls = np.zeros(n_features*t_size, dtype=np.complex)
            outsess_events = events[events.session!=sess]
            n_recalls = np.sum(outsess_events.recalled)
            n_non_recalls = len(outsess_events) - n_recalls
            for sess1 in sessions:
                if sess1 != sess:
                    outsess_theta_avg_recalls += theta_sum_recalls[sess1]
                    outsess_theta_avg_non_recalls += theta_sum_non_recalls[sess1]
            self.theta_avg_recalls[sess] = outsess_theta_avg_recalls / n_recalls
            self.theta_avg_non_recalls[sess] = outsess_theta_avg_non_recalls / n_non_recalls
            #print 'theta_avg_recalls:', describe(np.real(self.theta_avg_recalls[sess].reshape(-1)))
            #print 'theta_avg_non_recalls:', describe(np.imag(self.theta_avg_non_recalls[sess].reshape(-1)))
    #         import sys
    #         sys.exit(0)

    # def prepare_theta_avgs(self, events, theta_sum_recalls, theta_sum_non_recalls, n_features, t_size):
    #     sessions = np.unique(events.session)
    #     for sess in sessions:
    #         sess_events = events[events.session==sess]
    #         n_recalls = np.sum(sess_events.recalled)
    #         n_non_recalls = len(sess_events) - n_recalls
    #         self.theta_avg_recalls[sess] = theta_sum_recalls[sess] / n_recalls
    #         self.theta_avg_non_recalls[sess] = theta_sum_non_recalls[sess] / n_non_recalls

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        print "Computing outsample PPC features"

        events = self.get_passed_object(task+'_events')
        wavelets = self.get_passed_object('wavelets')
        theta_sum_recalls = self.get_passed_object('theta_sum_recalls')
        theta_sum_non_recalls = self.get_passed_object('theta_sum_non_recalls')

        n_freqs, n_bps, n_events, t_size = wavelets.shape
        n_features = n_freqs * n_bps * (n_bps-1) / 2

        self.prepare_theta_avgs(events, theta_sum_recalls, theta_sum_non_recalls, n_features, t_size)

        for i,ev in enumerate(events):
            ev_wavelets = wavelets[...,i,:].ravel()
            sess = ev.session
            ev_outsample_ppc_features = np.empty(n_features, dtype=float)
            circular_stat.single_trial_outsample_ppc_features(ev_wavelets, self.theta_avg_recalls[sess], self.theta_avg_non_recalls[sess], ev_outsample_ppc_features, n_freqs, n_bps, 40)

            #cheat: for debugging purposes
            #if not ev.recalled:
            #    ev_outsample_ppc_features = -ev_outsample_ppc_features

            self.outsample_ppc_features = np.concatenate((self.outsample_ppc_features,ev_outsample_ppc_features), axis=0) if self.outsample_ppc_features is not None else ev_outsample_ppc_features

        self.pass_object('outsample_ppc_features', self.outsample_ppc_features)

        joblib.dump(self.outsample_ppc_features, self.get_path_to_resource_in_workspace(subject + '-' + task + '-outsample_ppc_features.pkl'))
