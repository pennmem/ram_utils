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

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.ppc_features = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-ppc_features.pkl'))

        self.pass_object('ppc_features', self.ppc_features)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        print "Computing PPC features"

        wavelets = self.get_passed_object('wavelets')
        events = self.get_passed_object(task+'_events')

        n_freqs, n_bps, n_events, t_size = wavelets.shape
        n_features = n_freqs * n_bps * (n_bps-1) / 2
        #self.ppc_features = np.empty(shape=(n_events,n_features), dtype=float)

        sessions = np.unique([events.session])

        for sess in sessions:
            print 'Session', sess

            sess_sel = (events.session==sess)
            sess_events = events[sess_sel]
            n_events = len(sess_events)

            sess_wavelets = wavelets[:,:,sess_sel,:].ravel()

            sess_recalls = np.array(sess_events.recalled, dtype=np.bool).ravel()
            sess_ppc_features = np.empty(n_features*n_events, dtype=float)

            circular_stat.single_trial_ppc_all_features(sess_recalls, sess_wavelets, sess_ppc_features, n_freqs, n_bps, 50)

            sess_ppc_features = sess_ppc_features.reshape((n_features,n_events)).transpose()

            self.ppc_features = np.concatenate((self.ppc_features,sess_ppc_features), axis=0) if self.ppc_features is not None else sess_ppc_features

        self.pass_object('ppc_features', self.ppc_features)

        joblib.dump(self.ppc_features, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ppc_features.pkl'))
