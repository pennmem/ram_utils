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

        events = self.get_passed_object(task+'_events')
        recalls = np.array(events.recalled, dtype=np.bool)

        wavelets = self.get_passed_object('wavelets')
        n_freqs, n_bps, n_events, t_size = wavelets.shape
        wavelets = wavelets.reshape(-1)

        n_features = n_freqs * n_bps * (n_bps-1)
        self.ppc_features = np.empty(n_features*n_events, dtype=float)

        circular_stat.single_trial_ppc_all_features(recalls, wavelets, self.ppc_features, n_freqs, n_bps)

        self.ppc_features = self.ppc_features.reshape((n_features,n_events)).transpose()

        self.pass_object('ppc_features', self.ppc_features)

        joblib.dump(self.ppc_features, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ppc_features.pkl'))
