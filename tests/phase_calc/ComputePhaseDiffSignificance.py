from RamPipeline import *

from random import shuffle
import numpy as np
from scipy.stats import zmap
from circular_stat import compute_f_stat, compute_zscores

from sklearn.externals import joblib

from ReportUtils import ReportRamTask


#def compute_f_stat(phase_diff_mat, recalls, f_stat_mat):
#    n_bp_pairs, n_freqs, n_bins, n_events = phase_diff_mat.shape
#
#    n_recalls = recalls.sum()
#    n_non_recalls = n_events - n_recalls
#    for i in xrange(n_bp_pairs):
#        for f in xrange(n_freqs):
#            for j in xrange(n_bins):
#                phase_diffs_1 = np.array(phase_diff_mat[i,f,j,recalls], copy=True)
#                r_recalls = resultant_vector_length(phase_diffs_1)
#
#                phase_diffs_2 = np.array(phase_diff_mat[i,f,j,~recalls], copy=True)
#                r_non_recalls = resultant_vector_length(phase_diffs_2)
#
#                f_stat_mat[i,f,j] = ((n_recalls-1)*(n_non_recalls-r_non_recalls)) / ((n_non_recalls-1)*(n_recalls-r_recalls))


class ComputePhaseDiffSignificance(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputePhaseDiffSignificance,self).__init__(mark_as_completed)
        self.params = params
        self.zscore_mat = None
        self.connectivity_strength = None

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.connectivity_strength = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-connectivity_strength.pkl'))

        self.pass_object('connectivity_strength', self.connectivity_strength)

    def run(self):
        print "Computing f-stats"

        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task+'_events')

        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        bipolar_pair_pairs = self.get_passed_object('bipolar_pair_pairs')

        phase_diff_mat = self.get_passed_object('phase_diff_mat')

        n_bp_pairs, n_freqs, n_bins, n_events = phase_diff_mat.shape
        print 'n_bp_pairs =', n_bp_pairs, 'n_freqs =', n_freqs, 'n_bins =', n_bins, 'n_events =', n_events

        phase_diff_mat = phase_diff_mat.reshape(-1)
        recalls = np.array(events.recalled, dtype=np.bool)

        joblib.dump(recalls, self.get_path_to_resource_in_workspace(subject + '-' + task + '-recalls.pkl'))

        n_perms = self.params.n_perms
        shuffle_mat = np.empty(shape=(n_perms+1,n_bp_pairs*n_freqs*n_bins), dtype=np.float)
        compute_f_stat(phase_diff_mat, recalls, shuffle_mat[0])
        for i in xrange(1,n_perms+1):
            print "Permutation", i
            shuffle(recalls)
            compute_f_stat(phase_diff_mat, recalls, shuffle_mat[i])

        joblib.dump(shuffle_mat.reshape((n_perms+1,n_bp_pairs,n_freqs,n_bins)), self.get_path_to_resource_in_workspace(subject + '-' + task + '-fstat_mat.pkl'))

        shuffle_mat = shuffle_mat.reshape(-1)
        compute_zscores(shuffle_mat, n_perms+1)

        shuffle_mat = shuffle_mat.reshape((n_perms+1,n_bp_pairs,n_freqs,n_bins))

        joblib.dump(shuffle_mat, self.get_path_to_resource_in_workspace(subject + '-' + task + '-zscore_mat.pkl'))

        shuffle_mat = shuffle_mat.sum(axis=(2,3))

        n_bps = len(bipolar_pairs)
        shuffle_mat_bp = np.zeros(shape=(n_perms+1,n_bps), dtype=np.float)
        for i,bp_pair in enumerate(bipolar_pair_pairs):
            bp1, bp2 = bp_pair
            shuffle_mat_bp[:,bp1] += shuffle_mat[:,i]
            shuffle_mat_bp[:,bp2] += shuffle_mat[:,i]

        self.connectivity_strength = -1.0 * zmap(shuffle_mat_bp[0,:], shuffle_mat_bp[1:,:], axis=0, ddof=1)

        self.pass_object('connectivity_strength', self.connectivity_strength)

        joblib.dump(self.connectivity_strength, self.get_path_to_resource_in_workspace(subject + '-' + task + '-connectivity_strength.pkl'))
