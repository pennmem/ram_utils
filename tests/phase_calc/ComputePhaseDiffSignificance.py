from RamPipeline import *

from random import shuffle
import numpy as np
from circular_stat import resultant_vector_length

from sklearn.externals import joblib

from ReportUtils import ReportRamTask


def compute_f_stat(phase_diff_mat, recalls, f_stat_mat):
    n_events, n_bp_pairs, n_freqs, n_bins = phase_diff_mat.shape

    n_recalls = recalls.sum()
    n_non_recalls = n_events - n_recalls
    for i in xrange(n_bp_pairs):
        for f in xrange(n_freqs):
            for j in xrange(n_bins):
                phase_diffs_1 = np.array(phase_diff_mat[recalls,i,f,j], copy=True)
                r_recalls = resultant_vector_length(phase_diffs_1)

                phase_diffs_2 = np.array(phase_diff_mat[~recalls,i,f,j], copy=True)
                r_non_recalls = resultant_vector_length(phase_diffs_2)

                f_stat_mat[i,f,j] = ((n_recalls-1)*(n_non_recalls-r_non_recalls)) / ((n_non_recalls-1)*(n_recalls-r_recalls))


class ComputePhaseDiffSignificance(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputePhaseDiffSignificance,self).__init__(mark_as_completed)
        self.params = params
        self.f_stat_mat = None
        self.shuffled_f_stat_mat = None

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.f_stat_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-f_stat_mat.pkl'))
        self.shuffled_f_stat_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-shuffled_f_stat_mat.pkl'))

        self.pass_object('f_stat_mat', self.f_stat_mat)
        self.pass_object('shuffled_f_stat_mat', self.shuffled_f_stat_mat)

    def run(self):
        print "Computing f-stats"

        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task+'_events')

        phase_diff_mat = self.get_passed_object('phase_diff_mat')

        _, n_bp_pairs, n_freqs, n_bins = phase_diff_mat.shape
        self.f_stat_mat = np.empty(shape=(n_bp_pairs,n_freqs,n_bins), dtype=np.float)

        recalls = np.array(events.recalled, dtype=np.bool)

        compute_f_stat(phase_diff_mat, recalls, self.f_stat_mat)

        n_perms = self.params.n_perms
        self.shuffled_f_stat_mat = np.empty(shape=(n_perms,n_bp_pairs,n_freqs,n_bins), dtype=np.float)
        for i in xrange(n_perms):
            print "Permutation", i
            shuffle(recalls)
            compute_f_stat(phase_diff_mat, recalls, self.shuffled_f_stat_mat[i])

        self.pass_object('f_stat_mat', self.f_stat_mat)
        self.pass_object('shuffled_f_stat_mat', self.shuffled_f_stat_mat)

        joblib.dump(self.f_stat_mat, self.get_path_to_resource_in_workspace(subject + '-' + task + '-f_stat_mat.pkl'))
        joblib.dump(self.shuffled_f_stat_mat, self.get_path_to_resource_in_workspace(subject + '-' + task + '-shuffled_f_stat_mat.pkl'))
