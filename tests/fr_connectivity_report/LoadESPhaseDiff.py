from RamPipeline import *

from ReportUtils import ReportRamTask

import numpy as np
from scipy.io import loadmat

from sklearn.externals import joblib


class LoadESPhaseDiff(ReportRamTask):
    def __init__(self, params, mark_as_completed=False):
        super(LoadESPhaseDiff,self).__init__(mark_as_completed)
        self.params = params
        self.phase_diff_mat = None

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.phase_diff_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-phase_diff_mat.pkl'))
        self.pass_object('phase_diff_mat', self.phase_diff_mat)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task+'_events')
        n_events = len(events)
        recalls = np.array(events.recalled, dtype=np.bool)

        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        bipolar_pair_pairs = self.get_passed_object('bipolar_pair_pairs')
        n_bp_pairs = len(bipolar_pair_pairs)

        n_freqs = len(self.params.freqs)
        n_bins = self.params.fr1_n_bins

        self.phase_diff_mat = np.empty(shape=(n_bp_pairs, n_freqs, n_bins, n_events), dtype=np.complex)

        for i,bp_pair in enumerate(bipolar_pair_pairs):
            bp1 = bp_pair[0] + 1
            bp2 = bp_pair[1] + 1
            es_mat_file = '/scratch/esolo/plv_gamma_phasediff_200msavg/%s/%d_%d.mat' % (subject,bp1,bp2)
            es_phase_diff = loadmat(es_mat_file)
            recall_phase_diff = es_phase_diff['rem_phasediff']
            non_recall_phase_diff = es_phase_diff['nrem_phasediff']
            # print 'My shape =', self.phase_diff_mat[i,:,:,:].shape
            # print 'ES transposed shape =', np.transpose(recall_phase_diff, axes=[1,2,0]).shape
            # self.phase_diff_mat[i,:,:,recalls] = np.transpose(recall_phase_diff, axes=[1,2,0])
            # self.phase_diff_mat[i,:,:,~recalls] = np.transpose(non_recall_phase_diff, axes=[1,2,0])

            # there is probably a better way to do it in Numpy but I don't know how
            i_es_recalls = 0
            i_es_non_recalls = 0
            for j in xrange(n_events):
                if recalls[j]:
                    self.phase_diff_mat[i,:,:,j] = np.exp(1j*recall_phase_diff[i_es_recalls,:,:])
                    i_es_recalls += 1
                else:
                    self.phase_diff_mat[i,:,:,j] = np.exp(1j*non_recall_phase_diff[i_es_non_recalls,:,:])
                    i_es_non_recalls += 1

        self.pass_object('phase_diff_mat', self.phase_diff_mat)
        joblib.dump(self.phase_diff_mat, self.get_path_to_resource_in_workspace(subject + '-' + task + '-phase_diff_mat.pkl'))
