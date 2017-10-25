import os
from random import shuffle
import numpy as np
from scipy.stats import zmap
from ptsa.extensions.circular_stat.circular_stat import compute_f_stat, compute_zscores

from sklearn.externals import joblib
from ptsa.data.readers.IndexReader import JsonIndexReader
from ReportUtils import ReportRamTask

import hashlib


class ComputePhaseDiffSignificance(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputePhaseDiffSignificance,self).__init__(mark_as_completed)
        self.params = params
        self.zscore_mat = None
        self.connectivity_strength = None

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        fr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='FR1')))
        for fname in fr1_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        catfr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='catFR1')))
        for fname in catfr1_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def restore(self):
        subject = self.pipeline.subject

        self.connectivity_strength = joblib.load(self.get_path_to_resource_in_workspace(subject + '-connectivity_strength.pkl'))
        self.pass_object('connectivity_strength', self.connectivity_strength)

    def run(self):
        print "Computing f-stats"

        subject = self.pipeline.subject

        events = self.get_passed_object('events')

        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        bipolar_pair_pairs = self.get_passed_object('bipolar_pair_pairs')

        phase_diff_mat = self.get_passed_object('phase_diff_mat')

        n_bp_pairs, n_freqs, n_bins, n_events = phase_diff_mat.shape
        print 'n_bp_pairs =', n_bp_pairs, 'n_freqs =', n_freqs, 'n_bins =', n_bins, 'n_events =', n_events

        phase_diff_mat = phase_diff_mat.reshape(-1)
        recalls = np.array(events.recalled, dtype=np.bool)

        if self.params.save_fstat_and_zscore_mats:
            joblib.dump(recalls, self.get_path_to_resource_in_workspace(subject + '-recalls.pkl'))

        n_perms = self.params.n_perms
        shuffle_mat = np.empty(shape=(n_perms+1,n_bp_pairs*n_freqs*n_bins), dtype=np.float)
        compute_f_stat(phase_diff_mat, recalls, shuffle_mat[0])
        for i in xrange(1,n_perms+1):
            print 'Permutation', i
            shuffle(recalls)
            compute_f_stat(phase_diff_mat, recalls, shuffle_mat[i])

        if self.params.save_fstat_and_zscore_mats:
            joblib.dump(shuffle_mat.reshape((n_perms+1,n_bp_pairs,n_freqs,n_bins)), self.get_path_to_resource_in_workspace(subject + '-fstat_mat.pkl'))

        shuffle_mat = shuffle_mat.reshape(-1)
        compute_zscores(shuffle_mat, n_perms+1)

        shuffle_mat = shuffle_mat.reshape((n_perms+1,n_bp_pairs,n_freqs,n_bins))

        if self.params.save_fstat_and_zscore_mats:
            joblib.dump(shuffle_mat, self.get_path_to_resource_in_workspace(subject + '-zscore_mat.pkl'))

        shuffle_mat = shuffle_mat.sum(axis=(2,3))

        n_bps = len(bipolar_pairs)
        shuffle_mat_bp = np.zeros(shape=(n_perms+1,n_bps), dtype=np.float)
        for i,bp_pair in enumerate(bipolar_pair_pairs):
            bp1, bp2 = bp_pair
            shuffle_mat_bp[:,bp1] += shuffle_mat[:,i]
            shuffle_mat_bp[:,bp2] += shuffle_mat[:,i]

        self.connectivity_strength = -1.0 * zmap(shuffle_mat_bp[0,:], shuffle_mat_bp[1:,:], axis=0, ddof=1)

        self.pass_object('connectivity_strength', self.connectivity_strength)

        joblib.dump(self.connectivity_strength, self.get_path_to_resource_in_workspace(subject + '-connectivity_strength.pkl'))
