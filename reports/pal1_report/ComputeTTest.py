from RamPipeline import *

import numpy as np
from scipy.stats import ttest_ind
from sklearn.externals import joblib
from ptsa.data.readers.IndexReader import JsonIndexReader
import hashlib

from ReportUtils import ReportRamTask

class ComputeTTest(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeTTest,self).__init__(mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])
        task = self.pipeline.task

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        fr1_event_files = sorted(
            list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in fr1_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        pow_mat = self.get_passed_object('hf_pow_mat')

        #freq_sel = np.tile((self.params.freqs>=self.params.ttest_frange[0]) & (self.params.freqs<=self.params.ttest_frange[1]), pow_mat.shape[1] / self.params.freqs.size)
        #pow_mat = pow_mat[:,freq_sel]

        events = self.get_passed_object(self.pipeline.task+'_events')
        sessions = np.unique(events.session)

        # norm_func = normalize.standardize_pow_mat if self.params.norm_method=='zscore' else normalize.normalize_pow_mat
        # pow_mat = norm_func(pow_mat, events, sessions)[0]

        self.ttest = {}
        for sess in sessions:
            sel = (events.session==sess)
            sess_events = events[sel]

            sess_pow_mat = pow_mat[sel,:]

            sess_recalls = np.array(sess_events.correct, dtype=np.bool)

            recalled_sess_pow_mat = sess_pow_mat[sess_recalls,:]
            nonrecalled_sess_pow_mat = sess_pow_mat[~sess_recalls,:]

            t,p = ttest_ind(recalled_sess_pow_mat, nonrecalled_sess_pow_mat, axis=0)
            self.ttest[sess] = (t,p)

        recalls = np.array(events.correct, dtype=np.bool)

        recalled_pow_mat = pow_mat[recalls,:]
        nonrecalled_pow_mat = pow_mat[~recalls,:]

        t,p = ttest_ind(recalled_pow_mat, nonrecalled_pow_mat, axis=0)
        self.ttest[-1] = (t,p)

        self.pass_object('ttest', self.ttest)
        joblib.dump(self.ttest, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ttest.pkl'))

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        self.ttest = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-ttest.pkl'))
        self.pass_object('ttest', self.ttest)
