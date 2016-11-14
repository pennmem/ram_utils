from RamPipeline import *

from math import log
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.externals import joblib
from ReportUtils import ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader

import hashlib


def prob2perf_norm(xval_output, p):
    fi1 = fi0 = 1.0

    if p < 1e-6:
        return 0.0
    elif p < 1.0 - 1e-6:
        p_norm = log(p/(1.0-p))
        fi1 = norm.cdf(p_norm, loc=xval_output.mean1, scale=xval_output.pooled_std)
        fi0 = norm.cdf(p_norm, loc=xval_output.mean0, scale=xval_output.pooled_std)

    r = xval_output.n1*fi1 / (xval_output.n1*fi1 + xval_output.n0*fi0)
    return r


class ComputeControlTable(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeControlTable,self).__init__(mark_as_completed)
        self.params = params
        self.control_table = None

    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            hash_md5.update(open(fname,'rb').read())

        fr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='FR1')))
        for fname in fr1_event_files:
            hash_md5.update(open(fname,'rb').read())

        catfr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='catFR1')))
        for fname in catfr1_event_files:
            hash_md5.update(open(fname,'rb').read())

        event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in event_files:
            hash_md5.update(open(fname,'rb').read())

        return hash_md5.digest()

    def restore(self):
        subject = self.pipeline.subject
        self.control_table = pd.read_pickle(self.get_path_to_resource_in_workspace(subject + '-control_table.pkl'))
        self.pass_object('control_table', self.control_table)

    def run(self):
        subject = self.pipeline.subject

        events = self.get_passed_object('control_events')

        if len(events) == 0:
            self.control_table = pd.DataFrame(columns=['session','stimAnodeTag','stimCathodeTag','prob_pre','prob_diff','perf_diff','subject'])
            self.pass_object('control_table', self.control_table)
            self.control_table.to_pickle(self.get_path_to_resource_in_workspace(subject + '-control_table.pkl'))
            return

        event_sess = np.tile(events.session, 2)
        stimAnodeTag = np.tile(events.anode_label, 2)
        stimCathodeTag = np.tile(events.cathode_label, 2)

        lr_classifier = self.get_passed_object('lr_classifier')
        xval_output = self.get_passed_object('xval_output')

        pow_mat_pre = self.get_passed_object('control_pow_mat_pre')
        prob_pre = lr_classifier.predict_proba(pow_mat_pre)[:,1]

        pow_mat_post = self.get_passed_object('control_pow_mat_post')
        prob_post = lr_classifier.predict_proba(pow_mat_post)[:,1]

        prob_diff = prob_post - prob_pre

        probs = xval_output[-1].probs
        true_labels = xval_output[-1].true_labels
        performance_map = sorted(zip(probs,true_labels))
        probs, true_labels = zip(*performance_map)
        true_labels = np.array(true_labels)
        total_recall_performance = np.sum(true_labels) / float(len(true_labels))

        # the code below is not pythonic, but I'll figure it out later
        n_events = len(event_sess)
        perf_diff = np.zeros(n_events, dtype=float)
        for i in xrange(n_events):
            perf_pre = prob2perf_norm(xval_output[-1], prob_pre[i])
            perf_diff[i] = 100.0*(prob2perf_norm(xval_output[-1], prob_pre[i]+prob_diff[i]) - perf_pre) / total_recall_performance

        self.control_table = pd.DataFrame()
        self.control_table['session'] = event_sess
        self.control_table['stimAnodeTag'] = stimAnodeTag
        self.control_table['stimCathodeTag'] = stimCathodeTag
        self.control_table['prob_pre'] = prob_pre
        self.control_table['prob_diff'] = prob_diff
        self.control_table['perf_diff'] = perf_diff
        self.control_table['subject'] = subject

        self.pass_object('control_table', self.control_table)
        self.control_table.to_pickle(self.get_path_to_resource_in_workspace(subject + '-control_table.pkl'))
