import os
from math import log
import numpy as np
import pandas as pd
from scipy.stats import norm
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


class ComputePSTable(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputePSTable,self).__init__(mark_as_completed)
        self.params = params
        self.ps_table = None

    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            hash_md5.update(open(fname, 'rb').read())

        fr1_event_files = sorted(list(
            json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='TH1')))
        for fname in fr1_event_files:
            hash_md5.update(open(fname, 'rb').read())

        event_files = sorted(list(
            json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in event_files:
            hash_md5.update(open(fname, 'rb').read())

        return hash_md5.digest()

    def restore(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.experiment
        self.ps_table = pd.read_pickle(self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_table.pkl'))
        self.pass_object('ps_table', self.ps_table)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        ps_events = self.get_passed_object(task + '_events')
        bp_tal_structs = self.get_passed_object('bp_tal_structs')
        bp_tal_stim_only_structs = self.get_passed_object('bp_tal_stim_only_structs')

        lr_classifier = self.get_passed_object('lr_classifier')
        xval_output = self.get_passed_object('xval_output')

        ps_pow_mat_pre = self.get_passed_object('ps_pow_mat_pre')
        ps_pow_mat_post = self.get_passed_object('ps_pow_mat_post')

        n_events = len(ps_events)

        prob_pre, prob_diff = self.compute_prob_deltas(ps_pow_mat_pre, ps_pow_mat_post, lr_classifier)

        probs = xval_output[-1].probs
        true_labels = xval_output[-1].true_labels
        performance_map = sorted(zip(probs, true_labels))
        probs, true_labels = zip(*performance_map)
        true_labels = np.array(true_labels)
        total_recall_performance = np.sum(true_labels) / float(len(true_labels))

        # the code below is not pythonic, but I'll figure it out later
        perf_diff = np.zeros(n_events, dtype=float)
        for i in xrange(n_events):
            # perf_pre = prob2perf(probs, true_labels, prob_pre[i]+1e-7)
            # perf_diff[i] = 100.0*(prob2perf(probs, true_labels, prob_pre[i]+prob_diff[i]+1e-7) - perf_pre) / total_recall_performance
            perf_pre = prob2perf_norm(xval_output[-1], prob_pre[i])
            perf_diff[i] = 100.0 * (
            prob2perf_norm(xval_output[-1], prob_pre[i] + prob_diff[i]) - perf_pre
            ) / total_recall_performance

        region = [None] * n_events
        for i, ev in enumerate(ps_events):
            anode_tag = ev.anode_label.upper()
            cathode_tag = ev.cathode_label.upper()
            bp_label1 = anode_tag + '-' + cathode_tag
            bp_label2 = cathode_tag + '-' + anode_tag
            if bp_label1 in bp_tal_structs.index:
                region[i] = bp_tal_structs['bp_atlas_loc'].ix[bp_label1]
            elif bp_label2 in bp_tal_structs.index:
                region[i] = bp_tal_structs['bp_atlas_loc'].ix[bp_label2]
            elif bp_label1 in bp_tal_stim_only_structs.index:
                region[i] = bp_tal_stim_only_structs[bp_label1]
            elif bp_label2 in bp_tal_stim_only_structs.index:
                region[i] = bp_tal_stim_only_structs[bp_label2]

        self.ps_table = pd.DataFrame()
        self.ps_table['session'] = ps_events.session
        self.ps_table['mstime'] = ps_events.mstime
        self.ps_table['Pulse_Frequency'] = ps_events.pulse_freq
        self.ps_table['Amplitude'] = ps_events.amplitude / 1000.0
        self.ps_table['Duration'] = ps_events.stim_duration
        self.ps_table['Burst_Frequency'] = ps_events.burst_freq
        self.ps_table['stimAnodeTag'] = ps_events.anode_label
        self.ps_table['stimCathodeTag'] = ps_events.cathode_label
        self.ps_table['Region'] = region
        self.ps_table['prob_pre'] = prob_pre
        self.ps_table['prob_diff'] = prob_diff
        self.ps_table['perf_diff'] = perf_diff
        self.ps_table['isi'] = ps_events.isi

        self.pass_object('ps_table', self.ps_table)
        self.ps_table.to_pickle(self.get_path_to_resource_in_workspace(subject + '-' + task + '-ps_table.pkl'))

    def compute_prob_deltas(self, ps_pow_mat_pre, ps_pow_mat_post, lr_classifier):
        prob_pre = lr_classifier.predict_proba(ps_pow_mat_pre)[:, 1]
        prob_post = lr_classifier.predict_proba(ps_pow_mat_post)[:, 1]
        return prob_pre, prob_post - prob_pre
