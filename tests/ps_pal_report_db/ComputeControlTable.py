from RamPipeline import *

from math import log
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.externals import joblib
from ReportUtils import ReportRamTask


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

    def initialize(self):
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name='fr1_events',
                                        access_path = ['experiments','fr1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='catfr1_events',
                                        access_path = ['experiments','catfr1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='ps_events',
                                        access_path = ['experiments','ps','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar_json',
                                        access_path = ['electrodes','bipolar_json'])

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
        stimAnodeTag = np.tile(events.stimAnodeTag, 2)
        stimCathodeTag = np.tile(events.stimCathodeTag, 2)

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
