from RamPipeline import *

import numpy as np
import pandas as pd
from sklearn.externals import joblib


class ComputeControlTable(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.control_table = None

    def restore(self):
        subject = self.pipeline.subject
        self.control_table = pd.read_pickle(self.get_path_to_resource_in_workspace(subject + '-control_table.pkl'))
        self.pass_object('control_table', self.control_table)

    def run(self):
        subject = self.pipeline.subject

        lr_classifier = self.get_passed_object('lr_classifier')
        xval_output = self.get_passed_object('xval_output')
        thresh = xval_output[-1].jstat_thresh

        control_pow_mat_pre = self.get_passed_object('control_pow_mat_pre')
        control_prob_pre = lr_classifier.predict_proba(control_pow_mat_pre)[:,1]
        low_sel = (control_prob_pre < thresh)
        control_prob_pre = control_prob_pre[low_sel]

        control_pow_mat_045 = self.get_passed_object('control_pow_mat_045')
        control_prob_045 = lr_classifier.predict_proba(control_pow_mat_045)[:,1]
        control_prob_045 = control_prob_045[low_sel]

        control_pow_mat_07 = self.get_passed_object('control_pow_mat_07')
        control_prob_07 = lr_classifier.predict_proba(control_pow_mat_07)[:,1]
        control_prob_07 = control_prob_07[low_sel]

        control_pow_mat_12 = self.get_passed_object('control_pow_mat_12')
        control_prob_12 = lr_classifier.predict_proba(control_pow_mat_12)[:,1]
        control_prob_12 = control_prob_12[low_sel]

        self.control_table = pd.DataFrame()
        self.control_table['prob_pre'] = control_prob_pre
        self.control_table['prob_diff_250'] = control_prob_045 - control_prob_pre
        self.control_table['prob_diff_500'] = control_prob_07 - control_prob_pre
        self.control_table['prob_diff_1000'] = control_prob_12 - control_prob_pre

        self.pass_object('control_table', self.control_table)
        self.control_table.to_pickle(self.get_path_to_resource_in_workspace(subject + '-control_table.pkl'))
