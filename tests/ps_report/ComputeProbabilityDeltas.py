__author__ = 'm'


from RamPipeline import *
import numpy as np

import os
import os.path
import re
import numpy as np
from scipy.io import loadmat
from scipy.stats.mstats import zscore

from ptsa.data.events import Events
from ptsa.data.rawbinwrapper import RawBinWrapper

from get_bipolar_subj_elecs import get_bipolar_subj_elecs

from ptsa.wavelet import phase_pow_multi

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib


class ComputeProbabilityDeltas(RamTask):
    def __init__(self, params, task, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.task = task
        self.params = params

    def run(self):
        experiment = self.pipeline.experiment
        subject = self.pipeline.subject_id

        lr_classifier = self.get_passed_object('lr_classifier')
        ps_pow_mat_pre = self.get_passed_object('ps_pow_mat_pre')
        ps_pow_mat_post = self.get_passed_object('ps_pow_mat_post')

        prob_pre, prob_diff = self.compute_prob_deltas(ps_pow_mat_pre, ps_pow_mat_post, lr_classifier)

        joblib.dump(prob_pre, subject+'_prob_pre.pkl')
        joblib.dump(prob_diff, subject+'_prob_diff.pkl')


    def compute_prob_deltas(self, ps_pow_mat_pre, ps_pow_mat_post, lr_classifier):
        prob_pre = lr_classifier.predict_proba(ps_pow_mat_pre)[:,1]
        prob_post = lr_classifier.predict_proba(ps_pow_mat_post)[:,1]
        return prob_pre, prob_post - prob_pre
