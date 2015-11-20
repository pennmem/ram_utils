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


class ComputeClassifier(RamTask):
    def __init__(self, params, task, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.task = task
        self.params = params

    def run(self):

        pow_mat = self.get_passed_object('pow_mat')
        recalls = self.get_passed_object('recalls')




        lr_classifier = self.compute_classifier(pow_mat, recalls)

        joblib.dump(lr_classifier, self.get_path_to_resource_in_workspace(self.pipeline.subject_id+'_lr.pkl'))

    def compute_classifier(self, pow_mat, recalls):
        print 'Computing logistic regression:', pow_mat.shape[0], 'samples', pow_mat.shape[1], 'features'

        lr_classifier = LogisticRegressionCV(penalty='l1', solver='liblinear')
        lr_classifier.fit(pow_mat, recalls)
        probs = lr_classifier.predict_proba(pow_mat)[:,1]
        auc = roc_auc_score(recalls, probs)

        print 'AUC =', auc

        return lr_classifier


