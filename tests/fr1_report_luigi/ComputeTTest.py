from RamPipeline import *

import numpy as np
from scipy.stats import ttest_ind
from sklearn.externals import joblib

from ReportUtils import ReportRamTask

import luigi
import numpy as np
import os
import os.path
import numpy as np
from sklearn.externals import joblib

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import hashlib
from ReportTasks.RamTaskMethods import create_baseline_events

from RamTaskL import RamTaskL
from FR1EventPreparation import FR1EventPreparation
from MontagePreparation import MontagePreparation
from ComputeFR1Powers import ComputeFR1Powers
from ComputeFR1HFPowers import ComputeFR1HFPowers

class ComputeTTest(RamTaskL):
    def requires(self):
        yield ComputeFR1HFPowers(pipeline=self.pipeline)


    def define_outputs(self):

        self.add_file_resource('ttest')


    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        pow_mat = self.get_passed_object('hf_pow_mat')
        print 'pow_mat.shape:',pow_mat.shape

        try:
            events=self.get_passed_object('hf_events')
        except KeyError:
            # events = self.get_passed_object(self.pipeline.task+'_events')
            events = self.get_passed_object(self.pipeline.task+'_events_compute_powers')

        print 'len(events):',len(events)
        sessions = np.unique(events.session)

        self.ttest = {}
        for sess in sessions:
            sel = (events.session==sess)
            sess_events = events[sel]

            sess_pow_mat = pow_mat[sel,:]

            sess_recalls = np.array(sess_events.recalled, dtype=np.bool)

            recalled_sess_pow_mat = sess_pow_mat[sess_recalls,:]
            nonrecalled_sess_pow_mat = sess_pow_mat[~sess_recalls,:]

            t,p = ttest_ind(recalled_sess_pow_mat, nonrecalled_sess_pow_mat, axis=0)
            self.ttest[sess] = (t,p)

        recalls = np.array(events.recalled, dtype=np.bool)

        recalled_pow_mat = pow_mat[recalls,:]
        nonrecalled_pow_mat = pow_mat[~recalls,:]

        t,p = ttest_ind(recalled_pow_mat, nonrecalled_pow_mat, axis=0)
        self.ttest[-1] = (t,p)

        self.pass_object('ttest', self.ttest)
        # joblib.dump(self.ttest, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ttest.pkl'))

    # def restore(self):
    #     subject = self.pipeline.subject
    #     task = self.pipeline.task
    #     self.ttest = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-ttest.pkl'))
    #     self.pass_object('ttest', self.ttest)
