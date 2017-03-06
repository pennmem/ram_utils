from RamPipeline import *

from math import sqrt
import numpy as np
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression
from ReportTasks.RamTaskMethods import run_lolo_xval,run_loso_xval,permuted_lolo_AUCs,permuted_loso_AUCs,ModelOutput
from sklearn.externals import joblib
import warnings
from ptsa.data.readers.IndexReader import JsonIndexReader
from ReportUtils import ReportRamTask

import hashlib


def normalize_sessions(pow_mat, events):
    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)
    return pow_mat



class ComputeClassifier(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeClassifier,self).__init__(mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.lr_classifier = None
        self.xval_output = dict()   # ModelOutput per session; xval_output[-1] is across all sessions
        self.perm_AUCs = None
        self.pvalue = None

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


    def xval_test_type(self, events):
        event_sessions = events.session
        sessions = np.unique(event_sessions)
        if len(sessions) == 1:
            return 'lolo'
        for sess in sessions:
            sess_events = events[event_sessions == sess]
            if len(sess_events) >= 0.7 * len(events):
                return 'lolo'
        return 'loso'

    def run(self):
        subject = self.pipeline.subject
        task=self.pipeline.task

        events = self.get_passed_object(task+'_events')
        self.pow_mat = normalize_sessions(self.get_passed_object('pow_mat'), events)

        #n1 = np.sum(events.recalled)
        #n0 = len(events) - n1
        #w0 = (2.0/n0) / ((1.0/n0)+(1.0/n1))
        #w1 = (2.0/n1) / ((1.0/n0)+(1.0/n1))
        self.lr_classifier = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='auto', solver='liblinear')

        event_sessions = events.session
        recalls = events.correct

        if self.xval_test_type(events) == 'loso':
            print 'Performing permutation test'
            self.perm_AUCs = permuted_loso_AUCs(self,event_sessions, recalls)

            print 'Performing leave-one-session-out xval'
            run_loso_xval(event_sessions, recalls,
                                        self.pow_mat, self.lr_classifier,self.xval_output)
        else:
            print 'Performing in-session permutation test'
            self.perm_AUCs = permuted_lolo_AUCs(self,events)

            print 'Performing leave-one-list-out xval'
            run_lolo_xval(events, recalls, self.pow_mat,self.lr_classifier,self.xval_output, permuted=False)

        print 'AUC =', self.xval_output[-1].auc

        self.pvalue = np.sum(self.perm_AUCs >= self.xval_output[-1].auc) / float(self.perm_AUCs.size)
        print 'Perm test p-value =', self.pvalue

        print 'thresh =', self.xval_output[-1].jstat_thresh, 'quantile =', self.xval_output[-1].jstat_quantile

        # Finally, fitting classifier on all available data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lr_classifier.fit(self.pow_mat, recalls)

        self.pass_object('lr_classifier', self.lr_classifier)
        self.pass_object('xval_output', self.xval_output)
        self.pass_object('perm_AUCs', self.perm_AUCs)
        self.pass_object('pvalue', self.pvalue)

        joblib.dump(self.lr_classifier, self.get_path_to_resource_in_workspace(subject + '-lr_classifier.pkl'))
        joblib.dump(self.xval_output, self.get_path_to_resource_in_workspace(subject + '-xval_output.pkl'))
        joblib.dump(self.perm_AUCs, self.get_path_to_resource_in_workspace(subject + '-perm_AUCs.pkl'))
        joblib.dump(self.pvalue, self.get_path_to_resource_in_workspace(subject + '-pvalue.pkl'))

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        for attr in ['lr_classifier','xval_output','perm_AUCs','pvalue']:
            try:
                self.__setattr__(attr,joblib.load(self.get_path_to_resource_in_workspace(subject + '-%s.pkl'%attr)))
            except IOError:
                self.__setattr__(attr,joblib.load(self.get_path_to_resource_in_workspace(subject + '-'+task+'-%s.pkl'%attr)))

        self.pass_object('lr_classifier', self.lr_classifier)
        self.pass_object('xval_output', self.xval_output)
        self.pass_object('perm_AUCs', self.perm_AUCs)
        self.pass_object('pvalue', self.pvalue)



