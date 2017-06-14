from ram_utils.RamPipeline import *

from math import sqrt
import numpy as np
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.externals import joblib
from ...ReportTasks.RamTaskMethods import run_lolo_xval,run_loso_xval,permuted_loso_AUCs,permuted_lolo_AUCs,ModelOutput
import warnings
from ptsa.data.readers.IndexReader import JsonIndexReader
from ...ReportUtils import ReportRamTask

import hashlib


def normalize_sessions(pow_mat, events):
    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)
    return pow_mat


class ModelOutput(object):
    def __init__(self, true_labels, probs):
        self.true_labels = np.array(true_labels)
        self.probs = np.array(probs)
        self.auc = np.nan
        self.fpr = np.nan
        self.tpr = np.nan
        self.thresholds = np.nan
        self.jstat_thresh = np.nan
        self.jstat_quantile = np.nan
        self.low_pc_diff_from_mean = np.nan
        self.mid_pc_diff_from_mean = np.nan
        self.high_pc_diff_from_mean = np.nan
        self.n1 = np.nan
        self.mean1 = np.nan
        #self.std1 = np.nan
        self.n0 = np.nan
        self.mean0 = np.nan
        #self.std0 = np.nan
        self.pooled_std = np.nan

    def compute_normal_approx(self):
        class1_mask = (self.true_labels==1)
        class1_probs = self.probs[class1_mask]
        self.n1 = len(class1_probs)
        class1_normal = np.log(class1_probs/(1.0-class1_probs))
        self.mean1 = np.mean(class1_normal)
        #self.std1 = np.std(class1_normal, ddof=1)
        var1 = np.var(class1_normal, ddof=1)
        print 'Positive class: mean =', self.mean1, 'variance =', var1, 'n =', self.n1

        class0_probs = self.probs[~class1_mask]
        self.n0 = len(class0_probs)
        class0_normal = np.log(class0_probs/(1.0-class0_probs))
        self.mean0 = np.mean(class0_normal)
        #self.std0 = np.std(class0_normal, ddof=1)
        var0 = np.var(class0_normal, ddof=1)
        print 'Negative class: mean =', self.mean0, 'variance =', var0, 'n =', self.n0

        self.pooled_std = sqrt((var1*(self.n1-1) + var0*(self.n0-1)) / (self.n1+self.n0-2))

        #if self.mean1 < self.mean0:
        #    print 'BAD CLASSIFIER: recall class mean is less than non-recall class mean!!'
        #    sys.exit(0)

    def compute_roc(self):
        try:
            self.auc = roc_auc_score(self.true_labels, self.probs)
        except ValueError:
            return
        self.fpr, self.tpr, self.thresholds = roc_curve(self.true_labels, self.probs)
        self.jstat_quantile = 0.5
        self.jstat_thresh = np.median(self.probs)

    def compute_tercile_stats(self):
        thresh_low = np.percentile(self.probs, 100.0/3.0)
        thresh_high = np.percentile(self.probs, 2.0*100.0/3.0)

        low_terc_sel = (self.probs <= thresh_low)
        high_terc_sel = (self.probs >= thresh_high)
        mid_terc_sel = ~(low_terc_sel | high_terc_sel)

        low_terc_recall_rate = np.sum(self.true_labels[low_terc_sel]) / float(np.sum(low_terc_sel))
        mid_terc_recall_rate = np.sum(self.true_labels[mid_terc_sel]) / float(np.sum(mid_terc_sel))
        high_terc_recall_rate = np.sum(self.true_labels[high_terc_sel]) / float(np.sum(high_terc_sel))

        recall_rate = np.sum(self.true_labels) / float(self.true_labels.size)

        self.low_pc_diff_from_mean = 100.0 * (low_terc_recall_rate-recall_rate) / recall_rate
        self.mid_pc_diff_from_mean = 100.0 * (mid_terc_recall_rate-recall_rate) / recall_rate
        self.high_pc_diff_from_mean = 100.0 * (high_terc_recall_rate-recall_rate) / recall_rate


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

        for experiment in ['FR1','FR3','catFR1','catFR3']:
            event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=experiment)))
            for fname in event_files:
                with open(fname,'rb') as f:
                    hash_md5.update(f.read())

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

        events = self.get_passed_object('THR_events')
        self.pow_mat = normalize_sessions(self.get_passed_object('pow_mat'), events)

        #n1 = np.sum(events.recalled)
        #n0 = len(events) - n1
        #w0 = (2.0/n0) / ((1.0/n0)+(1.0/n1))
        #w1 = (2.0/n1) / ((1.0/n0)+(1.0/n1))
        self.lr_classifier = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='auto', solver='liblinear')

        event_sessions = events.session
        recalls = events.recalled
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

        print 'AUC = ',self.xval_output[-1].auc

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
        self.pass_object('classifier_path',self.get_path_to_resource_in_workspace(subject+'-lr_classifier.pkl'))

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

        self.pass_object('classifier_path',self.get_path_to_resource_in_workspace(subject+'-lr_classifier.pkl'))
        self.pass_object('lr_classifier', self.lr_classifier)
        self.pass_object('xval_output', self.xval_output)
        self.pass_object('perm_AUCs', self.perm_AUCs)
        self.pass_object('pvalue', self.pvalue)


