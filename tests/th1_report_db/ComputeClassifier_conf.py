from RamPipeline import *

import numpy as np
from copy import deepcopy
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.cross_validation import StratifiedKFold
from random import shuffle
from sklearn.externals import joblib
from ReportUtils import ReportRamTask

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

    def compute_roc(self):
        try:
            self.auc = roc_auc_score(self.true_labels, self.probs)
        except ValueError:
            return
        self.fpr, self.tpr, self.thresholds = roc_curve(self.true_labels, self.probs)
        self.jstat_quantile = 0.5
        self.jstat_thresh = np.median(self.probs)
        # idx = np.argmax(self.tpr-self.fpr)
        # self.jstat_thresh = self.thresholds[idx]
        # self.jstat_quantile = np.sum(self.probs <= self.jstat_thresh) / float(self.probs.size)

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


class ComputeClassifier_conf(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeClassifier_conf,self).__init__(mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.lr_classifier_conf = None
        self.xval_output_conf = dict()   # ModelOutput per session; xval_output_conf[-1] is across all sessions
        self.perm_AUCs_conf = None
        self.pvalue_conf = None

    def initialize(self):
        task = self.pipeline.task
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name=task+'_events',
                                        access_path = ['experiments','th1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar_json',
                                        access_path = ['electrodes','bipolar_json'])

    def run_loso_xval(self, event_sessions, recalls, permuted=False):
        probs = np.empty_like(recalls, dtype=np.float)

        sessions = np.unique(event_sessions)

        for sess in sessions:
            insample_mask = (event_sessions != sess)
            insample_pow_mat = self.pow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]

            self.lr_classifier_conf.fit(insample_pow_mat, insample_recalls)

            outsample_mask = ~insample_mask
            outsample_pow_mat = self.pow_mat[outsample_mask]
            outsample_recalls = recalls[outsample_mask]

            outsample_probs = self.lr_classifier_conf.predict_proba(outsample_pow_mat)[:,1]
            if not permuted:
                self.xval_output_conf[sess] = ModelOutput(outsample_recalls, outsample_probs)
                self.xval_output_conf[sess].compute_roc()
                self.xval_output_conf[sess].compute_tercile_stats()
            probs[outsample_mask] = outsample_probs

        if not permuted:
            self.xval_output_conf[-1] = ModelOutput(recalls, probs)
            self.xval_output_conf[-1].compute_roc()
            self.xval_output_conf[-1].compute_tercile_stats()

        return probs

    def permuted_loso_AUCs(self, event_sessions, recalls):
        n_perm = self.params.n_perm
        permuted_recalls = np.array(recalls)
        AUCs = np.empty(shape=n_perm, dtype=np.float)
        for i in xrange(n_perm):
            for sess in event_sessions:
                sel = (event_sessions == sess)
                sess_permuted_recalls = permuted_recalls[sel]
                shuffle(sess_permuted_recalls)
                permuted_recalls[sel] = sess_permuted_recalls
            probs = self.run_loso_xval(event_sessions, permuted_recalls, permuted=True)
            AUCs[i] = roc_auc_score(recalls, probs)
            print 'AUC =', AUCs[i]
        return AUCs

    def run_lolo_xval(self, sess, event_lists, recalls, permuted=False):
        probs = np.empty_like(recalls, dtype=np.float)

        lists = np.unique(event_lists)

        for lst in lists:
            insample_mask = (event_lists != lst)
            insample_pow_mat = self.pow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]

            self.lr_classifier_conf.fit(insample_pow_mat, insample_recalls)

            outsample_mask = ~insample_mask
            outsample_pow_mat = self.pow_mat[outsample_mask]

            probs[outsample_mask] = self.lr_classifier_conf.predict_proba(outsample_pow_mat)[:,1]

        if not permuted:
            xval_output_conf = ModelOutput(recalls, probs)
            xval_output_conf.compute_roc()
            xval_output_conf.compute_tercile_stats()
            self.xval_output_conf[sess] = self.xval_output_conf[-1] = xval_output_conf

        return probs

    def permuted_lolo_AUCs(self, sess, event_lists, recalls):
        n_perm = self.params.n_perm
        permuted_recalls = np.array(recalls)
        AUCs = np.empty(shape=n_perm, dtype=np.float)
        for i in xrange(n_perm):
            shuffle(permuted_recalls)
            # for lst in event_lists:
                # sel = (event_lists == lst)
                # list_permuted_recalls = permuted_recalls[sel]
                # shuffle(list_permuted_recalls)
                # permuted_recalls[sel] = list_permuted_recalls
            probs = self.run_lolo_xval(sess, event_lists, permuted_recalls, permuted=True)
            AUCs[i] = roc_auc_score(recalls, probs)
            print 'AUC = ', AUCs[i]
        return AUCs

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task + '_events')
        self.pow_mat = normalize_sessions(self.get_passed_object('classify_pow_mat'), events)


        # self.lr_classifier_conf = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='auto', solver='liblinear')
        self.lr_classifier_conf = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='auto', solver='liblinear')
        # self.lr_classifier_conf = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='balanced',solver='liblinear',fit_intercept=False)

        event_sessions = events.session    
        recalls = events.confidence == 2

        # Don't run confidence decoding if there are too few examples in each class
        if (np.sum(recalls==False) < 5) or (np.sum(recalls==True) < 5):
            self.conf_decode_success = False
            self.pass_object('conf_decode_success', self.conf_decode_success)
            joblib.dump(self.pvalue_conf, self.get_path_to_resource_in_workspace(subject + '-' + task + '-conf_decode_success.pkl')) 
        else:       

        
            sessions = np.unique(event_sessions)
            if len(sessions) > 1:
                print 'Performing permutation test'
                self.perm_AUCs_conf = self.permuted_loso_AUCs(event_sessions, recalls)

                print 'Performing leave-one-session-out xval'
                self.run_loso_xval(event_sessions, recalls, permuted=False)
            else:
                sess = sessions[0]
                event_lists = deepcopy(events.trial)
                skf = StratifiedKFold(recalls, n_folds=8,shuffle=True)
                # rand_order = np.random.permutation(len(events))
                # event_lists = event_lists[rand_order]
                if self. params.doStratKFold:
                    skf = StratifiedKFold(recalls, n_folds=self.params.n_folds,shuffle=True)
                    for i, (train_index, test_index) in enumerate(skf):
                        event_lists[test_index] = i
            
                print 'Performing in-session permutation test'
                self.perm_AUCs_conf = self.permuted_lolo_AUCs(sess, event_lists, recalls)

                if self. params.doStratKFold:
                    print 'Performing %d-fold stratified xval'%(self.n_folds)
                else:
                    print 'Performing leave-one-list-out xval'
                self.run_lolo_xval(sess, event_lists, recalls, permuted=False)

            print 'AUC =', self.xval_output_conf[-1].auc

            self.pvalue_conf = np.sum(self.perm_AUCs_conf >= self.xval_output_conf[-1].auc) / float(self.perm_AUCs_conf.size)
            print 'Perm test p-value =', self.pvalue_conf, ' mean null = ', np.mean(self.perm_AUCs_conf)

            print 'thresh =', self.xval_output_conf[-1].jstat_thresh, 'quantile =', self.xval_output_conf[-1].jstat_quantile

            # Finally, fitting classifier on all available data
            self.lr_classifier_conf.fit(self.pow_mat, recalls)
            self.conf_decode_success = True

            self.pass_object('lr_classifier_conf', self.lr_classifier_conf)
            self.pass_object('xval_output_conf', self.xval_output_conf)
            self.pass_object('perm_AUCs_conf', self.perm_AUCs_conf)
            self.pass_object('pvalue_conf', self.pvalue_conf)
            self.pass_object('conf_decode_success', self.conf_decode_success)

            joblib.dump(self.lr_classifier_conf, self.get_path_to_resource_in_workspace(subject + '-' + task + '-lr_classifier_conf.pkl'))
            joblib.dump(self.xval_output_conf, self.get_path_to_resource_in_workspace(subject + '-' + task + '-xval_output_conf.pkl'))
            joblib.dump(self.perm_AUCs_conf, self.get_path_to_resource_in_workspace(subject + '-' + task + '-perm_AUCs_conf.pkl'))
            joblib.dump(self.pvalue_conf, self.get_path_to_resource_in_workspace(subject + '-' + task + '-pvalue_conf.pkl'))
            joblib.dump(self.conf_decode_success, self.get_path_to_resource_in_workspace(subject + '-' + task + '-conf_decode_success.pkl'))        

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.lr_classifier_conf = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-lr_classifier_conf.pkl'))
        self.xval_output_conf = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-xval_output_conf.pkl'))
        self.perm_AUCs_conf = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-perm_AUCs_conf.pkl'))
        self.pvalue_conf = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-pvalue_conf.pkl'))
        self.conf_decode_success = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-conf_decode_success.pkl'))        

        self.pass_object('lr_classifier_conf', self.lr_classifier_conf)
        self.pass_object('xval_output_conf', self.xval_output_conf)
        self.pass_object('perm_AUCs_conf', self.perm_AUCs_conf)
        self.pass_object('pvalue_conf', self.pvalue_conf)
        self.pass_object('conf_decode_success', self.conf_decode_success)        
