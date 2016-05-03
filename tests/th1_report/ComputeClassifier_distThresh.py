from RamPipeline import *

import numpy as np
from copy import deepcopy
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.cross_validation import StratifiedKFold
from random import shuffle
from normalize import standardize_pow_mat
from sklearn.externals import joblib
from ReportUtils import ReportRamTask

def normalize_sessions(pow_mat, events):
    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)
    return pow_mat

class ModelOutput(object):
    def __init__(self):
        self.aucs_by_thresh = np.nan
        self.pval_by_thresh = np.nan
        self.pCorr_by_thresh = np.nan        
        self.thresholds = np.nan

class ComputeClassifier_distThresh(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeClassifier_distThresh,self).__init__(mark_as_completed)
        self.params = params
        self.pow_mat = None
               
    def initialize(self):
        task = self.pipeline.task
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name=task+'_events',
                                        access_path = ['experiments','th1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                    access_path = ['electrodes','bipolar'])

    def run_loso_xval(self, event_sessions, recalls, permuted=False):
        probs = np.empty_like(recalls, dtype=np.float)
        sessions = np.unique(event_sessions)

        for sess in sessions:
            insample_mask = (event_sessions != sess)
            insample_pow_mat = self.pow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]

            self.lr_classifier.fit(insample_pow_mat, insample_recalls)

            outsample_mask = ~insample_mask
            outsample_pow_mat = self.pow_mat[outsample_mask]
            outsample_recalls = recalls[outsample_mask]

            outsample_probs = self.lr_classifier.predict_proba(outsample_pow_mat)[:,1]
            probs[outsample_mask] = outsample_probs
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
        return AUCs

    def run_lolo_xval(self, sess, event_lists, recalls, permuted=False):
        probs = np.empty_like(recalls, dtype=np.float)

        lists = np.unique(event_lists)

        for lst in lists:
            
            # zpow_mat = standardize_pow_mat(self.pow_mat,self.events,[sess],lst)[0]
            insample_mask = (event_lists != lst)
            insample_pow_mat = self.pow_mat[insample_mask]

            # insample_pow_mat = zpow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]

            self.lr_classifier.fit(insample_pow_mat, insample_recalls)

            outsample_mask = ~insample_mask
            outsample_pow_mat = self.pow_mat[outsample_mask]
            # outsample_pow_mat = zpow_mat[outsample_mask]

            probs[outsample_mask] = self.lr_classifier.predict_proba(outsample_pow_mat)[:,1]

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
        return AUCs

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task + '_events')
        self.events = events
        self.pow_mat = normalize_sessions(self.get_passed_object('classify_pow_mat'), events)
        
        self.lr_classifier = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='auto',solver='liblinear')        

        event_sessions = events.session    
        recalls = events.recalled
        distErrs = np.sort(events.distErr)
        lowThresh = np.percentile(distErrs,5)
        highThresh = np.percentile(distErrs,95)
        thresholds =  np.arange(np.ceil(lowThresh),np.floor(highThresh))
    
        
        sessions = np.unique(event_sessions)
        
        self.model_output_thresh = ModelOutput()
        self.model_output_thresh.thresholds = thresholds
        self.model_output_thresh.aucs_by_thresh  = np.zeros(len(thresholds),dtype=np.float)
        self.model_output_thresh.pval_by_thresh  = np.zeros(len(thresholds),dtype=np.float)
        self.model_output_thresh.pCorr_by_thresh = np.zeros(len(thresholds),dtype=np.float)
        for i,thresh in enumerate(thresholds):

            recalls = events.distErr <= thresh
            self.model_output_thresh.pCorr_by_thresh[i] = np.mean(recalls)
            
            if len(sessions) > 1:
                perm_AUCs = self.permuted_loso_AUCs(event_sessions, recalls)

                probs = self.run_loso_xval(event_sessions, recalls, permuted=False)
                self.model_output_thresh.aucs_by_thresh[i] = roc_auc_score(recalls, probs)
                self.model_output_thresh.pval_by_thresh[i] = np.mean(self.model_output_thresh.aucs_by_thresh[i] < perm_AUCs)
                print 'Thresh:', thresh, 'AUC =', self.model_output_thresh.aucs_by_thresh[i], 'pval =', self.model_output_thresh.pval_by_thresh[i]
            else:
                sess = sessions[0]
                event_lists = deepcopy(events.trial)
                if self. params.doStratKFold:
                    skf = StratifiedKFold(recalls, n_folds=self.params.n_folds,shuffle=True)
                    for ind, (train_index, test_index) in enumerate(skf):
                        event_lists[test_index] = ind

                perm_AUCs = self.permuted_lolo_AUCs(sess, event_lists, recalls)            
                probs = self.run_lolo_xval(sess, event_lists, recalls, permuted=False)
                self.model_output_thresh.aucs_by_thresh[i] = roc_auc_score(recalls, probs)
                self.model_output_thresh.pval_by_thresh[i] = np.mean(self.model_output_thresh.aucs_by_thresh[i] < perm_AUCs)
                print 'Thresh:', thresh, 'AUC =', self.model_output_thresh.aucs_by_thresh[i], 'pval =', self.model_output_thresh.pval_by_thresh[i]

        self.pass_object('model_output_thresh', self.model_output_thresh)     
        joblib.dump(self.model_output_thresh, self.get_path_to_resource_in_workspace(subject + '-' + task + '-model_output_thresh.pkl'))             

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.model_output_thresh = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-model_output_thresh.pkl'))            
        self.pass_object('model_output_thresh', self.model_output_thresh)
     

