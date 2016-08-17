from RamPipeline import *

from math import sqrt
import numpy as np
from numpy.linalg import norm
from numpy.random import randn
from scipy.stats.mstats import zscore
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from random import shuffle
from sklearn.externals import joblib
import warnings

from ReportUtils import ReportRamTask

from scipy.stats import describe


def normalize_sessions(pow_mat, events):
    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)
    return pow_mat

def norm2_sessions(ppc_features, events):
    print norm(ppc_features.reshape(-1))
    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)
        ppc_features[sess_event_mask] = normalize(ppc_features[sess_event_mask], norm='l2', axis=0)
    print norm(ppc_features.reshape(-1))
    return ppc_features

def feature_index_to_freq_and_elects(feature_idx, n_bps):
    n_bp_pairs = n_bps * (n_bps-1) / 2
    freq = feature_idx / n_bp_pairs
    bp_pair_idx = feature_idx - freq*n_bp_pairs
    i = int((1.0 + sqrt(1.00001+8.0*bp_pair_idx)) / 2.0)
    j = bp_pair_idx - i*(i-1)/2
    return freq, i, j

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


class ComputeClassifier(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeClassifier,self).__init__(mark_as_completed)
        self.params = params
        self.ppc_features = None
        self.selected_features = None
        self.pow_mat = None
        self.ppc_amplifier = None
        self.matrices = None
        self.lr_classifier = None
        self.xval_output = dict()   # ModelOutput per session; xval_output[-1] is across all sessions
        self.perm_AUCs = None
        self.pvalue = None
        self.fpr = None
        self.tpr = None
        self.thresholds = None
        self.heldout_AUCs = None
        self.heldout_Cs = None
        self.heldout_amps = None
        self.auc = None


    def initialize(self):
        task_prefix = 'cat' if self.pipeline.task == 'RAM_CatFR1' else ''
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name=task_prefix+'fr1_events',
                                        access_path = ['experiments',task_prefix+'fr1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])

    # def prepare_matrices(self, event_sessions):
    #     sessions = np.unique(event_sessions)
    #     self.matrices = dict()
    #     for sess in sessions:
    #         sess_sel_features = self.selected_features[sess]
    #         insample_mask = (event_sessions != sess)
    #         insample_ppc_features = self.ppc_features[insample_mask]
    #         insample_pows = self.pow_mat[insample_mask]
    #         insample_features = np.concatenate((self.ppc_amplifier*insample_ppc_features, insample_pows), axis=1)
    #         outsample_mask = ~insample_mask
    #         outsample_ppc_features = self.ppc_features[outsample_mask]
    #         outsample_pows = self.pow_mat[outsample_mask]
    #         outsample_features = np.concatenate((self.ppc_amplifier*outsample_ppc_features, outsample_pows), axis=1)
    #         self.matrices[sess] = (insample_features, outsample_features)

    # def prepare_matrices(self, event_sessions):
    #     sessions = np.unique(event_sessions)
    #     self.matrices = dict()
    #     for sess in sessions:
    #         sess_sel_features = self.selected_features[sess]
    #         insample_mask = (event_sessions != sess)
    #         insample_features = self.ppc_features[insample_mask]
    #         insample_features = insample_features[:,sess_sel_features]
    #         outsample_mask = ~insample_mask
    #         outsample_features = self.ppc_features[outsample_mask]
    #         outsample_features = outsample_features[:,sess_sel_features]
    #         self.matrices[sess] = (insample_features, outsample_features)

    def run_loso_xval_power(self, event_sessions, recalls, permuted=False):
        probs = np.empty_like(recalls, dtype=np.float)

        sessions = np.unique(event_sessions)

        for sess in sessions:
            insample_mask = (event_sessions != sess)
            insample_pow_mat = self.pow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                self.lr_classifier.fit(insample_pow_mat, insample_recalls)

            outsample_mask = ~insample_mask
            outsample_pow_mat = self.pow_mat[outsample_mask]
            outsample_recalls = recalls[outsample_mask]

            outsample_probs = self.lr_classifier.predict_proba(outsample_pow_mat)[:,1]
            if not permuted:
                self.xval_output[sess] = ModelOutput(outsample_recalls, outsample_probs)
                self.xval_output[sess].compute_roc()
                self.xval_output[sess].compute_tercile_stats()
            probs[outsample_mask] = outsample_probs

        if not permuted:
            self.xval_output[-1] = ModelOutput(recalls, probs)
            self.xval_output[-1].compute_roc()
            self.xval_output[-1].compute_tercile_stats()

        return probs

    def run_loso_xval(self, event_sessions, recalls):
        probs = np.empty_like(recalls, dtype=np.float)

        sessions = np.unique(event_sessions)

        for sess in sessions:
            insample_mask = (event_sessions != sess)
            insample_recalls = recalls[insample_mask]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                self.lr_classifier.fit(self.ppc_features[insample_mask,:], insample_recalls)

            outsample_mask = ~insample_mask
            outsample_recalls = recalls[outsample_mask]

            sess_outsample_ppc_features = self.ppc_features[outsample_mask,:]
            outsample_probs = self.lr_classifier.predict_proba(sess_outsample_ppc_features)[:,1]
            #outsample_probs_non_recall = 1.0 - self.lr_classifier.predict_proba(-sess_outsample_ppc_features)[:,1]
            #outsample_probs = outsample_probs_recall / (outsample_probs_recall + outsample_probs_non_recall)
            probs[outsample_mask] = outsample_probs

        self.auc = roc_auc_score(recalls, probs)
        self.fpr, self.tpr, self.thresholds = roc_curve(recalls, probs)
        print 'AUC =', self.auc
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
            AUCs[i] = roc_auc_score(permuted_recalls, probs)
            #print 'AUC =', AUCs[i]
        return AUCs


    def heldout_auc(self, event_sessions, recalls):
        probs_opt = np.empty_like(recalls, dtype=np.float)
        probs = np.empty_like(recalls, dtype=np.float)
        sessions = np.unique(event_sessions)
        self.heldout_AUCs = dict()
        self.heldout_amps = dict()
        for heldout_sess in sessions:
            print 'Heldout session', heldout_sess
            heldout_mask = (event_sessions == heldout_sess)
            used_data_mask = ~heldout_mask
            amps = np.logspace(np.log10(1e-2), np.log10(1e4), 50)
            max_auc = 0.0
            best_amp = None
            for i in xrange(50):
                #features = np.concatenate((amps[i]*self.ppc_features, self.pow_mat), axis=1)
                probs.fill(np.nan)
                for sess in sessions:
                    if sess != heldout_sess:
                        insample_mask = (event_sessions != sess) & (event_sessions != heldout_sess)
                        insample_recalls = recalls[insample_mask]
                        insample_features = self.ppc_features[insample_mask,:]

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.lr_classifier.fit(np.concatenate((amps[i]*insample_features,self.pow_mat[insample_mask,:]), axis=1), insample_recalls)

                        outsample_mask = (event_sessions == sess)
                        outsample_features = self.outsample_ppc_features[outsample_mask,:]
                        outsample_probs = self.lr_classifier.predict_proba(np.concatenate((amps[i]*outsample_features,self.pow_mat[outsample_mask,:]), axis=1))[:,1]
                        probs[outsample_mask] = outsample_probs

                auc = roc_auc_score(recalls[used_data_mask], probs[used_data_mask])
                print 'amp =', amps[i], 'AUC =', auc
                if auc > max_auc:
                    max_auc = auc
                    best_amp = amps[i]

            self.heldout_AUCs[heldout_sess] = max_auc
            self.heldout_amps[heldout_sess] = best_amp
            print 'Best amp =', best_amp, 'max AUC =', max_auc

            features = np.concatenate((best_amp*self.ppc_features[used_data_mask], self.pow_mat[used_data_mask]), axis=1)
            outsample_features = np.concatenate((best_amp*self.outsample_ppc_features[heldout_mask], self.pow_mat[heldout_mask]), axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.lr_classifier.fit(features, recalls[used_data_mask])
                heldout_probs = self.lr_classifier.predict_proba(outsample_features)[:,1]
                probs_opt[heldout_mask] = heldout_probs

        self.auc = roc_auc_score(recalls, probs_opt)
        self.fpr, self.tpr, self.thresholds = roc_curve(recalls, probs_opt)
        print 'AUC =', self.auc
        return probs_opt


    def heldout_auc_ppc_only(self, event_sessions, recalls):
        probs_opt = np.empty_like(recalls, dtype=np.float)
        probs = np.empty_like(recalls, dtype=np.float)
        sessions = np.unique(event_sessions)
        self.heldout_AUCs = dict()
        self.heldout_Cs = dict()
        for heldout_sess in sessions:
            print 'Heldout session', heldout_sess
            heldout_mask = (event_sessions == heldout_sess)
            used_data_mask = ~heldout_mask
            C = np.logspace(np.log10(1e-5), np.log10(1e3), 50)
            max_auc = 0.0
            best_C = None
            for i in xrange(50):
                probs.fill(np.nan)
                for sess in sessions:
                    if sess != heldout_sess:
                        insample_mask = (event_sessions != sess) & (event_sessions != heldout_sess)
                        insample_recalls = recalls[insample_mask]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.lr_classifier = LogisticRegression(C=C[i], penalty=self.params.penalty_type, class_weight='auto', solver='liblinear')
                            self.lr_classifier.fit(self.ppc_features[insample_mask,:], insample_recalls)

                        outsample_mask = (event_sessions == sess)
                        outsample_probs = self.lr_classifier.predict_proba(self.outsample_ppc_features[outsample_mask,:])[:,1]
                        probs[outsample_mask] = outsample_probs

                auc = roc_auc_score(recalls[used_data_mask], probs[used_data_mask])
                print 'C =', C[i], 'AUC =', auc
                if auc > max_auc:
                    max_auc = auc
                    best_C = C[i]

            self.heldout_AUCs[heldout_sess] = max_auc
            self.heldout_Cs[heldout_sess] = best_C
            print 'Best C =', best_C, 'max AUC =', max_auc

            features = self.ppc_features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.lr_classifier.fit(self.ppc_features[used_data_mask,:], recalls[used_data_mask])
                heldout_probs = self.lr_classifier.predict_proba(self.outsample_ppc_features[heldout_mask,:])[:,1]
                probs_opt[heldout_mask] = heldout_probs

        self.auc = roc_auc_score(recalls, probs_opt)
        self.fpr, self.tpr, self.thresholds = roc_curve(recalls, probs_opt)
        print 'AUC =', self.auc
        return probs_opt


    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task + '_events')
        self.ppc_features = self.get_passed_object('ppc_features')
        self.outsample_ppc_features = self.get_passed_object('outsample_ppc_features')

        #temporary reshape - will do it in computing piece
        #n_samples, n_features = self.ppc_features.shape
        self.outsample_ppc_features = self.outsample_ppc_features.reshape(self.ppc_features.shape)
        #print 'Features difference:', describe((self.ppc_features-self.outsample_ppc_features).reshape(-1))

        self.pow_mat = joblib.load('/scratch/mswat/automated_reports/FR1_reports/RAM_FR1_%s/%s-RAM_FR1-pow_mat.pkl' % (subject,subject))
        self.pow_mat = normalize_sessions(self.pow_mat, events)

        #self.ppc_features = norm2_sessions(self.ppc_features, events)
        #self.outsample_ppc_features = norm2_sessions(self.outsample_ppc_features, events)

        #n1 = np.sum(events.recalled)
        #n0 = len(events) - n1
        #w0 = (2.0/n0) / ((1.0/n0)+(1.0/n1))
        #w1 = (2.0/n1) / ((1.0/n0)+(1.0/n1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lr_classifier = LogisticRegression(dual=True, C=self.params.C, penalty=self.params.penalty_type, class_weight='auto', solver='liblinear')

        event_sessions = events.session
        recalls = events.recalled

        #self.permuted_loso_AUCs(event_sessions, recalls)
        #probs = self.run_loso_xval(event_sessions, recalls)
        #joblib.dump(probs, self.get_path_to_resource_in_workspace(subject + '-' + task + '-probs.pkl'))
        #joblib.dump(self.fpr, self.get_path_to_resource_in_workspace(subject + '-' + task + '-fpr.pkl'))
        #joblib.dump(self.fpr, self.get_path_to_resource_in_workspace(subject + '-' + task + '-fpr.pkl'))
        #joblib.dump(self.tpr, self.get_path_to_resource_in_workspace(subject + '-' + task + '-tpr.pkl'))
        #joblib.dump(self.thresholds, self.get_path_to_resource_in_workspace(subject + '-' + task + '-thresholds.pkl'))
        #joblib.dump(self.auc, self.get_path_to_resource_in_workspace(subject + '-' + task + '-auc.pkl'))

        self.heldout_auc(event_sessions, recalls)
        joblib.dump(self.fpr, self.get_path_to_resource_in_workspace(subject + '-' + task + '-both_fpr.pkl'))
        joblib.dump(self.tpr, self.get_path_to_resource_in_workspace(subject + '-' + task + '-both_tpr.pkl'))
        joblib.dump(self.thresholds, self.get_path_to_resource_in_workspace(subject + '-' + task + '-both_thresholds.pkl'))
        joblib.dump(self.auc, self.get_path_to_resource_in_workspace(subject + '-' + task + '-both_auc.pkl'))
        joblib.dump(self.heldout_amps, self.get_path_to_resource_in_workspace(subject + '-' + task + '-both_heldout_amps.pkl'))
        joblib.dump(self.heldout_AUCs, self.get_path_to_resource_in_workspace(subject + '-' + task + '-both_heldout_AUCs.pkl'))

        # probs = self.heldout_auc_ppc_only(event_sessions, recalls)
        # joblib.dump(probs, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ppc_only_probs.pkl'))
        # joblib.dump(self.fpr, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ppc_only_fpr.pkl'))
        # joblib.dump(self.tpr, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ppc_only_tpr.pkl'))
        # joblib.dump(self.thresholds, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ppc_only_thresholds.pkl'))
        # joblib.dump(self.auc, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ppc_only_auc.pkl'))
        # joblib.dump(self.heldout_Cs, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ppc_only_heldout_Cs.pkl'))
        # joblib.dump(self.heldout_AUCs, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ppc_only_heldout_AUCs.pkl'))


    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.lr_classifier = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-lr_classifier.pkl'))
        self.xval_output = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-xval_output.pkl'))
        self.perm_AUCs = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-perm_AUCs.pkl'))
        self.pvalue = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-pvalue.pkl'))

        self.pass_object('lr_classifier', self.lr_classifier)
        self.pass_object('xval_output', self.xval_output)
        self.pass_object('perm_AUCs', self.perm_AUCs)
        self.pass_object('pvalue', self.pvalue)
