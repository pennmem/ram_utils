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


    def initialize(self):
        task_prefix = 'cat' if self.pipeline.task == 'RAM_CatFR1' else ''
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name=task_prefix+'fr1_events',
                                        access_path = ['experiments',task_prefix+'fr1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])

    def prepare_matrices(self, event_sessions):
        sessions = np.unique(event_sessions)
        self.matrices = dict()
        for sess in sessions:
            sess_sel_features = self.selected_features[sess]
            insample_mask = (event_sessions != sess)
            insample_ppc_features = self.ppc_features[insample_mask]
            insample_pows = self.pow_mat[insample_mask]
            #noise = 2*zscore(randn(insample_pows.shape[0], insample_pows.shape[1]), axis=0, ddof=1)
            #insample_features = insample_ppc_features[:,sess_sel_features]
            insample_features = np.concatenate((self.ppc_amplifier*insample_ppc_features, insample_pows), axis=1)
            outsample_mask = ~insample_mask
            outsample_ppc_features = self.ppc_features[outsample_mask]
            outsample_pows = self.pow_mat[outsample_mask]
            #noise = 2*zscore(randn(outsample_pows.shape[0], outsample_pows.shape[1]), axis=0, ddof=1)
            #outsample_features = outsample_ppc_features[:,sess_sel_features]
            outsample_features = np.concatenate((self.ppc_amplifier*outsample_ppc_features, outsample_pows), axis=1)
            self.matrices[sess] = (insample_features, outsample_features)


    def run_loso_xval(self, event_sessions, recalls, permuted=False):
        probs = np.empty_like(recalls, dtype=np.float)

        sessions = np.unique(event_sessions)

        for sess in sessions:
            sess_features = self.matrices[sess]

            insample_mask = (event_sessions != sess)
            #insample_ppc_features = self.ppc_features[insample_mask]
            #insample_ppc_features = insample_ppc_features[:,self.selected_features[sess]]
            insample_recalls = recalls[insample_mask]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                self.lr_classifier.fit(sess_features[0], insample_recalls)

            # print 'Outsample session', sess, 'nonzero weights'
            #
            # bipolar_pairs = self.get_passed_object('bipolar_pairs')
            # n_bps = len(bipolar_pairs)
            # nonzero_feature_idx = np.where(self.lr_classifier.coef_[0])[0]
            # print [feature_index_to_freq_and_elects(idx, n_bps) for idx in nonzero_feature_idx]

            outsample_mask = ~insample_mask
            #outsample_ppc_features = self.ppc_features[outsample_mask]
            #outsample_ppc_features = outsample_ppc_features[:,self.selected_features[sess]]
            outsample_recalls = recalls[outsample_mask]

            outsample_probs = self.lr_classifier.predict_proba(sess_features[1])[:,1]
            if not permuted:
                self.xval_output[sess] = ModelOutput(outsample_recalls, outsample_probs)
                self.xval_output[sess].compute_roc()
                self.xval_output[sess].compute_tercile_stats()
            probs[outsample_mask] = outsample_probs

        # if not permuted:
        #     for i in xrange(len(recalls)):
        #         print i, probs[i], recalls[i]

        if not permuted:
            self.xval_output[-1] = ModelOutput(recalls, probs)
            self.xval_output[-1].compute_roc()
            self.xval_output[-1].compute_tercile_stats()

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

    def heldout_auc(self, event_sessions, recalls):
        probs_opt = np.empty_like(recalls, dtype=np.float)
        probs = np.empty_like(recalls, dtype=np.float)
        sessions = np.unique(event_sessions)
        for heldout_sess in sessions:
            print 'Heldout session', heldout_sess
            heldout_mask = (event_sessions == heldout_sess)
            used_data_mask = ~heldout_mask
            amps = np.linspace(0.0, 30.0, 60)
            max_auc = 0.0
            best_amp = None
            for i in xrange(60):
                features = np.concatenate((amps[i]*self.ppc_features, self.pow_mat), axis=1)
                probs.fill(np.nan)
                for sess in sessions:
                    if sess != heldout_sess:
                        insample_mask = (event_sessions != sess) & (event_sessions != heldout_sess)
                        insample_recalls = recalls[insample_mask]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.lr_classifier.fit(features[insample_mask,:], insample_recalls)

                        outsample_mask = (event_sessions == sess)
                        outsample_probs = self.lr_classifier.predict_proba(features[outsample_mask,:])[:,1]
                        probs[outsample_mask] = outsample_probs

                auc = roc_auc_score(recalls[used_data_mask], probs[used_data_mask])
                if auc > max_auc:
                    max_auc = auc
                    best_amp = amps[i]

            print 'Best amp =', best_amp, 'max AUC =', max_auc

            features = np.concatenate((best_amp*self.ppc_features, self.pow_mat), axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.lr_classifier.fit(features[used_data_mask,:], recalls[used_data_mask])
                heldout_probs = self.lr_classifier.predict_proba(features[heldout_mask,:])[:,1]
                probs_opt[heldout_mask] = heldout_probs

        auc = roc_auc_score(recalls, probs_opt)
        print 'AUC =', auc
        return auc


    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        n_bps = len(bipolar_pairs)

        events = self.get_passed_object(task + '_events')
        #self.ppc_features = normalize_sessions(self.get_passed_object('ppc_features'), events)
        self.ppc_features = self.get_passed_object('ppc_features')
        self.selected_features = self.get_passed_object('selected_features')

        self.pow_mat = joblib.load('/scratch/mswat/automated_reports/FR1_reports/RAM_FR1_%s/%s-RAM_FR1-pow_mat.pkl' % (subject,subject))
        self.pow_mat = normalize_sessions(self.pow_mat, events)

        #n1 = np.sum(events.recalled)
        #n0 = len(events) - n1
        #w0 = (2.0/n0) / ((1.0/n0)+(1.0/n1))
        #w1 = (2.0/n1) / ((1.0/n0)+(1.0/n1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lr_classifier = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='auto', solver='liblinear')

        event_sessions = events.session
        recalls = events.recalled

        self.heldout_auc(event_sessions, recalls)

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
