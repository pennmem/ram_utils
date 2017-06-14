import warnings

import numpy as np
from ReportUtils import ReportRamTask
from scipy.stats import ttest_ind
from scipy.stats.mstats import zscore
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


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


class AnalyzeAUC(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(AnalyzeAUC,self).__init__(mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.lr_classifier = None
        self.outsample_probs = None
        self.AUCs = None

        self.xval_output = dict()   # ModelOutput per session; xval_output[-1] is across all sessions
        self.perm_AUCs = None
        self.pvalue = None

    def initialize(self):
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name='fr1_events',
                                        access_path = ['experiments','fr1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='catfr1_events',
                                        access_path = ['experiments','catfr1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])

    def run_loso_xval(self, event_sessions, recalls, permuted=False):
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

    def run(self):
        events = self.get_passed_object('events')
        self.pow_mat = normalize_sessions(self.get_passed_object('pow_mat'), events)

        n_events = len(events)
        if n_events != self.pow_mat.shape[0]:
            raise Exception("Wrong power matrix size!")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lr_classifier = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='auto', solver='liblinear')

        event_sessions = events.session
        recalls = np.array(events.recalled, dtype=np.bool)

        print 'Performing leave-one-session-out xval'

        max_feat_num = min(501, self.pow_mat.shape[1])
        self.outsample_probs = np.empty(shape=(max_feat_num-1,n_events), dtype=float)

        sessions = np.unique(events.session)
        for heldout_sess in sessions:
            insample_mask = (event_sessions!=heldout_sess)
            outsample_mask = ~insample_mask

            insample_pow_mat = self.pow_mat[insample_mask,:]
            insample_recalls = recalls[insample_mask]
            insample_nonrecalls = ~insample_recalls

            outsample_pow_mat = self.pow_mat[outsample_mask,:]

            recalled_insample_pow_mat = insample_pow_mat[insample_recalls,:]
            nonrecalled_insample_pow_mat = insample_pow_mat[insample_nonrecalls,:]

            t,_ = ttest_ind(recalled_insample_pow_mat, nonrecalled_insample_pow_mat, axis=0)
            feat_idx_best_to_worst = np.argsort(-np.abs(t))
            for size in xrange(1,max_feat_num):
                selected_features = feat_idx_best_to_worst[:size]
                selected_insample_pow_mat = insample_pow_mat[:,selected_features]
                selected_outsample_pow_mat = outsample_pow_mat[:,selected_features]

                self.lr_classifier.fit(selected_insample_pow_mat, insample_recalls)
                probs = self.lr_classifier.predict_proba(selected_outsample_pow_mat)[:,1]
                self.outsample_probs[size-1,outsample_mask] = probs

        self.AUCs = np.empty(max_feat_num-1, dtype=float)

        for i in xrange(max_feat_num-1):
            self.AUCs[i] = roc_auc_score(recalls, self.outsample_probs[i])
            print i+1, 'features: AUC =', self.AUCs[i]

        self.pass_object('outsample_probs', self.outsample_probs)
        self.pass_object('AUCs', self.AUCs)

        joblib.dump(self.outsample_probs, self.get_path_to_resource_in_workspace('fs_outsample_probs.pkl'))
        joblib.dump(self.AUCs, self.get_path_to_resource_in_workspace('fs_AUCs.pkl'))


    def restore(self):
        self.outsample_probs = joblib.load(self.get_path_to_resource_in_workspace('fs_outsample_probs.pkl'))
        self.AUCs = joblib.load(self.get_path_to_resource_in_workspace('fs_AUCs.pkl'))

        self.pass_object('outsample_probs', self.outsample_probs)
        self.pass_object('AUCs', self.AUCs)
