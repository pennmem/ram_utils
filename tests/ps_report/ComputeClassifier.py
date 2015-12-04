from RamPipeline import *

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from scipy.stats.mstats import zscore, zmap


class ComputeClassifier(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.lr_classifier = None

    def standardize_pow_mat(self, stripped_pow_mat, events, sessions, outsample_session, outsample_list):
        zpow_mat = np.array(stripped_pow_mat)
        outsample_mask = None
        for session in sessions:
            sess_event_mask = (events.session == session)
            if session == outsample_session:
                outsample_mask = (events.list == outsample_list) & sess_event_mask
                insample_mask = ~outsample_mask & sess_event_mask
                zpow_mat[outsample_mask] = zmap(zpow_mat[outsample_mask], zpow_mat[insample_mask], axis=0, ddof=1)
                zpow_mat[insample_mask] = zscore(zpow_mat[insample_mask], axis=0, ddof=1)
            else:
                zpow_mat[sess_event_mask] = zscore(zpow_mat[sess_event_mask], axis=0, ddof=1)
        return zpow_mat, outsample_mask

    def leave_one_out_auc(self, events, sessions, stripped_pow_mat, C):
        self.lr_classifier = LogisticRegression(C=C, penalty=self.params.penalty_type, solver='liblinear')
        recalls = events.recalled
        probs = np.empty(shape=recalls.shape, dtype=np.float)
        for outsample_session in sessions:
            sess_event_mask = (events.session == outsample_session)
            lists = np.unique(events[sess_event_mask].list)
            for outsample_list in lists:
                zpow_mat, outsample_mask = self.standardize_pow_mat(stripped_pow_mat, events, sessions, outsample_session, outsample_list)
                insample_mask = ~outsample_mask
                self.lr_classifier.fit(zpow_mat[insample_mask], recalls[insample_mask])
                probs[outsample_mask] = self.lr_classifier.predict_proba(zpow_mat[outsample_mask])[:,1]
        return roc_auc_score(recalls, probs)

    def run(self):
        print 'Optimizing logistic regression classifier'

        self.pow_mat = self.get_passed_object('pow_mat')

        events = self.get_passed_object('FR1_events')
        sessions = np.unique(events.session)

        n_events, n_bps, n_freqs, _ = self.pow_mat.shape

        assert n_events == len(events)

        auc_best = 0.0
        t_best = C_best = None
        for t in xrange(self.params.timewin_start, self.params.timewin_end-self.params.timewin_width, self.params.timewin_step):
            pow_mat_t = np.reshape(np.mean(self.pow_mat[:,:,:,t:t+self.params.timewin_width], axis=3), (n_events, n_bps*n_freqs))
            for C in self.params.Cs:
                auc = self.leave_one_out_auc(events, sessions, pow_mat_t, C)
                print 't =', t, 'C =', C, 'AUC =', auc
                if auc > auc_best:
                    auc_best = auc
                    t_best = t
                    C_best = C

        stripped_pow_mat = np.reshape(np.mean(self.pow_mat[:,:,:,t_best:t_best+self.params.timewin_width], axis=3), (n_events, n_bps*n_freqs))
        recalls = events.recalled

        self.lr_classifier = LogisticRegression(C=C_best, penalty=self.params.penalty_type, solver='liblinear')
        self.lr_classifier.fit(stripped_pow_mat, recalls)

        print 't_best =', t_best, 'C_best =', C_best, 'AUC =', auc_best

        self.pass_object('lr_classifier', self.lr_classifier)
        joblib.dump(self.lr_classifier, self.get_path_to_resource_in_workspace(self.pipeline.subject + '-lr.pkl'))

    def restore(self):
        self.lr_classifier = joblib.load(self.get_path_to_resource_in_workspace(self.pipeline.subject + '-lr.pkl'))
        self.pass_object('lr_classifier', self.lr_classifier)
