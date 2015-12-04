from RamPipeline import *

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

import normalize


# from scipy.stats.mstats import zscore, zmap
# from numpy.linalg import norm
#
#
# def standardize_pow_mat(stripped_pow_mat, events, sessions, outsample_session, outsample_list):
#     zpow_mat = np.array(stripped_pow_mat)
#     outsample_mask = None
#     for session in sessions:
#         sess_event_mask = (events.session == session)
#         if session == outsample_session:
#             outsample_mask = (events.list == outsample_list) & sess_event_mask
#             insample_mask = ~outsample_mask & sess_event_mask
#             zpow_mat[outsample_mask] = zmap(zpow_mat[outsample_mask], zpow_mat[insample_mask], axis=0, ddof=1)
#             zpow_mat[insample_mask] = zscore(zpow_mat[insample_mask], axis=0, ddof=1)
#         else:
#             zpow_mat[sess_event_mask] = zscore(zpow_mat[sess_event_mask], axis=0, ddof=1)
#     return zpow_mat, outsample_mask
#
#
# def normalize_pow_mat(stripped_pow_mat, events, sessions, outsample_session, outsample_list):
#     normal_mat = np.array(stripped_pow_mat)
#     outsample_mask = None
#     for session in sessions:
#         sess_event_mask = (events.session == session)
#         if session == outsample_session:
#             outsample_mask = (events.list == outsample_list) & sess_event_mask
#             insample_mask = ~outsample_mask & sess_event_mask
#             insample_median = np.median(normal_mat[insample_mask], axis=0)
#             normal_mat[insample_mask] -= insample_median
#             insample_norm = norm(normal_mat[insample_mask], ord=1, axis=0)
#             normal_mat[insample_mask] /= insample_norm
#             normal_mat[outsample_mask] -= insample_median
#             normal_mat[outsample_mask] /= insample_norm
#         else:
#             med = np.median(normal_mat[sess_event_mask])
#             normal_mat[sess_event_mask] -= med
#             nrm = norm(normal_mat[sess_event_mask], ord=1, axis=0)
#             normal_mat[sess_event_mask] /= nrm
#     return normal_mat, outsample_mask


class ComputeClassifier(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.lr_classifiers = dict()
        self.probs = dict()
        self.timebins = dict()


    def leave_one_out_auc(self, events, sessions, stripped_pow_mat, C):
        lr_classifier = LogisticRegression(C=C, penalty=self.params.penalty_type, solver='liblinear')
        recalls = events.recalled
        probs = np.empty(shape=recalls.shape, dtype=np.float)
        for outsample_session in sessions:
            sess_event_mask = (events.session == outsample_session)
            lists = np.unique(events[sess_event_mask].list)
            for outsample_list in lists:
                norm_func = normalize.standardize_pow_mat if self.params.norm_method=='zscore' else normalize.normalize_pow_mat
                zpow_mat, outsample_mask = norm_func(stripped_pow_mat, events, sessions, outsample_session, outsample_list)
                insample_mask = ~outsample_mask
                lr_classifier.fit(zpow_mat[insample_mask], recalls[insample_mask])
                probs[outsample_mask] = lr_classifier.predict_proba(zpow_mat[outsample_mask])[:,1]
        return probs

    def run_sessions(self, sessions):
        print 'Optimizing logistic regression classifier for session(s)', sessions

        pow_mat = self.get_passed_object('pow_mat')

        events = self.get_passed_object(self.pipeline.task + '_events')
        sel = np.array([(ev.session in sessions) for ev in events], dtype=bool)

        events = events[sel]
        pow_mat = pow_mat[sel,:,:,:]

        n_events, n_bps, n_freqs, _ = pow_mat.shape

        assert n_events == len(events)

        recalls = events.recalled

        auc_best = 0.0
        t_best = C_best = probs_best = None
        for t in xrange(self.params.timewin_start, self.params.timewin_end-self.params.timewin_width, self.params.timewin_step):
            pow_mat_t = np.reshape(np.mean(pow_mat[:,:,:,t:t+self.params.timewin_width], axis=3), (n_events, n_bps*n_freqs))
            for C in self.params.Cs:
                probs = self.leave_one_out_auc(events, sessions, pow_mat_t, C)
                auc = roc_auc_score(recalls, probs)
                print 't =', t, 'C =', C, 'AUC =', auc
                if auc > auc_best:
                    auc_best = auc
                    t_best = t
                    C_best = C
                    probs_best = probs

        stripped_pow_mat = np.reshape(np.mean(pow_mat[:,:,:,t_best:t_best+self.params.timewin_width], axis=3), (n_events, n_bps*n_freqs))
        recalls = events.recalled

        lr_classifier = LogisticRegression(C=C_best, penalty=self.params.penalty_type, solver='liblinear')
        lr_classifier.fit(stripped_pow_mat, recalls)

        print 't_best =', t_best, 'C_best =', C_best, 'AUC =', auc_best

        return lr_classifier, probs_best, t_best

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task + '_events')
        sessions = np.unique(events.session)

        self.lr_classifiers = dict()
        self.probs = dict()
        self.timebins = dict()
        for sess in sessions:
            self.lr_classifiers[sess], self.probs[sess], self.timebins[sess] = self.run_sessions([sess])
        self.lr_classifiers[-1], self.probs[-1], self.timebins[-1] = self.run_sessions(sessions)

        self.pass_object('lr_classifiers', self.lr_classifiers)
        self.pass_object('probs', self.probs)
        self.pass_object('timebins', self.timebins)

        joblib.dump(self.lr_classifiers, self.get_path_to_resource_in_workspace(subject + '-' + task + '-lr_classifiers.pkl'))
        joblib.dump(self.probs, self.get_path_to_resource_in_workspace(subject + '-' + task + '-probs.pkl'))
        joblib.dump(self.timebins, self.get_path_to_resource_in_workspace(subject + '-' + task + '-timebins.pkl'))

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.lr_classifiers = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-lr_classifiers.pkl'))
        self.probs = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-probs.pkl'))
        self.timebins = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-timebins.pkl'))

        self.pass_object('lr_classifiers', self.lr_classifiers)
        self.pass_object('probs', self.probs)
        self.pass_object('timebins', self.timebins)
