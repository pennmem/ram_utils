from collections import defaultdict
import numpy as np
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

from ramutils.pipeline import RamTask


def none_function():
    return None


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

    def compute_auc(self):
        self.auc = roc_auc_score(self.true_labels, self.probs)


class ComputeAUCs(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.lr_classifier = None
        self.xval_output = None

    def run_loso_xval(self, event_sessions, recalls):
        probs = np.empty_like(recalls, dtype=np.float)

        sessions = np.unique(event_sessions)

        for sess in sessions:
            insample_mask = (event_sessions != sess)
            insample_pow_mat = self.pow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]

            self.lr_classifier.fit(insample_pow_mat, insample_recalls)

            outsample_mask = ~insample_mask
            outsample_pow_mat = self.pow_mat[outsample_mask]

            outsample_probs = self.lr_classifier.predict_proba(outsample_pow_mat)[:,1]
            probs[outsample_mask] = outsample_probs

        xval_output = ModelOutput(recalls, probs)
        xval_output.compute_auc()
        return xval_output

    def run(self):
        self.xval_output = defaultdict(none_function)

        task = self.pipeline.task
        subject = self.pipeline.subject

        events = self.get_passed_object(task + '_events')
        event_sessions = events.session
        sessions = np.unique(event_sessions)

        if (len(sessions) > 1) and (len(events) >= 480):
            self.pow_mat = normalize_sessions(self.get_passed_object('pow_mat'), events)
            recalls = events.recalled
            for C in self.params.Cs:
                self.lr_classifier = LogisticRegression(C=C, penalty=self.params.penalty_type, class_weight='balanced', solver='liblinear')

                print 'Performing leave-one-session-out xval'
                self.xval_output[C] = self.run_loso_xval(event_sessions, recalls)

        for C,xval_output in self.xval_output.iteritems():
            print 'C =', C, 'AUC =', xval_output.auc

        self.pass_object('xval_output', self.xval_output)

        joblib.dump(self.xval_output, self.get_path_to_resource_in_workspace(subject + '-' + task + '-xval_output.pkl'))
