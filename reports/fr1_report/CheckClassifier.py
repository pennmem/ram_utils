import numpy as np

import normalize
from RamPipeline import *


class CheckClassifier(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    def apply_classifier_on_session(self, sess, lr_classifier, timebin):
        events = self.get_passed_object(self.pipeline.task + '_events')
        sessions = np.unique(events.session)

        sel = (events.session == sess)
        events = events[sel]

        pow_mat = self.get_passed_object('pow_mat')
        pow_mat = pow_mat[sel,:,:,:]

        n_events, n_bps, n_freqs, _ = pow_mat.shape

        pow_mat = np.reshape(np.mean(pow_mat[:,:,:,timebin:timebin+self.params.timewin_width], axis=3), (n_events, n_bps*n_freqs))

        norm_func = normalize.standardize_pow_mat if self.params.norm_method == 'zscore' else normalize.normalize_pow_mat
        pow_mat = norm_func(pow_mat, events, sessions)[0]

        return lr_classifier.predict_proba(pow_mat)[:,1]

    def run(self):
        task = self.pipeline.task

        events = self.get_passed_object(task + '_events')
        sessions = np.unique(events.session)

        lr_classifiers = self.get_passed_object('lr_classifiers')
        probs = self.get_passed_object('probs')
        timebins = self.get_passed_object('timebins')

        for sess0 in sessions:
            lr_classifier = lr_classifiers[sess0]
            probs_sess0 = probs[sess0]
            timebin = timebins[sess0]
            # probs0 = self.apply_classifier_on_session(sess0, lr_classifier, timebin)
            low_terc_sess0 = np.percentile(probs_sess0, 100.0/3.0)
            print 'Session', sess0, ':', len(probs_sess0), 'WORD events, lowest tercile threshold =', low_terc_sess0

            for sess in sessions:
                if sess != sess0:
                    probs_sess = self.apply_classifier_on_session(sess, lr_classifier, timebin)
                    n_stim = np.sum(probs_sess<low_terc_sess0)
                    print 'Session', sess, ':', len(probs_sess), 'WORD events,', n_stim, 'would be stimulated'
