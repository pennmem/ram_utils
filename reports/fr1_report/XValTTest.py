import numpy as np
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import normalize
from RamPipeline import *

class XValResults(object):
    def __init__(self):
        self.auc_results = None


class XValTTest(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    def run(self):
        subject = self.pipeline.subject

        print 'Computing t-stats leave-one-session-out for', subject

        pow_mat = self.get_passed_object('pow_mat')
        freq_sel = (self.params.freqs>=self.params.ttest_frange[0]) & (self.params.freqs<=self.params.ttest_frange[1])
        pow_mat = pow_mat[:,:,freq_sel,:]
        pow_mat = np.mean(pow_mat, axis=(2,3))

        n_events, n_bps = pow_mat.shape

        events = self.get_passed_object(self.pipeline.task+'_events')
        sessions = np.unique(events.session)

        norm_func = normalize.standardize_pow_mat if self.params.norm_method == 'zscore' else normalize.normalize_pow_mat
        pow_mat = norm_func(pow_mat, events, sessions)[0]

        t_thresh_array = np.arange(start=1.0, stop=3.0, step=0.2)
        xval_results = XValResults()

        xval_results.auc_results =  np.recarray((len(t_thresh_array)*len(sessions),),dtype=[('t_thresh', float), ('outsample_session', int), ('auc', float), ('num_features',int) ])


        i=0
        for s_out in sessions:

            print 'Session', s_out

            insample_sel = (events.session!=s_out) & (events.list > 2)
            insample_events = events[insample_sel]

            insample_pow_mat = pow_mat[insample_sel,:]
            insample_recalls = np.array(insample_events.recalled, dtype=np.bool)

            recalled_insample_pow_mat = insample_pow_mat[insample_recalls,:]
            nonrecalled_insample_pow_mat = insample_pow_mat[~insample_recalls,:]

            t,p = ttest_ind(recalled_insample_pow_mat, nonrecalled_insample_pow_mat, axis=0, equal_var=False)

            for t_thresh in t_thresh_array:

                # feature_sel = (np.abs(t) >= 2.5)
                feature_sel = (np.abs(t) >= t_thresh)
                insample_pow_mat_tmp = insample_pow_mat[:,feature_sel]

                num_features = insample_pow_mat_tmp.shape[1]
                print num_features, 'features selected'

                lr_classifier = LogisticRegression(penalty=self.params.penalty_type, solver='liblinear')

                lr_classifier.fit(insample_pow_mat_tmp, insample_recalls)

                outsample_sel = (~insample_sel) & (events.list > 2)
                outsample_events = events[outsample_sel]

                outsample_pow_mat = pow_mat[outsample_sel,:]
                outsample_pow_mat = outsample_pow_mat[:,feature_sel]
                outsample_recalls = np.array(outsample_events.recalled, dtype=np.bool)

                outsample_probs = lr_classifier.predict_proba(outsample_pow_mat)[:,1]
                auc = roc_auc_score(outsample_recalls, outsample_probs)

                print 'AUC =', auc

                xval_results.auc_results[i].t_thresh = t_thresh
                xval_results.auc_results[i].outsample_session = s_out
                xval_results.auc_results[i].auc = auc
                xval_results.auc_results[i].num_features = num_features
                i+=1

        self.pass_object(name='xval_results',obj=xval_results)