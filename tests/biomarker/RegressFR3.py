from RamPipeline import *

import numpy as np
import pandas as pd
from scipy.stats.mstats import zscore, zmap
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import re


def get_biomarker_probs():
    probs = []
    re_prob = re.compile(r'prob=(?P<prob>\d.\d+), ')
    f = open('biomarker_probs.txt','r')
    for line in f:
        m = re_prob.match(line)
        probs.append(float(m.group('prob')))
    return probs


class RegressFR3(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.lr_classifier = None
        self.xval_output = None
        self.stim_item_mask = None

    def run(self):
        subject = self.pipeline.subject
        task3 = self.pipeline.task3

        events = self.get_passed_object(task3 + '_events')
        self.pow_mat = self.get_passed_object('pow_mat')
        self.lr_classifier = self.get_passed_object('lr_classifier')
        self.xval_output = self.get_passed_object('xval_output')
        self.stim_item_mask = self.get_passed_object(task3 + '_stim_item_mask')

        thresh = self.xval_output[-1].jstat_thresh

        print 'thresh =', thresh

        realtime_probs = np.array(get_biomarker_probs(), dtype=float)
        offline_probs = np.empty_like(realtime_probs)
        realtime_decisions = np.empty(len(realtime_probs), dtype=np.int)
        offline_decisions = np.empty_like(realtime_decisions, dtype=np.int)
        session_numbers = np.empty_like(realtime_decisions, dtype=np.int)
        list_numbers = np.empty_like(realtime_decisions, dtype=np.int)
        item_numbers = np.empty_like(realtime_decisions, dtype=np.int)
        i_bio_prob = 0

        event_sessions = np.unique(events.session)

        for sess in event_sessions:
            print 'Session', sess

            sess_event_mask = (events.session == sess)
            sess_events = events[sess_event_mask]
            sess_pow = self.pow_mat[sess_event_mask]
            sess_stim_item_mask = self.stim_item_mask[sess_event_mask]

            n_events = len(sess_events)

            lists = np.unique(sess_events.list)

            normalizing_mask = np.zeros(n_events, dtype=np.bool)
            for lst in lists:
                print 'List', lst
                list_event_mask = (sess_events.list==lst)
                list_events = sess_events[list_event_mask]
                if list_events[0].stimList == 1:
                    list_pow = zmap(sess_pow[list_event_mask], sess_pow[normalizing_mask], ddof=1)
                    list_stim_item_mask = sess_stim_item_mask[list_event_mask]
                    list_probs = self.lr_classifier.predict_proba(list_pow)[:,1]
                    for i,p in enumerate(list_probs):
                        realtime_prob = realtime_probs[i_bio_prob]
                        offline_probs[i_bio_prob] = p
                        realtime_decision = int(realtime_prob<thresh)
                        realtime_decisions[i_bio_prob] = realtime_decision
                        offline_decision = int(p<thresh)
                        offline_decisions[i_bio_prob] = offline_decision
                        session_numbers[i_bio_prob] = sess
                        list_numbers[i_bio_prob] = lst
                        item_numbers[i_bio_prob] = i
                        print 'realtime_prob =', realtime_prob, 'offline_prob =', p, 'realtime_decision =', realtime_decision, 'offline_decision =', offline_decision, ('PASS' if realtime_decision==offline_decision else 'FAIL')
                        i_bio_prob += 1
                else:
                    normalizing_mask |= list_event_mask

        regression_df = pd.DataFrame({'session_number':session_numbers, 'list_number':list_numbers, 'item_number':item_numbers, 'realtime_prob':realtime_probs, 'offline_prob':offline_probs, 'realtime_decision':realtime_decisions, 'offline_decison':offline_decisions})
        regression_hdf = pd.HDFStore('regression.h5')
        regression_hdf[subject + '_' + task3] = regression_df
        regression_hdf.close()

        self.pass_object('realtime_probs', realtime_probs)
        self.pass_object('offline_probs', offline_probs)
        self.pass_object('realtime_decisions', realtime_decisions)
        self.pass_object('offline_decisions', offline_decisions)
        self.pass_object('session_numbers', session_numbers)
