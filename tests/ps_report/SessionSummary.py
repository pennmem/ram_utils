__author__ = 'm'

import numpy as np

class SessionSummary(object):
    def __init__(self):
        self.sess_num = None
        self.low_quantile_classifier_delta_plot = None
        self.low_quantile_recall_delta_plot = None
        self.high_quantile_classifier_delta_plot = None
        self.high_quantile_recall_delta_plot = None
        self.all_classifier_delta_plot = None
        self.all_recall_delta_plot = None
        self.stimtag = None
        self.region_of_interest = None
        self.const_param_value = None
        self.name = None
        self.date = None
        self.length = None
        self.isi_mid = None
        self.isi_half_range = None
        self.anova_fvalues = [np.nan, np.nan, np.nan]
        self.anova_pvalues = [np.nan, np.nan, np.nan]
        # self.param1_best = None
        # self.t1 = None
        # self.p1 = None
        self.param1_ttest_table = None
        # self.param2_best = None
        # self.t2 = None
        # self.p2 = None
        self.param2_ttest_table = None
        # self.param12_best = None
        # self.t12 = None
        # self.p12 = None
        self.param12_ttest_table = None
