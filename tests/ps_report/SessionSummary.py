__author__ = 'm'

import numpy as np

class SessionSummary(object):
    def __init__(self):
        self.sess_num = None
        self.plots = None
        self.constant_name = self.constant_value = self.constant_unit = None
        self.stimtag = None
        self.region_of_interest = None
        self.parameter1 = self.parameter2 = None
        self.name = None
        self.date = None
        self.length = None
        self.isi_mid = None
        self.isi_half_range = None
        self.anova_fvalues = [np.nan, np.nan, np.nan]
        self.anova_pvalues = [np.nan, np.nan, np.nan]
