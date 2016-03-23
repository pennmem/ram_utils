__author__ = 'm'

class SessionSummary(object):
    def __init__(self):
        self.number = None
        self.name = None
        self.date = None
        self.length = None
        self.n_pairs = None
        self.n_correct_pairs = None
        self.pc_correct_pairs = None
        self.wilson1 = None
        self.wilson2 = None
        self.n_voc_pass = 0
        self.pc_voc_pass = 0.0
        self.n_nonvoc_pass = 189
        self.pc_nonvoc_pass = 63.0
        self.n_pli = None
        self.pc_pli = None
        self.n_eli = None
        self.pc_eli = None
        self.prob_recall = None
        self.study_lag_values = None
        self.prob_study_lag = None
        self.n_math = 0
        self.n_correct_math = 0
        self.pc_correct_math = 0.0
        self.math_per_list = 0.0
        self.auc = None
        self.ltt = None
        self.fpr = None
        self.tpr = None
        self.pc_diff_from_mean = None
        self.perm_AUCs = None
        self.perm_test_pvalue = None
        self.jstat_thresh = None
        self.jstat_percentile = None
