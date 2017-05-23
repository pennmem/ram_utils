__author__ = 'm'

class SessionSummary(object):
    def __init__(self):
        self.number = None
        self.name = None
        self.date = None
        self.length = None
        self.n_words = None
        self.n_correct_words = None
        self.pc_correct_words = None
        self.prob_recall = None
        self.prob_probe_recall = None
        self.auc = None
        self.ltt = None
        self.fpr = None
        self.tpr = None
        self.pc_diff_from_mean = None
        self.perm_AUCs = None
        self.perm_test_pvalue = None
        self.jstat_thresh = None
        self.jstat_percentile = None
        self.repetition_ratio = None
