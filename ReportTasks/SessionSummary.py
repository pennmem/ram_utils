class SessionSummary(object):
    def __init__(self):
        self.number=-999
        self.name = ''
        self.date = ''
        self.length = -99.99
        self.n_trials = 0
        self.n_correct_trials = -1
        self.pc_correct_trials = -500.0
        self.auc = []
        self.fpr = []
        self.tpr = []
        self.perm_AUCs =        []
        self.perm_test_pvalue = -1
        self.jstat_thresh =     0
        self.jstat_percentile = -1


class FR1SessionSummary(SessionSummary):
    def __init__(self):
        super(FR1SessionSummary,self).__init__()
        self.n_pli = None
        self.pc_pli = None
        self.n_eli = None
        self.pc_eli = None
        self.prob_recall = None
        self.prob_first_recall = None
        self.n_math = 0
        self.n_correct_math = 0
        self.pc_correct_math = 0.0
        self.math_per_list = 0.0
        self.ltt = None
        self.pc_diff_from_mean = None

class catFR1SessionSummary(FR1SessionSummary):
    def __init__(self):
        super(catFR1SessionSummary,self).__init__()
        self.repetition_ratio = -1.0
        self.irt_within_cat = -999
        self.irt_between_cat = -999


class PAL1SessionSummary(SessionSummary):
    def __init__(self):
        super(PAL1SessionSummary,self).__init__()
        self.wilson1 = 0.0
        self.wilson2 = 0.0
        self.n_voc_pass = None
        self.pc_voc_pass = None
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
        self.ltt = None
        self.pc_diff_from_mean = None
        self.positions = None

class TH1SessionSummary(SessionSummary):
    def __init__(self):
        super(TH1SessionSummary,self).__init__()
        self.n_transposed_items = None
        self.pc_transposed_items = None
        self.completed = None
        self.score = None
        self.mean_norm_err = None

        self.prob_by_conf = None
        self.percent_conf = None
        self.dist_hist = None
        self.err_by_block = None
        self.err_by_block_sem = None
        self.ltt = None

        self.pc_diff_from_mean = None
        self.aucs_by_thresh = None
        self.pval_by_thresh = None
        self.pCorr_by_thresh = None
        self.thresholds = None

        self.fpr_conf = None
        self.tpr_conf = None
        self.auc_conf = None
        self.pc_diff_from_mean_conf = None
        self.perm_AUCs_conf = None
        self.perm_test_pvalue_conf = None
        self.jstat_thresh_conf = None
        self.jstat_percentile_conf = None

        self.fpr_transpose = None
        self.tpr_transpose = None
        self.auc_transpose = None
        self.pc_diff_from_mean_transpose = None
        self.perm_AUCs_transpose = None
        self.perm_test_pvalue_transpose = None
        self.jstat_thresh_transpose = None
        self.jstat_percentile_transpose = None



