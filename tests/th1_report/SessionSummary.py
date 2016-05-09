__author__ = 'm'

class SessionSummary(object):
    def __init__(self):
        self.number = None
        self.name = None
        self.date = None
        self.length = None
        self.n_items = None
        self.n_correct_items = None
        self.pc_correct_items = None
        self.n_transposed_items = None
        self.pc_transposed_items = None        
        self.completed = None
        self.score = None
        
        self.prob_by_conf = None
        self.percent_conf = None
        self.dist_hist = None
        self.err_by_block = None
        self.err_by_block_sem = None
        
        self.auc = None
        self.ltt = None
        self.fpr = None
        self.tpr = None
        
        
        self.pc_diff_from_mean = None
        self.perm_AUCs = None
        self.perm_test_pvalue = None
        self.jstat_thresh = None
        self.jstat_percentile = None
        
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