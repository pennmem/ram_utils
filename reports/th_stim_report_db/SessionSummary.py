__author__ = 'm'

class SessionSummary(object):
    def __init__(self):
        self.sessions = None
        self.stimtag = None
        self.region_of_interest = None
        self.frequency = None
        self.n_lists = None
        self.n_words = None
        self.n_correct_words = None
        self.pc_correct_words = None
        self.n_mid_high_conf = None
        self.pc_mid_high_conf = None

        self.list_number = None
        self.is_stim_list = None
        self.n_recalls_per_list = None
        self.n_stims_per_list = None
        self.is_stim_list = None
        self.n_correct_stim = None
        self.n_total_stim = None
        self.pc_from_stim = None
        self.n_correct_nonstim = None
        self.n_total_nonstim = None
        self.pc_from_nonstim = None
        self.chisqr = None
        self.pvalue = None

        # low biomarker non-stim items
        self.n_correct_nonstim_low = None
        self.n_total_nonstim_low = None
        self.pc_from_nonstim_low = None
        self.chisqr_low = None
        self.pvalue_low = None

        # item level stim
        self.is_stim_item = None
        self.n_correct_stim_item = None
        self.dist_err_stim_item = None
        self.n_total_stim_item = None
        self.pc_from_stim_item = None
        self.n_correct_nonstim_item = None
        self.dist_err_nonstim_item = None
        self.n_total_nonstim_item = None
        self.pc_from_nonstim_item = None
        self.chisqr_item = None
        self.pvalue_item = None
        self.all_dist_errs = None
        self.correct_thresh = None

        # item level post stim
        self.is_post_stim_item = None
        self.n_correct_post_stim_item = None
        self.dist_err_post_stim_item = None
        self.n_total_post_stim_item = None
        self.pc_from_post_stim_item = None
        self.n_correct_post_nonstim_item = None
        self.dist_err_post_nonstim_item = None
        self.n_total_post_nonstim_item = None
        self.pc_from_post_nonstim_item = None
        self.chisqr_post_item = None
        self.pvalue_post_item = None

        self.n_stim_mid_high_conf = None
        self.pc_stim_mid_high_conf = None
        self.n_nonstim_mid_high_conf = None
        self.pc_nonstim_mid_high_conf = None
        self.chisqr_conf = None
        self.pvalue_conf = None
        #self.n_stim_intr = None
        #self.pc_from_stim_intr = None
        #self.n_nonstim_intr = None
        #self.pc_from_nonstim_intr = None
        #self.chisqr_intr = None
        #self.pvalue_intr = None

        self.auc = None
        self.auc_p = None

        self.stim_vs_non_stim_pc_diff_from_mean = None
        self.post_stim_vs_non_stim_pc_diff_from_mean = None

        self.mean_prob_diff_all_stim_item = None
        self.sem_prob_diff_all_stim_item = None
        self.mean_prob_diff_low_stim_item = None
        self.sem_prob_diff_low_stim_item = None

        self.mean_prob_diff_all_post_stim_item = None
        self.sem_prob_diff_all_post_stim_item = None
        self.mean_prob_diff_low_post_stim_item = None
        self.sem_prob_diff_low_post_stim_item = None

        self.control_mean_prob_diff_all = None
        self.control_sem_prob_diff_all = None
        self.control_mean_prob_diff_low = None
        self.control_sem_prob_diff_low = None
