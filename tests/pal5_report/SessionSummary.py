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
        self.irt_within_cat = None
        self.irt_between_cat = None
        self.list_number = None
        self.n_recalls_per_list = None
        self.n_stims_per_list = None
        self.is_stim_list = None
        self.n_correct_stim = None
        self.n_total_stim = None
        self.pc_from_stim = None
        self.n_correct_nonstim = None
        self.n_total_nonstim = None
        self.pc_from_nonstim = -1.0
        self.chisqr = -1.0
        self.pvalue = -1.0
        self.n_stim_intr = None
        self.pc_from_stim_intr = -1.0
        self.n_nonstim_intr = None
        self.pc_from_nonstim_intr = -1.0
        self.chisqr_intr = -1.0
        self.pvalue_intr = -1.0

        self.stim_vs_non_stim_pc_diff_from_mean = None

        self.mean_prob_diff_all = None
        self.sem_prob_diff_all = None
        self.mean_prob_diff_low = None
        self.sem_prob_diff_low = None
        self.control_mean_prob_diff_all = None
        self.control_sem_prob_diff_all = None
        self.control_mean_prob_diff_low = None
        self.control_sem_prob_diff_low = None
