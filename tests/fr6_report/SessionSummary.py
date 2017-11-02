import numpy as np


class FR6SessionSummary(object):
    def __init__(self):
        # TODO: Loop over a list of state variables to do the initialization
        # Properties that are unique to a session
        self.session = None
        self.n_words = 0
        self.n_correct_words = 0
        self.pc_correct_words = 0
        self.n_lists = 0
        self.n_pli = 0
        self.pc_pli = 0
        self.n_eli = 0
        self.pc_eli = 0
        self.n_math = 0
        self.n_correct_math = 0
        self.pc_correct_math = 0
        self.math_per_list = 0
        self.n_correct_nonstim = 0
        self.n_total_nonstim = 0
        self.pc_from_nonstim = 0
        self.n_nonstim_intr = 0
        self.n_correct_nonstim_low_bio_items = 0
        self.n_total_nonstim_low_bio_items = 0
        self.pc_nonstim_post_low_bio_items = 0
        self.n_correct_nonstim_post_low_bio_items = 0
        self.n_total_nonstim_post_low_bio_items = 0
        self.pc_nonstim_post_low_bio_items = 0
        self.control_mean_prob_diff_all = 0
        self.control_sem_prob_diff_all = 0
        self.control_mean_prob_diff_low = 0
        self.control_sem_prob_diff_low = 0

        # Properties that vary by stim target
        self.stimtag = {}
        self.region_of_interest = {}
        self.frequency = {}
        self.amplitude = {}
        self.prob_recall = {}
        self.prob_stim_recall = {}
        self.prob_nostim_recall = {}
        self.prob_stim = {}
        self.prob_first_recall = {}
        self.prob_first_stim_recall = {}
        self.prob_first_nostim_recall = {}
        self.list_number = {}
        self.n_recalls_per_list = {}
        self.n_stims_per_list = {}
        self.is_stim_list = {}
        self.is_nonstim_list = {}
        self.is_baseline_list = {}
        self.is_ps_list = {}
        self.n_correct_stim = {}
        self.n_total_stim = {}
        self.pc_from_stim = {}
        self.mean_prob_diff_all_stim_item = {}
        self.sem_prob_diff_all_stim_item = {}
        self.mean_prob_diff_low_stim_item = {}
        self.sem_prob_diff_low_stim_item = {}
        self.mean_prob_diff_all_post_stim_item = {}
        self.sem_prob_diff_all_post_stim_item = {}
        self.mean_prob_diff_low_post_stim_item = {}
        self.sem_prob_diff_low_post_stim_item = {}
        self.chisqr = {}
        self.pvalue = {}
        self.n_stim_intr = {}
        self.n_nonstim_intr = {}
        self.pc_from_stim_intr = {}
        self.pc_from_nonstim_intr = {}
        self.chisqr_stim_item = {}
        self.pvalue_stim_item = {}
        self.chisqr_post_stim_item = {}
        self.pvalue_post_stim_item = {}
        self.n_correct_stim_items = {}
        self.n_total_stim_items = {}
        self.pc_stim_items = {}
        self.pc_diff_from_mean = {}
        self.n_correct_post_stim_items = {}
        self.n_total_post_stim_items = {}
        self.pc_post_stim_items = {}

        # Plot files
        self.STIM_AND_RECALL_PLOT_FILE = {}
        self.PROB_STIM_PLOT_FILE = {}
        self.STIM_VS_NON_STIM_HALVES_PLOT_FILE = {}
        self.PROB_RECALL_PLOT_FILE = {}
        # self.chisqr_last = -999
        # self.prob_stim_recall = []
        # self.prob_nostim_recall = []
        # self.prob_first_stim_recall = []
        # self.prob_first_nostim_recall = []
        # self.pc_diff_from_mean = []
        # self.n_stims_per_list = []
        # self.frequency = ''
        # self.list_number = []
        # self.is_stim_list = []
        # self.is_ps_list = []
        # self.is_baseline_list = []
        # self.prob_stim = []
        # self.pc_stim_hits = 0.0
        # self.pc_nonstim_hits = 0.0
        # self.pc_false_alarms = 0.0
        # self.pc_stim_item_hits = 0.0
        # self.pc_low_biomarker_hits = 0.0
        # self.dprime = -999.
    
    def __str__(self):
        """ Convenience representation of the object for debugging """
        for attr in dir(self):
            print("%s = %s" % (attr, getattr(self, attr)))
        return


class PS4SessionSummary(object):
    def __init__(self):
        self.locations = []
        self.amplitudes = []
        self.delta_classifiers = []
        self.PS_PLOT_FILE = ''
        self.preferred_location = ''
        self.preferred_amplitude = 0.
        self.pvalue = 0.
        self.tstat = 0.
