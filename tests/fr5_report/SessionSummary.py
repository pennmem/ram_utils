class FR5SessionSummary(object):
    def __init__(self):
        self.STIM_AND_RECALL_PLOT_FILE = ''
        self.PROB_STIM_PLOT_FILE = ''
        self.STIM_VS_NON_STIM_HALVES_PLOT_FILE = ''
        self.PROB_RECALL_PLOT_FILE = ''
        self.STIMTAG = ''
        self.REGION = ''
        self.N_WORDS = 0.0
        self.N_CORRECT_WORDS = 0.0
        self.PC_CORRECT_WORDS = 0.0
        self.N_PLI = 0.0
        self.PC_PLI = 0.0
        self.N_ELI = 0.0
        self.PC_ELI = 0.0
        self.N_MATH = 0.0
        self.N_CORRECT_MATH = 0.0
        self.PC_CORRECT_MATH = 0.0
        self.MATH_PER_LIST = 0.0
        self.N_CORRECT_STIM = 0.0
        self.N_TOTAL_STIM = 0.0
        self.PC_FROM_STIM = 0.0
        self.N_CORRECT_NONSTIM = 0.0
        self.N_TOTAL_NONSTIM = 0.0
        self.PC_FROM_NONSTIM = 0.0
        self.CHISQR = 0.0
        self.PVALUE = ''
        self.ITEMLEVEL_COMPARISON = ''
        self.N_STIM_INTR = ''
        self.N_TOTAL_STIM = ''
        self.PC_FROM_STIM_INTR = ''
        self.N_NONSTIM_INTR = ''
        self.N_TOTAL_NONSTIM = ''
        self.PC_FROM_NONSTIM_INTR = ''
        self.prob_stim_recall = []
        self.prob_nostim_recall = []
        self.prob_first_stim_recall = []
        self.prob_first_nostim_recall = []
        self.pc_diff_from_mean = []
        self.n_stims_per_list = []
        self.frequency = ''
        self.list_number = []
        self.is_stim_list = []
        self.is_ps_list = []
        self.is_baseline_list = []
        self.prob_stim = []


class PS4SessionSummary(object):
    def __init__(self):
        self.LOC1 = ''
        self.LOC2 = ''
        self.loc_1_amplitudes = []
        self.loc_2_amplitudes = []
        self.loc1_delta_classifiers = []
        self.loc2_delta_classifiers = []
        self.PS_PLOT_FILE = ''
