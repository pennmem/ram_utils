from ram_utils.RamPipeline import *

import numpy as np
from scipy.stats import ttest_ind
from sklearn.externals import joblib

from scipy.stats import describe
from scipy.stats.mstats import zscore
from ReportUtils import ReportRamTask

def normalize_sessions(pow_mat, events):
    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)
    return pow_mat

class ComputeTTest(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeTTest,self).__init__(mark_as_completed)
        self.params = params

    def run(self):
        print 'Computing t-stats'

        subject = self.pipeline.subject
        task = self.pipeline.task
        events = self.get_passed_object(self.pipeline.task+'_events')
        sessions = np.unique(events.session)
        recalls = np.array(events.recalled, dtype=np.bool)
        
        # get pow mat, normalize, and reshape for easier frequency averaging
        pow_mat = self.get_passed_object('ttest_pow_mat')
        pow_mat = normalize_sessions(pow_mat,events)
        freq_bins = self.params.ttest_frange
        n_elecs = pow_mat.shape[1] / self.params.freqs.size         
        pow_mat_reshape = np.reshape(pow_mat,(len(events), n_elecs,pow_mat.shape[1]/n_elecs))
        
        # loop over each frequency been in params
        # dict will hold ttest results for each bin
        self.ttest = {}
        for bin_count, freq_bin in enumerate(freq_bins):

            # boolean for frequencies in range
            freq_sel = (self.params.freqs>=freq_bin[0]) & (self.params.freqs<=freq_bin[1])
            
            # reduce to just freq bin and mean
            pow_mat_bin = pow_mat_reshape[:,:,freq_sel]        
            pow_mat_bin = np.mean(pow_mat_bin,axis=2)
            print 'Power Matrix stats for %.2f - %.2f:'%(freq_bin[0],freq_bin[1])
            print describe(pow_mat_bin, axis=None, ddof=1)
            
            # split into rec and nonrec
            recalled_pow_mat = pow_mat_bin[recalls]
            nonrecalled_pow_mat = pow_mat_bin[~recalls]

            # 2 sample ttest
            t,p = ttest_ind(recalled_pow_mat, nonrecalled_pow_mat, axis=0)
            self.ttest[bin_count] = (t,p,freq_bin)
        self.pass_object('ttest', self.ttest)
        joblib.dump(self.ttest, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ttest.pkl'))

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        self.ttest = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-ttest.pkl'))
        self.pass_object('ttest', self.ttest)
