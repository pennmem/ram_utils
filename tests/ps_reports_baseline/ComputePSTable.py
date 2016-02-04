from RamPipeline import *

import numpy as np
import pandas as pd
from sklearn.externals import joblib


class ComputePSTable(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.ps_table = None

    def restore(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.experiment
        self.ps_table = pd.read_pickle(self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_table.pkl'))
        self.pass_object('ps_table', self.ps_table)

    def run(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.experiment

        ps_events = self.get_passed_object(experiment+'_events')
        loc_tag = self.get_passed_object('loc_tag')

        lr_classifier = self.get_passed_object('lr_classifier')

        ps_pow_mat_pre = self.get_passed_object('ps_pow_mat_pre')
        ps_pow_mat_post = self.get_passed_object('ps_pow_mat_post')

        prob_pre, prob_diff = self.compute_prob_deltas(ps_pow_mat_pre, ps_pow_mat_post, lr_classifier)

        #define region
        bipolar_label = pd.Series(ps_events.stimAnodeTag) + '-' + pd.Series(ps_events.stimCathodeTag)
        bipolar_label = bipolar_label.apply(lambda bp: bp.upper())
        region = bipolar_label.apply(lambda bp: None if not (bp in loc_tag) or loc_tag[bp]=='' or loc_tag[bp]=='[]' else loc_tag[bp])

        self.ps_table = pd.DataFrame()
        self.ps_table['session'] = ps_events.session
        self.ps_table['mstime'] = ps_events.mstime
        self.ps_table['Pulse_Frequency'] = ps_events.pulse_frequency
        self.ps_table['Amplitude'] = ps_events.amplitude
        self.ps_table['Duration'] = ps_events.pulse_duration
        self.ps_table['Burst_Frequency'] = ps_events.burst_frequency
        self.ps_table['stimAnodeTag'] = ps_events.stimAnodeTag
        self.ps_table['stimCathodeTag'] = ps_events.stimCathodeTag
        self.ps_table['Region'] = region
        self.ps_table['prob_pre'] = prob_pre
        self.ps_table['prob_diff'] = prob_diff
        self.ps_table['isi'] = ps_events.isi
        
        self.pass_object('ps_table', self.ps_table)
        self.ps_table.to_pickle(self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_table.pkl'))

    def compute_prob_deltas(self, ps_pow_mat_pre, ps_pow_mat_post, lr_classifier):
        prob_pre = lr_classifier.predict_proba(ps_pow_mat_pre)[:,1]
        prob_post = lr_classifier.predict_proba(ps_pow_mat_post)[:,1]
        return prob_pre, prob_post - prob_pre
