from RamPipeline import *

from math import log
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from bisect import bisect_right
from scipy.stats import norm
import sys
from scipy.stats import describe
from ReportUtils import ReportRamTask

# def prob2perf(probs, true_labels, p):
#     idx = bisect_right(probs, p)
#     return np.sum(true_labels[0:idx]) / float(idx) if idx>0 else 0.0


def prob2perf_norm(xval_output, p):
    fi1 = fi0 = 1.0

    if p < 1e-6:
        return 0.0
    elif p < 1.0 - 1e-6:
        p_norm = log(p/(1.0-p))
        fi1 = norm.cdf(p_norm, loc=xval_output.mean1, scale=xval_output.pooled_std)
        fi0 = norm.cdf(p_norm, loc=xval_output.mean0, scale=xval_output.pooled_std)

    r = xval_output.n1*fi1 / (xval_output.n1*fi1 + xval_output.n0*fi0)
    return r


def bipolar_label_to_loc_tag(bp, loc_tags):
    if bp=='' or bp=='[]':
        return None
    label = bp[0]+'-'+bp[1]
    if label in loc_tags:
        lt = loc_tags[label]
        return lt if lt!='' and lt!='[]' else None
    label = bp[1]+'-'+bp[0]
    if label in loc_tags:
        lt = loc_tags[label]
        return lt if lt!='' and lt!='[]' else None
    else:
        return None


class ComputePSTable(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputePSTable,self).__init__(mark_as_completed)
        self.params = params
        self.ps_table = None

    def initialize(self):

        if self.dependency_inventory:

            self.dependency_inventory.add_dependent_resource(resource_name='fr1_events',
                                        access_path = ['experiments','fr1','events'])

            self.dependency_inventory.add_dependent_resource(resource_name='catfr1_events',
                                        access_path = ['experiments','catfr1','events'])

            self.dependency_inventory.add_dependent_resource(resource_name='ps_events',
                                        access_path = ['experiments','ps','events'])

            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])

            # self.dependency_inventory.add_dependent_resource(resource_name='fr1_info',
            #                             access_path = ['experiments','fr1','info'])


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
        xval_output = self.get_passed_object('xval_output')
        thresh = xval_output[-1].jstat_thresh

        ps_pow_mat_pre = self.get_passed_object('ps_pow_mat_pre')
        ps_pow_mat_post = self.get_passed_object('ps_pow_mat_post')

        control_table = self.get_passed_object('control_table')

        n_events = len(ps_events)

        prob_pre, prob_diff = self.compute_prob_deltas(ps_pow_mat_pre, ps_pow_mat_post, lr_classifier)
        #print 'prob_pre', describe(prob_pre)
        #print 'prob_diff', describe(prob_diff)
        #print 'prob_post', describe(prob_pre+prob_diff)
        #sys.exit(0)

        control_table_low = control_table[control_table['prob_pre']<thresh]
        control_table_high = control_table[control_table['prob_pre']>1.0-thresh]
        
        control_low_250 = control_table_low['prob_diff_250'].mean()
        control_low_500 = control_table_low['prob_diff_500'].mean()
        control_low_1000 = control_table_low['prob_diff_1000'].mean()

        control_high_250 = control_table_high['prob_diff_250'].mean()
        control_high_500 = control_table_high['prob_diff_500'].mean()
        control_high_1000 = control_table_high['prob_diff_1000'].mean()

        probs = xval_output[-1].probs
        true_labels = xval_output[-1].true_labels
        performance_map = sorted(zip(probs,true_labels))
        probs, true_labels = zip(*performance_map)
        probs = np.array(probs)
        true_labels = np.array(true_labels)
        total_recall_performance = np.sum(true_labels) / float(len(true_labels))

        # the code below is not pythonic, but I'll figure it out later
        perf_diff = np.zeros(n_events, dtype=float)
        prob_diff_control_low = np.zeros(n_events, dtype=float)
        prob_diff_control_high = np.zeros(n_events, dtype=float)
        perf_diff_control_low = np.zeros(n_events, dtype=float)
        perf_diff_control_high = np.zeros(n_events, dtype=float)
        for i in xrange(n_events):
            #perf_pre = prob2perf(probs, true_labels, prob_pre[i]+1e-7)
            #perf_diff[i] = 100.0*(prob2perf(probs, true_labels, prob_pre[i]+prob_diff[i]+1e-7) - perf_pre) / total_recall_performance
            perf_pre = prob2perf_norm(xval_output[-1], prob_pre[i])
            perf_diff[i] = 100.0*(prob2perf_norm(xval_output[-1], prob_pre[i]+prob_diff[i]) - perf_pre) / total_recall_performance
            prob_diff_low = prob_diff_high = prob_diff[i]
            if experiment=='PS2' or experiment=='PS3':
                prob_diff_low -= control_low_500
                prob_diff_high -= control_high_500
            else:
                if abs(ps_events[i].pulse_duration-250) < 1e-4:
                    prob_diff_low -= control_low_250
                    prob_diff_high -= control_high_250
                elif abs(ps_events[i].pulse_duration-1000) < 1e-4:
                    prob_diff_low -= control_low_1000
                    prob_diff_high -= control_high_1000
                else:
                    prob_diff_low -= control_low_500
                    prob_diff_high -= control_high_500

            prob_diff_control_low[i] = prob_diff_low
            prob_diff_control_high[i] = prob_diff_high
            # perf_diff_control_low[i] = 100.0*(prob2perf(probs, true_labels, prob_pre[i]+prob_diff_low+1e-7) - perf_pre) / total_recall_performance
            # perf_diff_control_high[i] = 100.0*(prob2perf(probs, true_labels, prob_pre[i]+prob_diff_high+1e-7) - perf_pre) / total_recall_performance
            perf_diff_control_low[i] = 100.0*(prob2perf_norm(xval_output[-1], prob_pre[i]+prob_diff_low) - perf_pre) / total_recall_performance
            perf_diff_control_high[i] = 100.0*(prob2perf_norm(xval_output[-1], prob_pre[i]+prob_diff_high) - perf_pre) / total_recall_performance

        #define region
        bipolar_label = pd.Series(zip([s.upper() for s in ps_events.stimAnodeTag], [s.upper() for s in ps_events.stimCathodeTag]))
        region = bipolar_label.apply(lambda bp: bipolar_label_to_loc_tag(bp, loc_tag))

        self.ps_table = pd.DataFrame()
        self.ps_table['session'] = ps_events.session
        self.ps_table['mstime'] = ps_events.mstime
        self.ps_table['sham'] = (ps_events.type == 'SHAM')
        self.ps_table['Pulse_Frequency'] = ps_events.pulse_frequency
        self.ps_table['Amplitude'] = ps_events.amplitude
        self.ps_table['Duration'] = ps_events.pulse_duration
        self.ps_table['Burst_Frequency'] = ps_events.burst_frequency
        self.ps_table['stimAnodeTag'] = ps_events.stimAnodeTag
        self.ps_table['stimCathodeTag'] = ps_events.stimCathodeTag
        self.ps_table['Region'] = region
        self.ps_table['prob_pre'] = prob_pre
        self.ps_table['prob_diff'] = prob_diff
        self.ps_table['perf_diff'] = perf_diff
        self.ps_table['prob_diff_with_control_low'] = prob_diff_control_low
        self.ps_table['perf_diff_with_control_low'] = perf_diff_control_low
        self.ps_table['prob_diff_with_control_high'] = prob_diff_control_high
        self.ps_table['perf_diff_with_control_high'] = perf_diff_control_high
        self.ps_table['isi'] = ps_events.isi
        
        self.pass_object('ps_table', self.ps_table)
        self.ps_table.to_pickle(self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_table.pkl'))

    def compute_prob_deltas(self, ps_pow_mat_pre, ps_pow_mat_post, lr_classifier):
        prob_pre = lr_classifier.predict_proba(ps_pow_mat_pre)[:,1]
        prob_post = lr_classifier.predict_proba(ps_pow_mat_post)[:,1]
        return prob_pre, prob_post - prob_pre
