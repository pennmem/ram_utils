__author__ = 'm'

import numpy as np
import pandas as pd
import time
from copy import deepcopy

from RamPipeline import *
from SessionSummary import SessionSummary

from PlotUtils import PlotData

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import ttest_ind

from sklearn.externals import joblib

from collections import OrderedDict


def classifier_delta_plot_data(ps_table, control_series, param1_name, param2_name, param2_unit):
    plots = OrderedDict()
    param1_vals = sorted(ps_table[param1_name].unique())
    param2_vals = sorted(ps_table[param2_name].unique())
    for p2, val2 in enumerate(param2_vals):
        ps_table_val2 = ps_table[ps_table[param2_name]==val2]
        means = np.empty(len(param1_vals), dtype=float)
        sems = np.empty(len(param1_vals), dtype=float)
        for i,val1 in enumerate(param1_vals):
            ps_table_val1_val2 = ps_table_val2[ps_table_val2[param1_name]==val1]
            means[i] = ps_table_val1_val2['prob_diff'].mean()
            sems[i] = ps_table_val1_val2['prob_diff'].sem()
        plots[val2] = PlotData(x=np.arange(2,len(param1_vals)+2)-p2*0.1,
                               y=means, yerr=sems,
                               label=param2_name+' '+str(val2)+' '+param2_unit
                               )
    control_means = np.empty(len(param1_vals)+1, dtype=float)
    control_mean = control_series.mean()
    control_means[0] = control_mean
    control_means[1:] = np.NAN
    control_sems = np.empty(len(param1_vals)+1, dtype=float)
    control_sems[0] = control_series.sem()
    control_sems[1:] = np.NAN
    plots['CONTROL'] = PlotData(x=np.arange(1,len(param1_vals)+2), y=control_means, yerr=control_sems, x_tick_labels=['CTRL']+[x if x>0 else 'PULSE' for x in param1_vals], xhline_pos=control_mean, color='k', markersize=10.0, elinewidth=3.0)
    return plots


def recall_delta_plot_data(ps_table, delta_column_name, param1_name, param2_name, param2_unit):
    plots = OrderedDict()
    param1_vals = sorted(ps_table[param1_name].unique())
    param2_vals = sorted(ps_table[param2_name].unique())
    for p2, val2 in enumerate(param2_vals):
        ps_table_val2 = ps_table[ps_table[param2_name]==val2]
        means = np.empty(len(param1_vals), dtype=float)
        sems = np.empty(len(param1_vals), dtype=float)
        for i,val1 in enumerate(param1_vals):
            ps_table_val1_val2 = ps_table_val2[ps_table_val2[param1_name]==val1]
            means[i] = ps_table_val1_val2[delta_column_name].mean()
            sems[i] = ps_table_val1_val2[delta_column_name].sem()
        plots[val2] = PlotData(x=np.arange(1,len(param1_vals)+1)-p2*0.1,
                               y=means, yerr=sems, x_tick_labels=[x if x>0 else 'PULSE' for x in param1_vals],
                               label=param2_name+' '+str(val2)+' '+param2_unit
                               )
    return plots


def anova_test(ps_table, param1_name, param2_name):
    if len(ps_table) < 10:
        return None
    ps_lm = ols('perf_diff ~ C(%s) * C(%s)' % (param1_name,param2_name), data=ps_table).fit()
    anova = anova_lm(ps_lm)
    return (anova['F'].values[0:3], anova['PR(>F)'].values[0:3])


# def ttest_one_param(ps_table, param_name):
#     param_vals = sorted(ps_table[param_name].unique())
#     val_max = param_vals[np.argmax([ps_table[ps_table[param_name]==val]['perf_diff'].mean() for val in param_vals])]
#     val_max_sel = (ps_table[param_name]==val_max)
#     population1 = ps_table[val_max_sel]['perf_diff'].values
#     population2 = ps_table[~val_max_sel]['perf_diff'].values
#     t,p = ttest_ind(population1, population2)
#     return val_max,t,p


def ttest_one_param(ps_table, param_name):
    ttest_table = []
    param_vals = sorted(ps_table[param_name].unique())
    for val in param_vals:
        val_sel = (ps_table[param_name]==val)
        population1 = ps_table[val_sel]['perf_diff'].values
        population2 = ps_table[~val_sel]['perf_diff'].values
        t,p = ttest_ind(population1, population2)
        if p<0.05 and t>0.0:
            ttest_table.append([val if val>=0 else 'PULSE', p, t])
    return ttest_table


# def ttest_interaction(ps_table, param1_name, param2_name):
#     param1_vals = sorted(ps_table[param1_name].unique())
#     param2_vals = sorted(ps_table[param2_name].unique())
#     mean_max = -1.0
#     val1_max = val2_max = None
#     for val1 in param1_vals:
#         for val2 in param2_vals:
#             ps_table_val1_val2 = ps_table[(ps_table[param1_name]==val1) & (ps_table[param2_name]==val2)]
#             mean = ps_table_val1_val2['perf_diff'].mean()
#             if mean > mean_max:
#                 mean_max = mean
#                 val1_max = val1
#                 val2_max = val2
#
#     val_max_sel = (ps_table[param1_name]==val1_max) & (ps_table[param2_name]==val2_max)
#
#     population1 = ps_table[val_max_sel]['perf_diff'].values
#     population2 = ps_table[~val_max_sel]['perf_diff'].values
#     t,p = ttest_ind(population1, population2)
#     return (val1_max,val2_max),t,p


def ttest_interaction(ps_table, param1_name, param2_name):
    ttest_table = []
    param1_vals = sorted(ps_table[param1_name].unique())
    param2_vals = sorted(ps_table[param2_name].unique())
    for val1 in param1_vals:
        val1_sel = (ps_table[param1_name]==val1)
        for val2 in param2_vals:
            val2_sel = (ps_table[param2_name]==val2)
            sel = val1_sel & val2_sel
            population1 = ps_table[sel]['perf_diff'].values
            population2 = ps_table[~sel]['perf_diff'].values
            t,p = ttest_ind(population1, population2)
            if p<0.05 and t>0.0:
                ttest_table.append([val1 if val1>=0 else 'PULSE', val2, p, t])
    return ttest_table


def format_ttest_table(ttest_table):
    result = deepcopy(ttest_table)
    for row in result:
        row[-1] = '$t = %.3f$' % row[-1]
        row[-2] = '$p %s$' % ('\leq 0.001' if row[-2]<=0.001 else ('= %.3f'%row[-2]))
    return result


class ComposeSessionSummary(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    def restore(self):
        pass

    def run(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.experiment

        monopolar_channels = self.get_passed_object('monopolar_channels')
        xval_output = self.get_passed_object('xval_output')

        ps_table = self.get_passed_object('ps_table')
        control_table = self.get_passed_object('control_table')

        sessions = sorted(ps_table.session.unique())

        self.pass_object('NUMBER_OF_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(monopolar_channels))

        thresh = xval_output[-1].jstat_thresh
        control_low_table = control_table[control_table['prob_pre']<thresh]
        control_high_table = control_table[control_table['prob_pre']>1.0-thresh]

        self.pass_object('AUC', xval_output[-1].auc)

        param1_name = param2_name = None
        param1_unit = param2_unit = None
        const_param_name = const_unit = None
        if experiment == 'PS1':
            param1_name = 'Pulse_Frequency'
            param2_name = 'Duration'
            param1_unit = 'Hz'
            param2_unit = 'ms'
            const_param_name = 'Amplitude'
            const_unit = 'mA'
        elif experiment == 'PS2':
            param1_name = 'Pulse_Frequency'
            param2_name = 'Amplitude'
            param1_unit = 'Hz'
            param2_unit = 'mA'
            const_param_name = 'Duration'
            const_unit = 'ms'
        elif experiment == 'PS3':
            param1_name = 'Burst_Frequency'
            param2_name = 'Pulse_Frequency'
            param1_unit = 'Hz'
            param2_unit = 'Hz'
            const_param_name = 'Duration'
            const_unit = 'ms'

        self.pass_object('param1_name', param1_name.replace('_', ' '))
        self.pass_object('param1_unit', param1_unit)

        self.pass_object('param2_name', param2_name.replace('_', ' '))
        self.pass_object('param2_unit', param2_unit)

        self.pass_object('const_param_name', const_param_name)
        self.pass_object('const_unit', const_unit)

        session_data = []
        session_summary_array = []

        anova_param1_sv = dict()
        anova_param2_sv = dict()
        anova_param12_sv = dict()

        anova_significance = dict()

        for session in sessions:
            ps_session_table = ps_table[ps_table.session==session]

            session_summary = SessionSummary()

            session_summary.sess_num = session

            first_time_stamp = ps_session_table.mstime.min()
            last_time_stamp = ps_session_table.mstime.max()
            session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
            session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp/1000))

            session_data.append([session, session_date, session_length])

            session_name = 'Sess%02d' % session

            stim_anode_tag = ps_session_table.stimAnodeTag.values[0].upper()
            stim_cathode_tag = ps_session_table.stimCathodeTag.values[0].upper()
            stim_tag = stim_anode_tag + '-' + stim_cathode_tag
            sess_loc_tag = ps_session_table.Region.values[0]
            roi = '{\em locTag not found}' if sess_loc_tag is None else sess_loc_tag

            isi_min = ps_session_table.isi.min()
            isi_max = ps_session_table.isi.max()
            isi_mid = (isi_max+isi_min) / 2.0
            isi_halfrange = isi_max - isi_mid

            print 'Session =', session_name, ' StimTag =', stim_tag, ' ISI =', isi_mid, '+/-', isi_halfrange

            session_summary.name = session_name
            session_summary.length = session_length
            session_summary.date = session_date
            session_summary.stimtag = stim_tag
            session_summary.region_of_interest = roi
            session_summary.isi_mid = isi_mid
            session_summary.isi_half_range = isi_halfrange
            session_summary.const_param_value = ps_session_table[const_param_name].unique().max()

            ps_session_low_table = pd.DataFrame(ps_session_table[ps_session_table['prob_pre']<thresh])
            ps_session_high_table = pd.DataFrame(ps_session_table[ps_session_table['prob_pre']>1.0-thresh])

            session_summary.low_quantile_classifier_delta_plot = classifier_delta_plot_data(ps_session_low_table, control_low_table['prob_diff_500'], param1_name, param2_name, param2_unit)
            session_summary.low_quantile_recall_delta_plot = recall_delta_plot_data(ps_session_low_table, 'perf_diff_with_control_low', param1_name, param2_name, param2_unit)

            session_summary.high_quantile_classifier_delta_plot = classifier_delta_plot_data(ps_session_high_table, control_high_table['prob_diff_500'], param1_name, param2_name, param2_unit)
            session_summary.high_quantile_recall_delta_plot = recall_delta_plot_data(ps_session_high_table, 'perf_diff_with_control_high', param1_name, param2_name, param2_unit)

            session_summary.all_classifier_delta_plot = classifier_delta_plot_data(ps_session_table, control_table['prob_diff_500'], param1_name, param2_name, param2_unit)
            session_summary.all_recall_delta_plot = recall_delta_plot_data(ps_session_table, 'perf_diff', param1_name, param2_name, param2_unit)

            if sess_loc_tag is not None and not (sess_loc_tag in anova_param1_sv):
                anova_param1_sv[sess_loc_tag] = []
                anova_param2_sv[sess_loc_tag] = []
                anova_param12_sv[sess_loc_tag] = []

            #anova = anova_test(ps_session_low_table, param1_name, param2_name)
            anova = anova_test(ps_session_table, param1_name, param2_name)
            if anova is not None:
                session_summary.anova_fvalues = anova[0]
                session_summary.anova_pvalues = anova[1]

                if anova[1][0] < 0.05: # first param significant
                    if stim_tag in anova_significance:
                        anova_significance[stim_tag][0] = True
                    else:
                        anova_significance[stim_tag] = np.array([True, False, False], dtype=np.bool)
                    param1_ttest_table = ttest_one_param(ps_session_table, param1_name)
                    if len(param1_ttest_table) > 0:
                        if sess_loc_tag is not None:
                            anova_param1_sv[sess_loc_tag].append(param1_ttest_table)
                        session_summary.param1_ttest_table = format_ttest_table(param1_ttest_table)

                if anova[1][1] < 0.05: # second param significant
                    if stim_tag in anova_significance:
                        anova_significance[stim_tag][1] = True
                    else:
                        anova_significance[stim_tag] = np.array([False, True, False], dtype=np.bool)
                    param2_ttest_table = ttest_one_param(ps_session_table, param2_name)
                    if len(param2_ttest_table) > 0:
                        if sess_loc_tag is not None:
                            anova_param2_sv[sess_loc_tag].append(param2_ttest_table)
                        session_summary.param2_ttest_table = format_ttest_table(param2_ttest_table)

                if anova[1][2] < 0.05: # interaction is significant
                    if stim_tag in anova_significance:
                        anova_significance[stim_tag][2] = True
                    else:
                        anova_significance[stim_tag] = np.array([False, False, True], dtype=np.bool)
                    param12_ttest_table = ttest_interaction(ps_session_table, param1_name, param2_name)
                    if len(param12_ttest_table) > 0:
                        if sess_loc_tag is not None:
                            anova_param12_sv[sess_loc_tag].append(param12_ttest_table)
                        session_summary.param12_ttest_table = format_ttest_table(param12_ttest_table)

            session_summary_array.append(session_summary)

        self.pass_object('SESSION_DATA', session_data)
        self.pass_object('session_summary_array', session_summary_array)

        joblib.dump(anova_param1_sv, self.get_path_to_resource_in_workspace(subject + '-' + experiment + '-anova_%s_sv.pkl'%param1_name))
        joblib.dump(anova_param2_sv, self.get_path_to_resource_in_workspace(subject + '-' + experiment + '-anova_%s_sv.pkl'%param2_name))
        joblib.dump(anova_param12_sv, self.get_path_to_resource_in_workspace(subject + '-' + experiment + '-anova_%s-%s_sv.pkl'%(param1_name,param2_name)))

        if len(anova_significance) > 0:
            joblib.dump(anova_significance, self.get_path_to_resource_in_workspace(subject + '-' + experiment + '-anova_significance.pkl'))

        isi_min = ps_table.isi.min()
        isi_max = ps_table.isi.max()
        isi_mid = (isi_max+isi_min) / 2.0
        isi_halfrange = isi_max - isi_mid

        print 'ISI =', isi_mid, '+/-', isi_halfrange

        self.pass_object('CUMULATIVE_ISI_MID', isi_mid)
        self.pass_object('CUMULATIVE_ISI_HALF_RANGE', isi_halfrange)

        ps_low_table = ps_table[ps_table['prob_pre']<thresh]
        ps_high_table = ps_table[ps_table['prob_pre']>1.0-thresh]

        cumulative_low_quantile_classifier_delta_plot = classifier_delta_plot_data(ps_low_table, control_low_table['prob_diff_500'], param1_name, param2_name, param2_unit)
        cumulative_low_quantile_recall_delta_plot = recall_delta_plot_data(ps_low_table, 'perf_diff_with_control_low', param1_name, param2_name, param2_unit)

        cumulative_high_quantile_classifier_delta_plot = classifier_delta_plot_data(ps_high_table, control_high_table['prob_diff_500'], param1_name, param2_name, param2_unit)
        cumulative_high_quantile_recall_delta_plot = recall_delta_plot_data(ps_high_table, 'perf_diff_with_control_high', param1_name, param2_name, param2_unit)

        cumulative_all_classifier_delta_plot = classifier_delta_plot_data(ps_table, control_table['prob_diff_500'], param1_name, param2_name, param2_unit)
        cumulative_all_recall_delta_plot = recall_delta_plot_data(ps_table, 'perf_diff', param1_name, param2_name, param2_unit)

        self.pass_object('cumulative_low_quantile_classifier_delta_plot', cumulative_low_quantile_classifier_delta_plot)
        self.pass_object('cumulative_low_quantile_recall_delta_plot', cumulative_low_quantile_recall_delta_plot)

        self.pass_object('cumulative_high_quantile_classifier_delta_plot', cumulative_high_quantile_classifier_delta_plot)
        self.pass_object('cumulative_high_quantile_recall_delta_plot', cumulative_high_quantile_recall_delta_plot)

        self.pass_object('cumulative_all_classifier_delta_plot', cumulative_all_classifier_delta_plot)
        self.pass_object('cumulative_all_recall_delta_plot', cumulative_all_recall_delta_plot)

        cumulative_anova_fvalues = cumulative_anova_pvalues = None
        cumulative_param1_ttest_table = cumulative_param2_ttest_table = cumulative_param12_ttest_table = None
        anova = anova_test(ps_table, param1_name, param2_name)
        if anova is not None:
            cumulative_anova_fvalues = anova[0]
            cumulative_anova_pvalues = anova[1]
            if anova[1][0] < 0.05:
                param1_ttest_table = ttest_one_param(ps_table, param1_name)
                if len(param1_ttest_table) > 0:
                    cumulative_param1_ttest_table = format_ttest_table(param1_ttest_table)

            if anova[1][1] < 0.05:
                param2_ttest_table = ttest_one_param(ps_table, param2_name)
                if len(param2_ttest_table) > 0:
                    cumulative_param2_ttest_table = format_ttest_table(param2_ttest_table)

            if anova[1][2] < 0.05:
                param12_ttest_table = ttest_interaction(ps_table, param1_name, param2_name)
                if len(param12_ttest_table) > 0:
                    cumulative_param12_ttest_table = format_ttest_table(param12_ttest_table)

        self.pass_object('CUMULATIVE_ANOVA_FVALUES', cumulative_anova_fvalues)
        self.pass_object('CUMULATIVE_ANOVA_PVALUES', cumulative_anova_pvalues)

        self.pass_object('CUMULATIVE_PARAM1_TTEST_TABLE', cumulative_param1_ttest_table)
        self.pass_object('CUMULATIVE_PARAM2_TTEST_TABLE', cumulative_param2_ttest_table)
        self.pass_object('CUMULATIVE_PARAM12_TTEST_TABLE', cumulative_param12_ttest_table)
