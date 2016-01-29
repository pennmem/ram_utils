__author__ = 'm'

import numpy as np
import pandas as pd
import time

from RamPipeline import *
from SessionSummary import SessionSummary

from PlotUtils import PlotData

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import ttest_ind

from sklearn.externals import joblib


def delta_plot_data(ps_table, param1_name, param2_name, param2_unit):
    plots = dict()
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
        plots[val2] = PlotData(x=np.arange(1,len(param1_vals)+1)-p2*0.1,
                               y=means, yerr=sems,
                               x_tick_labels=[x if x>0 else 'PULSE' for x in param1_vals],
                               label=param2_name+' '+str(val2)+' '+param2_unit
                               )
    return plots


def anova_test(ps_table, param1_name, param2_name):
    if len(ps_table) < 4:
        return None
    ps_lm = ols('prob_diff ~ C(%s) * C(%s)' % (param1_name,param2_name), data=ps_table).fit()
    anova = anova_lm(ps_lm)
    return (anova['F'].values[0:3], anova['PR(>F)'].values[0:3])


# def ttest_one_param(ps_table, param_name):
#     param_vals = sorted(ps_table[param_name].unique())
#     val_max = param_vals[np.argmax([ps_table[ps_table[param_name]==val]['prob_diff'].mean() for val in param_vals])]
#     val_max_sel = (ps_table[param_name]==val_max)
#     population1 = ps_table[val_max_sel]['prob_diff'].values
#     population2 = ps_table[~val_max_sel]['prob_diff'].values
#     t,p = ttest_ind(population1, population2)
#     return val_max,t,p


def ttest_one_param(ps_table, param_name):
    ttest_table = []
    param_vals = sorted(ps_table[param_name].unique())
    for val in param_vals:
        val_sel = (ps_table[param_name]==val)
        population1 = ps_table[val_sel]['prob_diff'].values
        population2 = ps_table[~val_sel]['prob_diff'].values
        t,p = ttest_ind(population1, population2)
        if p<0.05 and t>0.0:
            ttest_table.append([val, p, t])
    return ttest_table


# def ttest_interaction(ps_table, param1_name, param2_name):
#     param1_vals = sorted(ps_table[param1_name].unique())
#     param2_vals = sorted(ps_table[param2_name].unique())
#     mean_max = -1.0
#     val1_max = val2_max = None
#     for val1 in param1_vals:
#         for val2 in param2_vals:
#             ps_table_val1_val2 = ps_table[(ps_table[param1_name]==val1) & (ps_table[param2_name]==val2)]
#             mean = ps_table_val1_val2['prob_diff'].mean()
#             if mean > mean_max:
#                 mean_max = mean
#                 val1_max = val1
#                 val2_max = val2
#
#     val_max_sel = (ps_table[param1_name]==val1_max) & (ps_table[param2_name]==val2_max)
#
#     population1 = ps_table[val_max_sel]['prob_diff'].values
#     population2 = ps_table[~val_max_sel]['prob_diff'].values
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
            population1 = ps_table[sel]['prob_diff'].values
            population2 = ps_table[~sel]['prob_diff'].values
            t,p = ttest_ind(population1, population2)
            if p<0.05 and t>0.0:
                ttest_table.append([val1, val2, p, t])
    return ttest_table


def format_ttest_table(ttest_table):
    for row in ttest_table:
        row[-1] = '$t = %.3f$' % row[-1]
        row[-2] = '$p %s$' % ('\leq 0.001' if row[-2]<=0.001 else ('= %.3f'%row[-2]))
    return ttest_table


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
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        loc_tag = self.get_passed_object('loc_tag')
        xval_output = self.get_passed_object('xval_output')

        ps_table = self.get_passed_object('ps_table')

        sessions = sorted(ps_table.session.unique())

        self.pass_object('NUMBER_OF_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(monopolar_channels))

        thresh = xval_output[-1].jstat_thresh

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
            sess_loc_tag = None if (stim_tag not in loc_tag) or (loc_tag[stim_tag] in ['', '[]']) else loc_tag[stim_tag]
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

            session_summary.plots = delta_plot_data(ps_session_low_table, param1_name, param2_name, param2_unit)

            anova = anova_test(ps_session_low_table, param1_name, param2_name)
            if anova is not None:
                session_summary.anova_fvalues = anova[0]
                session_summary.anova_pvalues = anova[1]

                if anova[1][0] < 0.06: # first param significant
                    param1_ttest_table = ttest_one_param(ps_session_low_table, param1_name)
                    if sess_loc_tag is not None:
                        if sess_loc_tag in anova_param1_sv:
                            anova_param1_sv[sess_loc_tag].append(param1_ttest_table)
                        else:
                            anova_param1_sv[sess_loc_tag] = [param1_ttest_table]
                    session_summary.param1_ttest_table = format_ttest_table(param1_ttest_table)

                if anova[1][1] < 0.06: # second param significant
                    param2_ttest_table = ttest_one_param(ps_session_low_table, param2_name)
                    if sess_loc_tag is not None:
                        if sess_loc_tag in anova_param2_sv:
                            anova_param2_sv[sess_loc_tag].append(param2_ttest_table)
                        else:
                            anova_param2_sv[sess_loc_tag] = [param2_ttest_table]
                    session_summary.param2_ttest_table = format_ttest_table(param2_ttest_table)

                if anova[1][2] < 0.06: # interaction is significant
                    param12_ttest_table = ttest_interaction(ps_session_low_table, param1_name, param2_name)
                    if sess_loc_tag is not None:
                        if sess_loc_tag in anova_param12_sv:
                            anova_param12_sv[sess_loc_tag].append(param12_ttest_table)
                        else:
                            anova_param12_sv[sess_loc_tag] = [param12_ttest_table]
                    session_summary.param12_ttest_table = format_ttest_table(param12_ttest_table)

            session_summary_array.append(session_summary)

        self.pass_object('SESSION_DATA', session_data)
        self.pass_object('session_summary_array', session_summary_array)

        joblib.dump(anova_param1_sv, self.get_path_to_resource_in_workspace(subject + '-' + experiment + '-anova_%s_sv.pkl'%param1_name))
        joblib.dump(anova_param2_sv, self.get_path_to_resource_in_workspace(subject + '-' + experiment + '-anova_%s_sv.pkl'%param2_name))
        joblib.dump(anova_param12_sv, self.get_path_to_resource_in_workspace(subject + '-' + experiment + '-anova_%s-%s_sv.pkl'%(param1_name,param2_name)))

        isi_min = ps_table.isi.min()
        isi_max = ps_table.isi.max()
        isi_mid = (isi_max+isi_min) / 2.0
        isi_halfrange = isi_max - isi_mid

        print 'ISI =', isi_mid, '+/-', isi_halfrange

        self.pass_object('CUMULATIVE_ISI_MID', isi_mid)
        self.pass_object('CUMULATIVE_ISI_HALF_RANGE', isi_halfrange)

        cumulative_plots = delta_plot_data(ps_table[ps_table['prob_pre']<thresh], param1_name, param2_name, param2_unit)

        self.pass_object('cumulative_plots', cumulative_plots)
