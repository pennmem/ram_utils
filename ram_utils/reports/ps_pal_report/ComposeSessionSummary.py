__author__ = 'm'

import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from ...ReportUtils import ReportRamTask
from scipy.stats import ttest_ind, ttest_1samp
from sklearn.externals import joblib
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from .SessionSummary import SessionSummary
from ram_utils.PlotUtils import PlotData


def plot_data(ps_table, delta_column_name, ps_sham, param1_name, param2_name, param2_unit):
    x_start_pos = 2 if len(ps_sham)>0 else 1
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
        plots[val2] = PlotData(x=np.arange(x_start_pos,len(param1_vals)+x_start_pos)-p2*0.1,
                               y=means, yerr=sems, x_tick_labels=[x if x>0 else 'PULSE' for x in param1_vals],
                               label=param2_name+' '+str(val2)+' '+param2_unit
                               )
    if len(ps_sham)>0:
        sham_means = np.empty(len(param1_vals)+1, dtype=float)
        sham_mean = ps_sham.mean()
        sham_means[0] = sham_mean
        sham_means[1:] = np.NAN
        sham_sems = np.empty(len(param1_vals)+1, dtype=float)
        sham_sems[0] = ps_sham.sem()
        sham_sems[1:] = np.NAN
        plots['SHAM'] = PlotData(x=np.arange(1,len(param1_vals)+2), y=sham_means, yerr=sham_sems, x_tick_labels=['SHAM']+[x if x>0 else 'PULSE' for x in param1_vals], color='k', markersize=10.0, elinewidth=3.0)
    return plots


def anova_test(ps_table, param1_name, param2_name):
    if len(ps_table) < 10:
        return None
    ps_lm = ols('perf_diff ~ C(%s) * C(%s)' % (param1_name,param2_name), data=ps_table).fit()
    anova = anova_lm(ps_lm)
    return (anova['F'].values[0:3], anova['PR(>F)'].values[0:3])


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
        row[-1] = '$%.3f$' % row[-1]
        row[-2] = '$\leq 0.001$' if row[-2]<=0.001 else ('$%.3f$'%row[-2])
    return result


def ttest_against_zero(ps_table, param1_name, param2_name):
    ttest_table = []
    param1_vals = sorted(ps_table[param1_name].unique())
    param2_vals = sorted(ps_table[param2_name].unique())
    for val1 in param1_vals:
        val1_sel = (ps_table[param1_name]==val1)
        for val2 in param2_vals:
            val2_sel = (ps_table[param2_name]==val2)
            sel = val1_sel & val2_sel
            population = ps_table[sel]['perf_diff'].values
            t,p = ttest_1samp(population, 0.0)
            if p<0.05 and t>0.0:
                ttest_table.append([val1 if val1>=0 else 'PULSE', val2, p, t])
    return ttest_table


def ttest_against_sham(ps_table, ps_sham_table, param1_name, param2_name):
    sham = ps_sham_table['perf_diff'].values
    ttest_table = []
    param1_vals = sorted(ps_table[param1_name].unique())
    param2_vals = sorted(ps_table[param2_name].unique())
    for val1 in param1_vals:
        val1_sel = (ps_table[param1_name]==val1)
        for val2 in param2_vals:
            val2_sel = (ps_table[param2_name]==val2)
            sel = val1_sel & val2_sel
            population = ps_table[sel]['perf_diff'].values
            t,p = ttest_ind(population, sham)
            if p<0.05 and t>0.0:
                ttest_table.append([val1 if val1>=0 else 'PULSE', val2, p, t])
    return ttest_table


class ComposeSessionSummary(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComposeSessionSummary,self).__init__(mark_as_completed)
        self.params = params

    def restore(self):
        pass

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        monopolar_channels = self.get_passed_object('monopolar_channels')
        xval_output = self.get_passed_object('xval_output')
        thresh = xval_output[-1].jstat_thresh

        ps_table = self.get_passed_object('ps_table')
        ps_sham_table = self.get_passed_object('control_table')

        sessions = sorted(ps_table.session.unique())

        self.pass_object('NUMBER_OF_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(monopolar_channels))

        self.pass_object('AUC', xval_output[-1].auc)

        param1_name = param2_name = None
        param1_unit = param2_unit = None
        const_param_name = const_unit = None
        if task == 'PS1':
            param1_name = 'Pulse_Frequency'
            param2_name = 'Duration'
            param1_unit = 'Hz'
            param2_unit = 'ms'
            const_param_name = 'Amplitude'
            const_unit = 'mA'
        elif task in ['PS2', 'PS2.1']:
            param1_name = 'Pulse_Frequency'
            param2_name = 'Amplitude'
            param1_unit = 'Hz'
            param2_unit = 'mA'
            const_param_name = 'Duration'
            const_unit = 'ms'
        elif task == 'PS3':
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

        ps_table_by_session = ps_table.groupby(['session'])
        for session,ps_session_table in ps_table_by_session:
            first_time_stamp = ps_session_table.mstime.min()
            last_time_stamp = ps_session_table.mstime.max()
            session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
            session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp/1000))

            session_data.append([session, session_date, session_length])

        ps_table_by_bipolar_pair = ps_table.groupby(['stimAnodeTag','stimCathodeTag'])
        for bipolar_pair,ps_session_table in ps_table_by_bipolar_pair:
            ps_session_table_low = ps_session_table[ps_session_table['prob_pre']<thresh]

            ps_session_sham_table = ps_sham_table[(ps_sham_table.stimAnodeTag==bipolar_pair[0]) & (ps_sham_table.stimCathodeTag==bipolar_pair[1])]
            ps_session_sham_table_low = ps_session_sham_table[ps_session_sham_table['prob_pre']<thresh]

            session_summary = SessionSummary()

            session_summary.sessions = ps_session_table.session.unique().__str__()[1:-1]

            stim_anode_tag = bipolar_pair[0].upper()
            stim_cathode_tag = bipolar_pair[1].upper()
            stim_tag = stim_anode_tag + '-' + stim_cathode_tag
            sess_loc_tag = ps_session_table.Region.values[0]
            roi = '{\em locTag not found}' if sess_loc_tag is None else sess_loc_tag

            isi_min = ps_session_table.isi.min()
            isi_max = ps_session_table.isi.max()
            isi_mid = (isi_max+isi_min) / 2.0
            isi_halfrange = isi_max - isi_mid

            print ' StimTag =', stim_tag, ' ISI =', isi_mid, '+/-', isi_halfrange

            session_summary.stimtag = stim_tag
            session_summary.region_of_interest = roi
            session_summary.isi_mid = isi_mid
            session_summary.isi_half_range = isi_halfrange
            session_summary.const_param_value = ps_session_table[const_param_name].unique().max()

            session_summary.low_classifier_delta_plot = plot_data(ps_session_table_low, 'prob_diff', ps_session_sham_table_low['prob_diff'], param1_name, param2_name, param2_unit)
            session_summary.low_recall_delta_plot = plot_data(ps_session_table_low, 'perf_diff', ps_session_sham_table_low['perf_diff'], param1_name, param2_name, param2_unit)

            session_summary.all_classifier_delta_plot = plot_data(ps_session_table, 'prob_diff', ps_session_sham_table['prob_diff'], param1_name, param2_name, param2_unit)
            session_summary.all_recall_delta_plot = plot_data(ps_session_table, 'perf_diff', ps_session_sham_table['perf_diff'], param1_name, param2_name, param2_unit)

            if sess_loc_tag is not None and not (sess_loc_tag in anova_param1_sv):
                anova_param1_sv[sess_loc_tag] = []
                anova_param2_sv[sess_loc_tag] = []
                anova_param12_sv[sess_loc_tag] = []

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

            ttest_against_zero_table = ttest_against_zero(ps_session_table, param1_name, param2_name)
            session_summary.ttest_against_zero_table = format_ttest_table(ttest_against_zero_table)

            if len(ps_session_sham_table_low) > 0:
                ttest_against_sham_table = ttest_against_sham(ps_session_table_low, ps_session_sham_table_low, param1_name, param2_name)
                session_summary.ttest_against_sham_table = format_ttest_table(ttest_against_sham_table)

            session_summary_array.append(session_summary)

        self.pass_object('SESSION_DATA', session_data)
        self.pass_object('session_summary_array', session_summary_array)

        joblib.dump(anova_param1_sv, self.get_path_to_resource_in_workspace(subject + '-' + task + '-anova_%s_sv.pkl'%param1_name))
        joblib.dump(anova_param2_sv, self.get_path_to_resource_in_workspace(subject + '-' + task + '-anova_%s_sv.pkl'%param2_name))
        joblib.dump(anova_param12_sv, self.get_path_to_resource_in_workspace(subject + '-' + task + '-anova_%s-%s_sv.pkl'%(param1_name,param2_name)))

        if len(anova_significance) > 0:
            joblib.dump(anova_significance, self.get_path_to_resource_in_workspace(subject + '-' + task + '-anova_significance.pkl'))
