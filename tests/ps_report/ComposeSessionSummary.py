__author__ = 'm'

import numpy as np
import pandas as pd
import time

from RamPipeline import *
from SessionSummary import SessionSummary

from PlotUtils import PlotData

from mne.stats import f_mway_rm

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
    param1_vals = sorted(ps_table[param1_name].unique())
    n1 = len(param1_vals)

    param2_vals = sorted(ps_table[param2_name].unique())
    n2 = len(param2_vals)

    table = []
    for val1 in param1_vals:
        ps_table_val1 = ps_table[ps_table[param1_name]==val1]
        for val2 in param2_vals:
            ps_table_val1_val2 = ps_table_val1[ps_table_val1[param2_name]==val2]
            table.append(list(ps_table_val1_val2['prob_diff'].values))

    min_len = min([len(p) for p in table])

    print 'ANOVA test with', min_len, 'samples'

    if min_len < 3:
        return None

    table = [p[:min_len] for p in table]
    table = np.stack(table, axis=1)

    f_vals, p_vals = f_mway_rm(table, [n1,n2])
    return (f_vals, p_vals)


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
            param1_name = 'Pulse Frequency'
            param2_name = 'Duration'
            param1_unit = 'Hz'
            param2_unit = 'ms'
            const_param_name = 'Amplitude'
            const_unit = 'mA'
        elif experiment == 'PS2':
            param1_name = 'Pulse Frequency'
            param2_name = 'Amplitude'
            param1_unit = 'Hz'
            param2_unit = 'mA'
            const_param_name = 'Duration'
            const_unit = 'ms'
        elif experiment == 'PS3':
            param1_name = 'Burst Frequency'
            param2_name = 'Pulse Frequency'
            param1_unit = 'Hz'
            param2_unit = 'Hz'
            const_param_name = 'Duration'
            const_unit = 'ms'

        self.pass_object('CUMULATIVE_PARAMETER1', param1_name)
        self.pass_object('CUMULATIVE_PARAMETER2', param2_name)

        self.pass_object('CUMULATIVE_UNIT1', param1_unit)

        session_data = []
        session_summary_array = []

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
            roi = '{\em not found in bpTalStruct}' if (stim_tag not in loc_tag) or (loc_tag[stim_tag] in ['', '[]']) else loc_tag[stim_tag]

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
            session_summary.parameter1 = param1_name
            session_summary.parameter2 = param2_name
            session_summary.constant_name = const_param_name
            session_summary.constant_value = ps_session_table[const_param_name].unique().max()
            session_summary.constant_unit = const_unit

            anova = anova_test(ps_session_table, param1_name, param2_name)
            if anova is not None:
                session_summary.anova_fvalues = anova[0]
                session_summary.anova_pvalues = anova[1]
                joblib.dump(anova, self.get_path_to_resource_in_workspace(subject + '-' + experiment + '-anova.pkl'))

            session_summary.plots = delta_plot_data(ps_session_table[ps_session_table['prob_pre']<thresh], param1_name, param2_name, param2_unit)

            session_summary_array.append(session_summary)

        self.pass_object('SESSION_DATA', session_data)
        self.pass_object('session_summary_array', session_summary_array)

        isi_min = ps_table.isi.min()
        isi_max = ps_table.isi.max()
        isi_mid = (isi_max+isi_min) / 2.0
        isi_halfrange = isi_max - isi_mid

        print 'ISI =', isi_mid, '+/-', isi_halfrange

        self.pass_object('CUMULATIVE_ISI_MID', isi_mid)
        self.pass_object('CUMULATIVE_ISI_HALF_RANGE', isi_halfrange)

        cumulative_plots = delta_plot_data(ps_table[ps_table['prob_pre']<thresh], param1_name, param2_name, param2_unit)

        self.pass_object('cumulative_plots', cumulative_plots)
