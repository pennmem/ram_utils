from RamPipeline import *

import os
from os.path import join

import pandas as pd
from sklearn.externals import joblib


# HC: CA1/CA2/CA2/DG
# MTLC: PRC/PHC
# Frontal: ACg, DLPFC
# Temporal (non-MTLC): STG/TC

def brain_area(region):
    if region in ['CA1', 'CA2', 'CA3', 'DG']:
        return 'HC'
    elif region in ['PHC', 'PRC']:
        return 'MTLC'
    elif region in ['ACg', 'DLPFC']:
        return 'Frontal'
    elif region in ['STG', 'TC']:
        return 'Temporal'
    elif region == 'Undetermined':
        return 'Undetermined'
    else:
        return ''


class BuildAggregatePSTable(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.ps_table = None

    def restore(self):
        self.ps_table = pd.read_pickle(self.get_path_to_resource_in_workspace('ps_table.pkl'))
        self.pass_object('ps_table', self.ps_table)

    def run(self):
        task = self.pipeline.task

        ps1_root = self.get_path_to_resource_in_workspace('PS1/')
        ps1_subjects = sorted([s for s in os.listdir(ps1_root) if s[:2]=='R1'])
        ps1_tables = []
        for subject in ps1_subjects:
            try:
                ps1_table = pd.read_pickle(join(ps1_root, subject, subject+'-PS1-ps_table.pkl'))
                del ps1_table['isi']
                xval_output = control_table = None
                try:
                    xval_output = joblib.load(join(ps1_root, subject, subject+'-'+task+'-xval_output.pkl'))
                    if self.params.baseline_correction:
                        control_table = pd.read_pickle(join(ps1_root, subject, subject+'-'+task+'-control_table.pkl'))
                        baseline_delta = control_table['prob_diff_500'].mean()
                        ps1_table['prob_diff'] -= baseline_delta
                except IOError:
                    xval_output = joblib.load(join(ps1_root, subject, subject+'-xval_output.pkl'))
                    if self.params.baseline_correction:
                        control_table = pd.read_pickle(join(ps1_root, subject, subject+'-control_table.pkl'))
                        baseline_delta = control_table['prob_diff_500'].mean()
                        ps1_table['prob_diff'] -= baseline_delta
                thresh = xval_output[-1].jstat_thresh
                ps1_table = ps1_table[ps1_table['prob_pre']<thresh]
                ps1_table['prob_diff_centralized'] = ps1_table['prob_diff'] - ps1_table['prob_diff'].mean()
                ps1_table['Region'] = ps1_table['Region'].apply(lambda s: 'Undetermined' if s is None else s.replace('Left ','').replace('Right ',''))
                ps1_table['Area'] = ps1_table['Region'].apply(brain_area)
                ps1_table['Subject'] = subject
                ps1_tables.append(ps1_table)
            except IOError:
                pass

        ps1_tables = pd.concat(ps1_tables, ignore_index=True)
        ps1_tables['Experiment'] = 'PS1'

        ps2_root = self.get_path_to_resource_in_workspace('PS2/')
        ps2_subjects = sorted([s for s in os.listdir(ps2_root) if s[:2]=='R1'])
        ps2_tables = []
        for subject in ps2_subjects:
            try:
                ps2_table = pd.read_pickle(join(ps2_root, subject, subject+'-PS2-ps_table.pkl'))
                del ps2_table['isi']
                xval_output = control_table = None
                try:
                    xval_output = joblib.load(join(ps2_root, subject, subject+'-'+task+'-xval_output.pkl'))
                    if self.params.baseline_correction:
                        control_table = pd.read_pickle(join(ps2_root, subject, subject+'-'+task+'-control_table.pkl'))
                        baseline_delta = control_table['prob_diff_500'].mean()
                        ps2_table['prob_diff'] -= baseline_delta
                except IOError:
                    xval_output = joblib.load(join(ps2_root, subject, subject+'-xval_output.pkl'))
                    if self.params.baseline_correction:
                        control_table = pd.read_pickle(join(ps2_root, subject, subject+'-control_table.pkl'))
                        baseline_delta = control_table['prob_diff_500'].mean()
                        ps2_table['prob_diff'] -= baseline_delta
                thresh = xval_output[-1].jstat_thresh
                ps2_table = ps2_table[ps2_table['prob_pre']<thresh]
                ps2_table['prob_diff_centralized'] = ps2_table['prob_diff'] - ps2_table['prob_diff'].mean()
                ps2_table['Region'] = ps2_table['Region'].apply(lambda s: 'Undetermined' if s is None else s.replace('Left ','').replace('Right ',''))
                ps2_table['Area'] = ps2_table['Region'].apply(brain_area)
                ps2_table['Subject'] = subject
                ps2_tables.append(ps2_table)
            except IOError:
                pass

        ps2_tables = pd.concat(ps2_tables, ignore_index=True)
        ps2_tables['Experiment'] = 'PS2'

        self.ps_table = pd.concat([ps1_tables, ps2_tables], ignore_index=True)

        self.pass_object('ps_table', self.ps_table)
        self.ps_table.to_pickle(self.get_path_to_resource_in_workspace('ps_table.pkl'))
