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


class BuildAggregatePS3Table(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.ps3_table = None

    def restore(self):
        self.ps3_table = pd.read_pickle(self.get_path_to_resource_in_workspace('ps3_table.pkl'))
        self.pass_object('ps3_table', self.ps3_table)

    def run(self):
        task = self.pipeline.task

        ps3_root = self.get_path_to_resource_in_workspace('PS3/')
        ps3_subjects = sorted([s for s in os.listdir(ps3_root) if s[:2]=='R1'])
        ps3_tables = []
        for subject in ps3_subjects:
            try:
                ps3_table = pd.read_pickle(join(ps3_root, subject, subject+'-PS3-ps_table.pkl'))
                del ps3_table['isi']
                xval_output = control_table = None
                try:
                    xval_output = joblib.load(join(ps3_root, subject, subject+'-'+task+'-xval_output.pkl'))
                    if self.params.baseline_correction:
                        control_table = pd.read_pickle(join(ps3_root, subject, subject+'-'+task+'-control_table.pkl'))
                        baseline_delta = control_table['prob_diff_500'].mean()
                        ps3_table['prob_diff'] -= baseline_delta
                except IOError:
                    xval_output = joblib.load(join(ps3_root, subject, subject+'-xval_output.pkl'))
                    if self.params.baseline_correction:
                        control_table = pd.read_pickle(join(ps3_root, subject, subject+'-control_table.pkl'))
                        baseline_delta = control_table['prob_diff_500'].mean()
                        ps3_table['prob_diff'] -= baseline_delta
                thresh = xval_output[-1].jstat_thresh
                ps3_table = ps3_table[ps3_table['prob_pre']<thresh]
                ps3_table['prob_diff_centralized'] = ps3_table['prob_diff'] - ps3_table['prob_diff'].mean()
                ps3_table['Region'] = ps3_table['Region'].apply(lambda s: 'Undetermined' if s is None else s.replace('Left ','').replace('Right ',''))
                ps3_table['Area'] = ps3_table['Region'].apply(brain_area)
                ps3_table['Subject'] = subject
                ps3_tables.append(ps3_table)
            except IOError:
                pass

        self.ps3_table = pd.concat(ps3_tables, ignore_index=True)

        self.pass_object('ps3_table', self.ps3_table)
        self.ps3_table.to_pickle(self.get_path_to_resource_in_workspace('ps3_table.pkl'))
