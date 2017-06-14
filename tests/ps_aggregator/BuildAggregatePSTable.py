import os
from os.path import join

import pandas as pd

from ram_utils.RamPipeline import *


def brain_area(region):
    if region in ['CA1', 'CA2', 'CA3', 'DG', 'Sub']:
        return 'HC'
    elif region in ['PHC', 'PRC', 'EC']:
        return 'MTLC'
    elif region in ['ACg', 'PCg', 'DLPFC']:
        return 'Cing-PFC'
    #elif region in ['STG', 'TC']:
    #    return 'Temporal'
    elif region == 'Undetermined':
        return 'Undetermined'
    else:
        return 'Other'


class BuildAggregatePSTable(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.ps_table = None

    def restore(self):
        self.ps_table = pd.read_pickle(self.get_path_to_resource_in_workspace('ps_table.pkl'))
        self.pass_object('ps_table', self.ps_table)

    def run(self):
        # task = self.pipeline.task

        ps1_root = self.get_path_to_resource_in_workspace('PS1_reports/')
        ps1_subjects = sorted([s for s in os.listdir(ps1_root) if s[:2]=='R1'])
        ps1_tables = []
        for subject in ps1_subjects:
            try:
                ps1_table = pd.read_pickle(join(ps1_root, subject, subject+'-PS1-ps_table.pkl'))
                del ps1_table['isi']
                # xval_output = None
                # try:
                #     xval_output = joblib.load(join(ps1_root, subject, subject+'-xval_output.pkl'))
                # except IOError:
                #     xval_output = joblib.load(join(ps1_root, subject, subject+'-'+task+'-xval_output.pkl'))
                # thresh = xval_output[-1].jstat_thresh
                # ps1_table['thresh'] = thresh
                ps1_table['locTag'] = ps1_table['Region'].apply(lambda s: 'Undetermined' if s is None else s)
                ps1_table['Region'] = ps1_table['Region'].apply(lambda s: 'Undetermined' if s is None else s.replace('Left ','').replace('Right ',''))
                ps1_table['Area'] = ps1_table['Region'].apply(brain_area)
                ps1_table['Subject'] = subject
                ps1_tables.append(ps1_table)
            except IOError:
                pass

        ps1_tables = pd.concat(ps1_tables, ignore_index=True)
        ps1_tables['Experiment'] = 'PS1'

        ps2_root = self.get_path_to_resource_in_workspace('PS2_reports/')
        ps2_subjects = sorted([s for s in os.listdir(ps2_root) if s[:2]=='R1'])
        ps2_tables = []
        for subject in ps2_subjects:
            try:
                ps2_table = pd.read_pickle(join(ps2_root, subject, subject+'-PS2-ps_table.pkl'))
                del ps2_table['isi']
                # xval_output = None
                # try:
                #     xval_output = joblib.load(join(ps2_root, subject, subject+'-xval_output.pkl'))
                # except IOError:
                #     xval_output = joblib.load(join(ps2_root, subject, subject+'-'+task+'-xval_output.pkl'))
                # thresh = xval_output[-1].jstat_thresh
                # ps2_table['thresh'] = thresh
                ps2_table['locTag'] = ps2_table['Region'].apply(lambda s: 'Undetermined' if s is None else s)
                ps2_table['Region'] = ps2_table['Region'].apply(lambda s: 'Undetermined' if s is None else s.replace('Left ','').replace('Right ',''))
                ps2_table['Area'] = ps2_table['Region'].apply(brain_area)
                ps2_table['Subject'] = subject
                ps2_tables.append(ps2_table)
            except IOError:
                pass

        ps2_tables = pd.concat(ps2_tables, ignore_index=True)
        ps2_tables['Experiment'] = 'PS2'

        ps2_1_root = self.get_path_to_resource_in_workspace('PS2.1_reports/')
        ps2_1_subjects = sorted([s for s in os.listdir(ps2_1_root) if s[:2]=='R1'])
        ps2_1_tables = []
        for subject in ps2_1_subjects:
            try:
                ps2_1_table = pd.read_pickle(join(ps2_1_root, subject, subject+'-PS2.1-ps_table.pkl'))
                del ps2_1_table['isi']
                # xval_output = None
                # try:
                #     xval_output = joblib.load(join(ps2_root, subject, subject+'-xval_output.pkl'))
                # except IOError:
                #     xval_output = joblib.load(join(ps2_root, subject, subject+'-'+task+'-xval_output.pkl'))
                # thresh = xval_output[-1].jstat_thresh
                # ps2_table['thresh'] = thresh
                ps2_1_table['locTag'] = ps2_1_table['Region'].apply(lambda s: 'Undetermined' if s is None else s)
                ps2_1_table['Region'] = ps2_1_table['Region'].apply(lambda s: 'Undetermined' if s is None else s.replace('Left ','').replace('Right ',''))
                ps2_1_table['Area'] = ps2_1_table['Region'].apply(brain_area)
                ps2_1_table['Subject'] = subject
                ps2_1_tables.append(ps2_1_table)
            except IOError:
                pass

        ps2_1_tables = pd.concat(ps2_1_tables, ignore_index=True)
        ps2_1_tables['Experiment'] = 'PS2.1'

        self.ps_table = pd.concat([ps1_tables, ps2_tables, ps2_1_tables], ignore_index=True)

        self.pass_object('ps_table', self.ps_table)
        self.ps_table.to_pickle(self.get_path_to_resource_in_workspace('ps_table.pkl'))
