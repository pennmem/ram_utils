from RamPipeline import *

import os
from os.path import join

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from bisect import bisect_right


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


def prob2perf(probs, true_labels, p):
    idx = bisect_right(probs, p)
    return np.sum(true_labels[0:idx]) / float(idx) if idx>0 else 0.0


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
                xval_output = None
                try:
                    xval_output = joblib.load(join(ps3_root, subject, subject+'-xval_output.pkl'))
                except IOError:
                    xval_output = joblib.load(join(ps3_root, subject, subject+'-'+task+'-xval_output.pkl'))
                thresh = xval_output[-1].jstat_thresh
                ps3_table['thresh'] = thresh
                ps3_table['Region'] = ps3_table['Region'].apply(lambda s: 'Undetermined' if s is None else s.replace('Left ','').replace('Right ',''))
                ps3_table['Area'] = ps3_table['Region'].apply(brain_area)
                ps3_table['Subject'] = subject
                ps3_tables.append(ps3_table)
            except IOError:
                pass

        self.ps3_table = pd.concat(ps3_tables, ignore_index=True)

        self.pass_object('ps3_table', self.ps3_table)
        self.ps3_table.to_pickle(self.get_path_to_resource_in_workspace('ps3_table.pkl'))
