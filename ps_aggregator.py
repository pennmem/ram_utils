import sys
from glob import glob

import numpy as np
import pandas as pd
from sklearn.externals import joblib

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('/home1/busygin/ram_utils_new_ptsa')

from PlotUtils import BrickHeatmapPlotData, draw_brick_heatmap


all_freqs = [-999,   10,   25,   50,  100,  200]
all_amps = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
all_durs = [-999, 50, 250, 500, 1000, 1500]
all_burst_freqs = [-999, 3, 4, 5, 6, 7, 8]


class PS(object):
    def __init__(self, param1_name, param2_name):
        self.param1_name = param1_name
        self.param2_name = param2_name
        self.param1_vals = None
        self.param2_vals = None
        self.param1_param2_vals = None
        self.table12 = pd.DataFrame(columns=['Region', param1_name, param2_name])
        self.table1 = pd.DataFrame(columns=['Region', param1_name])
        self.table2 = pd.DataFrame(columns=['Region', param2_name])

    def collect_counts(self, ps_table_files):
        for f in ps_table_files:
            ps_table = pd.read_pickle(f)
            ps_table = ps_table.dropna()
            ps_table = ps_table[ps_table['Region']!=None]
            table12_loc = ps_table[['Region', self.param1_name, self.param2_name]].drop_duplicates()
            table1_loc = table12_loc[['Region', self.param1_name]].drop_duplicates()
            table2_loc = table12_loc[['Region', self.param2_name]].drop_duplicates()
            self.table12 = self.table12.append(table12_loc, ignore_index=True)
            self.table1 = self.table1.append(table1_loc, ignore_index=True)
            self.table2 = self.table2.append(table2_loc, ignore_index=True)
        self.param1_vals = sorted(self.table1[self.param1_name].unique())
        self.param2_vals = sorted(self.table2[self.param2_name].unique())
        self.param1_param2_vals = self.table12[[self.param1_name, self.param2_name]].drop_duplicates()
        self.table12 = self.table12.groupby(['Region', self.param1_name, self.param2_name]).size().to_frame()
        self.table1 = self.table1.groupby(['Region', self.param1_name]).size().to_frame()
        self.table2 = self.table2.groupby(['Region', self.param2_name]).size().to_frame()

    def add_anova_counts(self, anova1_files, anova2_files, anova12_files):
        self.table1['Significant'] = 0
        for f in anova1_files:
            significant_set = set()
            anova_table = joblib.load(f)
            for region, ttest_tables in anova_table.iteritems():
                for ttest_table in ttest_tables:
                    for row in ttest_table:
                        if row[1]<0.05:
                            significant_set.add((region,row[0]))
            for idx in significant_set:
                self.table1['Significant'][idx] += 1

        self.table2['Significant'] = 0
        for f in anova2_files:
            significant_set = set()
            anova_table = joblib.load(f)
            for region, ttest_tables in anova_table.iteritems():
                for ttest_table in ttest_tables:
                    for row in ttest_table:
                        if row[1]<0.05:
                            significant_set.add((region,row[0]))
            for idx in significant_set:
                self.table2['Significant'][idx] += 1

        self.table12['Significant'] = 0
        for f in anova12_files:
            significant_set = set()
            anova_table = joblib.load(f)
            for region, ttest_tables in anova_table.iteritems():
                for ttest_table in ttest_tables:
                    for row in ttest_table:
                        if row[2]<0.05:
                            significant_set.add((region,row[0],row[1]))
            for idx in significant_set:
                self.table12['Significant'][idx] += 1


ps1_table_files = glob('/scratch/busygin/PS1/R*/*-ps_table.pkl')
ps1_anova1_files = glob('/scratch/busygin/PS1/R*/*-anova_Pulse_Frequency_sv.pkl')
ps1_anova2_files = glob('/scratch/busygin/PS1/R*/*-anova_Duration_sv.pkl')
ps1_anova12_files = glob('/scratch/busygin/PS1/R*/*-anova_Pulse_Frequency-Duration_sv.pkl')
ps1 = PS('Pulse_Frequency', 'Duration')
ps1.collect_counts(ps1_table_files)
ps1.add_anova_counts(ps1_anova1_files, ps1_anova2_files, ps1_anova12_files)

ps2_table_files = glob('/scratch/busygin/PS2/R*/*-ps_table.pkl')
ps2_anova1_files = glob('/scratch/busygin/PS2/R*/*-anova_Pulse_Frequency_sv.pkl')
ps2_anova2_files = glob('/scratch/busygin/PS2/R*/*-anova_Amplitude_sv.pkl')
ps2_anova12_files = glob('/scratch/busygin/PS2/R*/*-anova_Pulse_Frequency-Amplitude_sv.pkl')
ps2 = PS('Pulse_Frequency', 'Amplitude')
ps2.collect_counts(ps2_table_files)
ps2.add_anova_counts(ps2_anova1_files, ps2_anova2_files, ps2_anova12_files)

ps3_table_files = glob('/scratch/busygin/PS3/R*/*-ps_table.pkl')
ps3_anova1_files = glob('/scratch/busygin/PS3/R*/*-anova_Burst_Frequency_sv.pkl')
ps3_anova2_files = glob('/scratch/busygin/PS3/R*/*-anova_Pulse_Frequency_sv.pkl')
ps3_anova12_files = glob('/scratch/busygin/PS3/R*/*-anova_Burst_Frequency-Pulse_Frequency_sv.pkl')
ps3 = PS('Burst_Frequency', 'Pulse_Frequency')
ps3.collect_counts(ps3_table_files)
ps3.add_anova_counts(ps3_anova1_files, ps3_anova2_files, ps3_anova12_files)
