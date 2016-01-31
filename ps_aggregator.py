import sys
from glob import glob

import numpy as np
import pandas as pd

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
        self.param1_vals = self.table1[self.param1_name].unique()
        self.param2_vals = self.table2[self.param2_name].unique()
        self.param1_param2_vals = self.table12[[self.param1_name, self.param2_name]].drop_duplicates()
        self.table12 = self.table12.groupby(['Region', self.param1_name, self.param2_name]).size()
        self.table1 = self.table1.groupby(['Region', self.param1_name]).size()
        self.table2 = self.table2.groupby(['Region', self.param2_name]).size()


ps1_table_files = glob('/scratch/busygin/PS1/R*/*-ps_table.pkl')
ps1 = PS('Pulse_Frequency', 'Duration')
ps1.collect_counts(ps1_table_files)

ps2_table_files = glob('/scratch/busygin/PS2/R*/*-ps_table.pkl')
ps2 = PS('Pulse_Frequency', 'Amplitude')
ps2.collect_counts(ps2_table_files)

ps3_table_files = glob('/scratch/busygin/PS3/R*/*-ps_table.pkl')
ps3 = PS('Burst_Frequency', 'Pulse_Frequency')
ps3.collect_counts(ps3_table_files)
