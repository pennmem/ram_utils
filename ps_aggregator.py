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


def duration_plot(ps1_dur_table):
    regions = []
    durs = []

    table = ps1_dur_table[ps1_dur_table['Significant']>0]
    for c in table.index:
        regions.append(c[0])
        durs.append(c[1])

    regions = np.unique(regions)
    n_regions = len(regions)

    durs = np.unique(durs)
    n_durs = len(durs)

    plot_data = np.zeros(shape=(n_regions,n_durs), dtype=float)
    text = dict()

    for i,reg in enumerate(regions):
        for j,dur in enumerate(durs):
            t = 0
            s = 0
            if (reg,dur) in table.index:
                t = table['Total'][(reg,dur)]
                s = table['Significant'][(reg,dur)]
            if t>0:
                plot_data[i,j] = s / float(t)
                text[(i,j)] = '%d/%d' % (s,t)

    return BrickHeatmapPlotData(df=plot_data, annot_dict=text, title='PS1 Aggregate Report for Duration', x_tick_labels=durs, y_tick_labels=regions, xlabel='Duration (ms)', ylabel='Region')


def amplitude_plot(ps2_amp_table):
    regions = []
    amps = []

    table = ps2_amp_table[ps2_amp_table['Significant']>0]
    for c in table.index:
        regions.append(c[0])
        amps.append(c[1])

    regions = np.unique(regions)
    n_regions = len(regions)

    amps = np.unique(amps)
    n_amps = len(amps)

    plot_data = np.zeros(shape=(n_regions,n_amps), dtype=float)
    text = dict()

    for i,reg in enumerate(regions):
        for j,amp in enumerate(amps):
            t = 0
            s = 0
            if (reg,amp) in table.index:
                t = table['Total'][(reg,amp)]
                s = table['Significant'][(reg,amp)]
            if t>0:
                plot_data[i,j] = s / float(t)
                text[(i,j)] = '%d/%d' % (s,t)

    return BrickHeatmapPlotData(df=plot_data, annot_dict=text, title='PS2 Aggregate Report for Amplitude', x_tick_labels=amps, y_tick_labels=regions, xlabel='Amplitude (mA)', ylabel='Region')


def ps3_frequency_plot(ps3_freq_table):
    regions = []
    freqs = []

    table = ps3_freq_table[ps3_freq_table['Significant']>0]
    for c in table.index:
        regions.append(c[0])
        freqs.append(c[1])

    regions = np.unique(regions)
    n_regions = len(regions)

    freqs = np.unique(freqs)
    n_freqs = len(freqs)

    plot_data = np.zeros(shape=(n_regions,n_freqs), dtype=float)
    text = dict()

    for i,reg in enumerate(regions):
        for j,freq in enumerate(freqs):
            t = 0
            s = 0
            if (reg,freq) in table.index:
                t = table['Total'][(reg,freq)]
                s = table['Significant'][(reg,freq)]
            if t>0:
                plot_data[i,j] = s / float(t)
                text[(i,j)] = '%d/%d' % (s,t)

    return BrickHeatmapPlotData(df=plot_data, annot_dict=text, title='PS3 Aggregate Report for Pulse Frequency', x_tick_labels=freqs, y_tick_labels=regions, xlabel='Pulse Frequency (Hz)', ylabel='Region')


def frequency_plot(ps1_freq_table, ps2_freq_table, ps3_burst_freq_table):
    regions = []
    freqs = []

    table1 = ps1_freq_table[ps1_freq_table['Significant']>0]
    for c in table1.index:
        regions.append(c[0])
        freqs.append(c[1])

    table2 = ps2_freq_table[ps2_freq_table['Significant']>0]
    for c in table2.index:
        regions.append(c[0])
        freqs.append(c[1])

    table3 = ps3_burst_freq_table[ps3_burst_freq_table['Significant']>0]
    for c in table3.index:
        regions.append(c[0])
        freqs.append(c[1])

    regions = np.unique(regions)
    n_regions = len(regions)

    freqs = np.unique(freqs)
    n_freqs = len(freqs)

    plot_data = np.zeros(shape=(n_regions,n_freqs), dtype=float)
    text = dict()

    for i,reg in enumerate(regions):
        for j,freq in enumerate(freqs):
            t = 0
            s = 0
            if freq>=10:
                if (reg,freq) in table1.index:
                    t += table1['Total'][(reg,freq)]
                    s += table1['Significant'][(reg,freq)]
                if (reg,freq) in table2.index:
                    t += table2['Total'][(reg,freq)]
                    s += table2['Significant'][(reg,freq)]
            elif freq>0:
                if (reg,freq) in table3.index:
                    t += table3['Total'][(reg,freq)]
                    s += table3['Significant'][(reg,freq)]
            else:
                if (reg,freq) in table2.index:
                    t += table2['Total'][(reg,freq)]
                    s += table2['Significant'][(reg,freq)]
            if t>0:
                plot_data[i,j] = s / float(t)
                text[(i,j)] = '%d/%d' % (s,t)

    x_tick_labels = [x if x>0 else 'PULSE' for x in freqs]
    return BrickHeatmapPlotData(df=plot_data, annot_dict=text, title='PS1,PS2,PS3 Aggregate Report for Pulse/Burst Frequency', x_tick_labels=x_tick_labels, y_tick_labels=regions, xlabel='Pulse or Burst Frequency (Hz)', ylabel='Region')


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
        self.table12 = self.table12.groupby(['Region', self.param1_name, self.param2_name]).size().to_frame(name='Total')
        self.table1 = self.table1.groupby(['Region', self.param1_name]).size().to_frame(name='Total')
        self.table2 = self.table2.groupby(['Region', self.param2_name]).size().to_frame(name='Total')

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

with PdfPages('/scratch/busygin/PS Aggregate Report.pdf') as pdf:
    plot = frequency_plot(ps1.table1, ps2.table1, ps3.table1)
    fig,ax = draw_brick_heatmap(plot)
    pdf.savefig()
    plt.clf()

    plot = ps3_frequency_plot(ps3.table2)
    fig,ax = draw_brick_heatmap(plot)
    pdf.savefig()
    plt.clf()

    plot = amplitude_plot(ps2.table2)
    fig,ax = draw_brick_heatmap(plot)
    pdf.savefig()
    plt.clf()

    plot = duration_plot(ps1.table2)
    fig,ax = draw_brick_heatmap(plot)
    pdf.savefig()
    plt.clf()
