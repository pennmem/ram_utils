from RamPipeline import *

import numpy as np
import pandas as pd

from PlotUtils import PlotData


def frequency_plot_data(ps_table, regions):
    plots = dict()
    #areas = sorted(ps_table['Area'].unique())
    areas = ['HC', 'MTLC', 'Frontal']
    regions = ['CA1', 'DG', 'PRC']
    freqs = sorted(ps_table['Pulse_Frequency'].unique())

    for i,area in enumerate(areas):
        if area!='':
            area_sel = (ps_table['Area']==area)
            means = np.empty(len(freqs), dtype=float)
            sems = np.empty(len(freqs), dtype=float)
            for j,freq in enumerate(freqs):
                freq_sel = (ps_table['Pulse_Frequency']==freq)
                ps_table_area_freq = ps_table[area_sel & freq_sel]
                means[j] = ps_table_area_freq['prob_diff'].mean()
                sems[j] = ps_table_area_freq['prob_diff'].sem()
            plots[area] = PlotData(x=np.arange(1,len(freqs)+1)-i*0.05,
                            y=means, yerr=sems,
                            x_tick_labels=[x if x>0 else 'PULSE' for x in freqs],
                            label=area
                            )

    for i,area in enumerate(regions):
        area_sel = (ps_table['Region']==area)
        means = np.empty(len(freqs), dtype=float)
        sems = np.empty(len(freqs), dtype=float)
        for j,freq in enumerate(freqs):
            freq_sel = (ps_table['Pulse_Frequency']==freq)
            ps_table_area_freq = ps_table[area_sel & freq_sel]
            means[j] = ps_table_area_freq['prob_diff'].mean()
            sems[j] = ps_table_area_freq['prob_diff'].sem()
        plots[area] = PlotData(x=np.arange(1,len(freqs)+1)-i*0.05,
                        y=means, yerr=sems,
                        x_tick_labels=[x if x>0 else 'PULSE' for x in freqs],
                        label=area
                        )

    return plots


def duration_plot(ps_table):
    plots = dict()
    #areas = sorted(ps_table['Area'].unique())
    areas = ['HC', 'MTLC']
    regions = ['CA1', 'DG', 'PRC']
    durs = [250, 500, 1000]   #sorted(ps_table['Duration'].unique())

    for i,area in enumerate(areas):
        if area!='':
            area_sel = (ps_table['Area']==area)
            means = np.empty(len(durs), dtype=float)
            sems = np.empty(len(durs), dtype=float)
            for j,dur in enumerate(durs):
                dur_sel = (ps_table['Duration']==dur)
                ps_table_area_dur = ps_table[area_sel & dur_sel]
                means[j] = ps_table_area_dur['prob_diff'].mean()
                sems[j] = ps_table_area_dur['prob_diff'].sem()
            plots[area] = PlotData(x=np.arange(1,len(durs)+1)-i*0.05,
                            y=means, yerr=sems,
                            x_tick_labels=durs,
                            label=area
                            )

    for i,area in enumerate(regions):
        area_sel = (ps_table['Region']==area)
        means = np.empty(len(durs), dtype=float)
        sems = np.empty(len(durs), dtype=float)
        for j,dur in enumerate(durs):
            dur_sel = (ps_table['Duration']==dur)
            ps_table_area_dur = ps_table[area_sel & dur_sel]
            means[j] = ps_table_area_dur['prob_diff'].mean()
            sems[j] = ps_table_area_dur['prob_diff'].sem()
        plots[area] = PlotData(x=np.arange(1,len(durs)+1)-i*0.05,
                        y=means, yerr=sems,
                        x_tick_labels=durs,
                        label=area
                        )

    return plots


def amplitude_plot(ps_table, regions):
    plots = dict()
    #areas = sorted(ps_table['Area'].unique())
    areas = ['HC', 'MTLC']
    regions = ['CA1', 'DG', 'PRC']
    amps = [0.25, 0.5, 0.75]  #sorted(ps_table['Amplitude'].unique())

    for i,area in enumerate(areas):
        if area!='':
            area_sel = (ps_table['Area']==area)
            means = np.empty(len(amps), dtype=float)
            sems = np.empty(len(amps), dtype=float)
            for j,amp in enumerate(amps):
                amp_sel = (ps_table['Amplitude']==amp)
                ps_table_area_amp = ps_table[area_sel & amp_sel]
                means[j] = ps_table_area_amp['prob_diff'].mean()
                sems[j] = ps_table_area_amp['prob_diff'].sem()
            plots[area] = PlotData(x=np.arange(1,len(amps)+1)-i*0.05,
                            y=means, yerr=sems,
                            x_tick_labels=amps,
                            label=area
                            )

    for i,area in enumerate(regions):
        area_sel = (ps_table['Region']==area)
        means = np.empty(len(amps), dtype=float)
        sems = np.empty(len(amps), dtype=float)
        for j,amp in enumerate(amps):
            amp_sel = (ps_table['Amplitude']==amp)
            ps_table_area_amp = ps_table[area_sel & amp_sel]
            means[j] = ps_table_area_amp['prob_diff'].mean()
            sems[j] = ps_table_area_amp['prob_diff'].sem()
        plots[area] = PlotData(x=np.arange(1,len(amps)+1)-i*0.05,
                        y=means, yerr=sems,
                        x_tick_labels=amps,
                        label=area
                        )

    return plots


class RunAnalysis(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.plot_data = None

    def run(self):
        ps_table = self.get_passed_object('ps_table')
        region_total = self.get_passed_object('region_session_total')
        regions = [r for r,c in region_total.iteritems() if c>=5]

        frequency_plot = frequency_plot_data(ps_table, regions)
        self.pass_object('frequency_plot', frequency_plot)

        low_freq_ps_table = ps_table[(ps_table['Pulse_Frequency']==10) | (ps_table['Pulse_Frequency']==25)]
        high_freq_ps_table = ps_table[(ps_table['Pulse_Frequency']==100) | (ps_table['Pulse_Frequency']==200)]

        low_freq_ps1_table = low_freq_ps_table[low_freq_ps_table['Experiment']=='PS1']
        low_freq_duration_plot = duration_plot(low_freq_ps1_table)
        self.pass_object('low_freq_duration_plot', low_freq_duration_plot)

        high_freq_ps1_table = high_freq_ps_table[high_freq_ps_table['Experiment']=='PS1']
        high_freq_duration_plot = duration_plot(high_freq_ps1_table)
        self.pass_object('high_freq_duration_plot', high_freq_duration_plot)

        low_freq_ps2_table = low_freq_ps_table[low_freq_ps_table['Experiment']=='PS2']
        low_freq_amplitude_plot = amplitude_plot(low_freq_ps2_table, regions)
        self.pass_object('low_freq_amplitude_plot', low_freq_amplitude_plot)

        high_freq_ps2_table = high_freq_ps_table[high_freq_ps_table['Experiment']=='PS2']
        high_freq_amplitude_plot = amplitude_plot(high_freq_ps2_table, regions)
        self.pass_object('high_freq_amplitude_plot', high_freq_amplitude_plot)
