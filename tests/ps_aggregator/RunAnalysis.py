from RamPipeline import *

import numpy as np
import pandas as pd
#import statsmodels.formula.api as smf
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.anova import anova_lm

from PlotUtils import PlotData


class RegionFrequencyAnalysis(object):
    def __init__(self):
        self.ps_table = None
        self.regions = self.areas = None
        self.freqs = None
        self.plots = self.centralized_plots = None
        self.n_experiments = None

    def run(self, ps_table, regions, areas):
        self.ps_table = ps_table
        self.plots = dict()
        self.centralized_plots = dict()
        #areas = sorted(ps_table['Area'].unique())
        self.regions = regions
        self.areas = areas
        self.freqs = sorted(self.ps_table['Pulse_Frequency'].unique())
        #self.n_experiments = np.zeros(shape=(len(self.areas)+len(self.regions),len(self.freqs)), dtype=np.int)
        self.n_experiments = [['','PULSE'] + ['$%d$ Hz' % f for f in self.freqs[1:]]]

        for i,area in enumerate(self.areas):
            if area!='':
                area_sel = (self.ps_table['Area']==area)
                means = np.empty(len(self.freqs), dtype=float)
                sems = np.empty(len(self.freqs), dtype=float)
                centralized_means = np.empty(len(self.freqs), dtype=float)
                centralized_sems = np.empty(len(self.freqs), dtype=float)
                table_line = [area]
                for j,freq in enumerate(self.freqs):
                    freq_sel = (self.ps_table['Pulse_Frequency']==freq)
                    ps_table_area_freq = self.ps_table[area_sel & freq_sel]
                    means[j] = ps_table_area_freq['prob_diff'].mean()
                    sems[j] = ps_table_area_freq['prob_diff'].sem()
                    centralized_means[j] = ps_table_area_freq['prob_diff_centralized'].mean()
                    centralized_sems[j] = ps_table_area_freq['prob_diff_centralized'].sem()
                    table_line.append(len(ps_table_area_freq[['Subject','stimAnodeTag','stimCathodeTag']].drop_duplicates()))
                self.plots[area] = PlotData(x=np.arange(1,len(self.freqs)+1)-i*0.05,
                                y=means, yerr=sems,
                                x_tick_labels=[x if x>0 else 'PULSE' for x in self.freqs],
                                label=area
                                )
                self.centralized_plots[area] = PlotData(x=np.arange(1,len(self.freqs)+1)-i*0.05,
                                y=centralized_means, yerr=centralized_sems,
                                x_tick_labels=[x if x>0 else 'PULSE' for x in self.freqs],
                                label=area
                                )
                self.n_experiments.append(table_line)

        for i,region in enumerate(regions):
            region_sel = (ps_table['Region']==region)
            means = np.empty(len(self.freqs), dtype=float)
            sems = np.empty(len(self.freqs), dtype=float)
            centralized_means = np.empty(len(self.freqs), dtype=float)
            centralized_sems = np.empty(len(self.freqs), dtype=float)
            table_line = [region]
            for j,freq in enumerate(self.freqs):
                freq_sel = (self.ps_table['Pulse_Frequency']==freq)
                ps_table_region_freq = self.ps_table[region_sel & freq_sel]
                means[j] = ps_table_region_freq['prob_diff'].mean()
                sems[j] = ps_table_region_freq['prob_diff'].sem()
                centralized_means[j] = ps_table_region_freq['prob_diff'].mean()
                centralized_sems[j] = ps_table_region_freq['prob_diff'].sem()
                table_line.append(len(ps_table_region_freq[['Subject','stimAnodeTag','stimCathodeTag']].drop_duplicates()))
            self.plots[region] = PlotData(x=np.arange(1,len(self.freqs)+1)-i*0.05,
                            y=means, yerr=sems,
                            x_tick_labels=[x if x>0 else 'PULSE' for x in self.freqs],
                            label=region
                            )
            self.centralized_plots[region] = PlotData(x=np.arange(1,len(self.freqs)+1)-i*0.05,
                            y=centralized_means, yerr=centralized_sems,
                            x_tick_labels=[x if x>0 else 'PULSE' for x in self.freqs],
                            label=region
                            )
            self.n_experiments.append(table_line)


def duration_plot(ps_table, regions, areas):
    plots = dict()
    #areas = sorted(ps_table['Area'].unique())
    #areas = ['HC', 'MTLC']
    #regions = ['CA1', 'DG', 'PRC']
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


def amplitude_plot(ps_table, regions, areas):
    plots = dict()
    #areas = sorted(ps_table['Area'].unique())
    #areas = ['HC', 'MTLC']
    #regions = ['CA1', 'DG', 'PRC']
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
        self.params = params
        self.ps_table = None

    def run(self):
        self.ps_table = self.get_passed_object('ps_table')
        #region_total = self.get_passed_object('region_session_total')
        #regions = [r for r,c in region_total.iteritems() if c>=5]

        rf = RegionFrequencyAnalysis()
        rf.run(self.ps_table, self.params.frequency_plot_regions, self.params.frequency_plot_areas)
        self.pass_object('frequency_plot', rf.plots)
        self.pass_object('n_region_frequency_experiment', rf.n_experiments)

        low_freq_ps_table = self.ps_table[(self.ps_table['Pulse_Frequency']==10) | (self.ps_table['Pulse_Frequency']==25)]
        high_freq_ps_table = self.ps_table[(self.ps_table['Pulse_Frequency']==100) | (self.ps_table['Pulse_Frequency']==200)]

        low_freq_ps1_table = low_freq_ps_table[low_freq_ps_table['Experiment']=='PS1']
        low_freq_duration_plot = duration_plot(low_freq_ps1_table, self.params.duration_plot_regions, self.params.duration_plot_areas)
        self.pass_object('low_freq_duration_plot', low_freq_duration_plot)

        high_freq_ps1_table = high_freq_ps_table[high_freq_ps_table['Experiment']=='PS1']
        high_freq_duration_plot = duration_plot(high_freq_ps1_table, self.params.duration_plot_regions, self.params.duration_plot_areas)
        self.pass_object('high_freq_duration_plot', high_freq_duration_plot)

        low_freq_ps2_table = low_freq_ps_table[low_freq_ps_table['Experiment']=='PS2']
        low_freq_amplitude_plot = amplitude_plot(low_freq_ps2_table, self.params.amplitude_plot_regions, self.params.amplitude_plot_areas)
        self.pass_object('low_freq_amplitude_plot', low_freq_amplitude_plot)

        high_freq_ps2_table = high_freq_ps_table[high_freq_ps_table['Experiment']=='PS2']
        high_freq_amplitude_plot = amplitude_plot(high_freq_ps2_table, self.params.amplitude_plot_regions, self.params.amplitude_plot_areas)
        self.pass_object('high_freq_amplitude_plot', high_freq_amplitude_plot)

        self.run_anova()

    def run_anova(self):
        ps_table_for_anova = self.ps_table[self.ps_table['Area'].isin(self.params.anova_areas)]

        #ps_lm = mixedlm('prob_diff ~ C(Area) * C(Pulse_Frequency)', data=ps_table_for_anova, groups=ps_table_for_anova['Subject']).fit()
        ps_lm = ols('prob_diff ~ C(Area) * C(Pulse_Frequency)', data=ps_table_for_anova).fit()
        anova = anova_lm(ps_lm)
        self.pass_object('fvalue_rf', anova['F'].values[0:3])
        self.pass_object('pvalue_rf', anova['PR(>F)'].values[0:3])

        ps_table_for_anova_low = ps_table_for_anova[ps_table_for_anova['Pulse_Frequency'].isin([10,25])]
        print 'nsamples =', len(ps_table_for_anova_low)

        ps_lm = ols('prob_diff ~ C(Area) * C(Duration)', data=ps_table_for_anova_low).fit()
        anova = anova_lm(ps_lm)
        self.pass_object('fvalue_rd_low', anova['F'].values[0:3])
        self.pass_object('pvalue_rd_low', anova['PR(>F)'].values[0:3])

        ps_lm = ols('prob_diff ~ C(Area) * C(Amplitude)', data=ps_table_for_anova_low).fit()
        anova = anova_lm(ps_lm)
        self.pass_object('fvalue_ra_low', anova['F'].values[0:3])
        self.pass_object('pvalue_ra_low', anova['PR(>F)'].values[0:3])

        ps_table_for_anova_high = ps_table_for_anova[ps_table_for_anova['Pulse_Frequency'].isin([100,200])]
        print 'nsamples =', len(ps_table_for_anova_high)

        ps_lm = ols('prob_diff ~ C(Area) * C(Duration)', data=ps_table_for_anova_high).fit()
        anova = anova_lm(ps_lm)
        self.pass_object('fvalue_rd_high', anova['F'].values[0:3])
        self.pass_object('pvalue_rd_high', anova['PR(>F)'].values[0:3])

        ps_lm = ols('prob_diff ~ C(Area) * C(Amplitude)', data=ps_table_for_anova_high).fit()
        anova = anova_lm(ps_lm)
        self.pass_object('fvalue_ra_high', anova['F'].values[0:3])
        self.pass_object('pvalue_ra_high', anova['PR(>F)'].values[0:3])
