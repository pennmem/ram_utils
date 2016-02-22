from RamPipeline import *

import numpy as np
import pandas as pd
#import statsmodels.formula.api as smf
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.anova import anova_lm

from PlotUtils import PlotData, BarPlotData


class RegionFrequencyAnalysis(object):
    def __init__(self, output_param):
        self.output_param = output_param
        self.ps_table = None
        self.regions = self.areas = None
        self.freqs = None
        self.plots = self.centralized_plots = None
        self.n_experiments = None
        self.region_plot = None
        self.frequency_plot = None

    def run(self, ps_table, regions, areas):
        self.ps_table = ps_table
        self.plots = dict()
        self.centralized_plots = dict()
        #areas = sorted(ps_table['Area'].unique())
        self.regions = regions
        self.areas = areas
        self.freqs = sorted(self.ps_table['Burst_Frequency'].unique())
        self.n_experiments = [[''] + ['$%d$ Hz' % f for f in self.freqs]]

        area_mean = np.zeros(len(self.areas)+len(self.regions), dtype=float)
        area_sem = np.zeros(len(self.areas)+len(self.regions), dtype=float)
        for i,area in enumerate(self.areas):
            if area!='':
                area_sel = (self.ps_table['Area']==area)
                means = np.empty(len(self.freqs), dtype=float)
                sems = np.empty(len(self.freqs), dtype=float)
                #centralized_means = np.empty(len(self.freqs), dtype=float)
                centralized_sems = np.empty(len(self.freqs), dtype=float)
                table_line = [area]
                for j,freq in enumerate(self.freqs):
                    freq_sel = (self.ps_table['Burst_Frequency']==freq)
                    ps_table_area_freq = self.ps_table[area_sel & freq_sel]
                    means[j] = ps_table_area_freq[self.output_param].mean()
                    sems[j] = ps_table_area_freq[self.output_param].sem()
                    #centralized_means[j] = ps_table_area_freq['prob_diff_centralized'].mean()
                    centralized_sems[j] = ps_table_area_freq[self.output_param].sem()
                    print 'delta sem =', sems[j]-centralized_sems[j]
                    table_line.append(len(ps_table_area_freq[['Subject','stimAnodeTag','stimCathodeTag']].drop_duplicates()))
                self.plots[area] = PlotData(x=np.arange(1,len(self.freqs)+1)-i*0.05,
                                y=means, yerr=sems,
                                x_tick_labels=self.freqs,
                                label=area
                                )
                self.centralized_plots[area] = PlotData(x=np.arange(1,len(self.freqs)+1)-i*0.05,
                                y=means, yerr=centralized_sems,
                                x_tick_labels=self.freqs,
                                label=area
                                )
                self.n_experiments.append(table_line)
                area_mean[i] = self.ps_table[area_sel][self.output_param].mean()
                area_sem[i] = self.ps_table[area_sel][self.output_param].sem()
                #area_sem[i] = self.ps_table[area_sel]['prob_diff_centralized'].sem()

        for i,region in enumerate(regions):
            region_sel = (ps_table['Region']==region)
            means = np.empty(len(self.freqs), dtype=float)
            sems = np.empty(len(self.freqs), dtype=float)
            #centralized_means = np.empty(len(self.freqs), dtype=float)
            centralized_sems = np.empty(len(self.freqs), dtype=float)
            table_line = [region]
            for j,freq in enumerate(self.freqs):
                freq_sel = (self.ps_table['Burst_Frequency']==freq)
                ps_table_region_freq = self.ps_table[region_sel & freq_sel]
                means[j] = ps_table_region_freq[self.output_param].mean()
                sems[j] = ps_table_region_freq[self.output_param].sem()
                #centralized_means[j] = ps_table_region_freq['prob_diff'].mean()
                centralized_sems[j] = ps_table_region_freq[self.output_param].sem()
                table_line.append(len(ps_table_region_freq[['Subject','stimAnodeTag','stimCathodeTag']].drop_duplicates()))
            self.plots[region] = PlotData(x=np.arange(1,len(self.freqs)+1)-i*0.05,
                            y=means, yerr=sems,
                            x_tick_labels=self.freqs,
                            label=region
                            )
            self.centralized_plots[region] = PlotData(x=np.arange(1,len(self.freqs)+1)-i*0.05,
                            y=means, yerr=centralized_sems,
                            x_tick_labels=self.freqs,
                            label=region
                            )
            self.n_experiments.append(table_line)
            area_mean[len(self.areas)+i] = self.ps_table[region_sel][self.output_param].mean()
            area_sem[len(self.areas)+i] = self.ps_table[region_sel][self.output_param].sem()
            #area_sem[len(self.areas)+i] = self.ps_table[region_sel]['prob_diff_centralized'].sem()

        burst_frequency_groups = self.ps_table.groupby('Burst_Frequency')
        freq_mean = burst_frequency_groups[self.output_param].mean().values
        freq_sem = burst_frequency_groups[self.output_param].sem().values
        #freq_sem = burst_frequency_groups['prob_diff_centralized'].sem().values

        area_ymin = np.nanmin(area_mean-area_sem)
        area_ymax = np.nanmax(area_mean+area_sem)

        freq_ymin = np.nanmin(freq_mean-freq_sem)
        freq_ymax = np.nanmax(freq_mean+freq_sem)

        #ylim = (min(0.0,min(area_ymin,freq_ymin)-0.001), max(area_ymax,freq_ymax)+0.001)

        self.region_plot = BarPlotData(x=np.arange(len(self.areas)+len(self.regions)), y=area_mean, yerr=area_sem, ylabel='', xlabel='Region', x_tick_labels=self.areas+self.regions, barcolors=['grey']*(len(self.areas)+len(self.regions)), barwidth=0.5)
        self.frequency_plot = PlotData(x=np.arange(1,len(self.freqs)+1), y=freq_mean, yerr=freq_sem, xlabel='Burst Frequency (Hz)', x_tick_labels=self.freqs)


class RunAnalysis(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.ps_table = None

    def run(self):
        self.ps_table = self.get_passed_object('ps3_table')
        #region_total = self.get_passed_object('region_session_total')
        #regions = [r for r,c in region_total.iteritems() if c>=5]

        rf = RegionFrequencyAnalysis(self.params.output_param)
        rf.run(self.ps_table, self.params.frequency_plot_regions, self.params.frequency_plot_areas)
        self.pass_object('frequency_plot', rf.plots)
        self.pass_object('centralized_frequency_plot', rf.centralized_plots)
        self.pass_object('n_region_frequency_experiment', rf.n_experiments)
        self.pass_object('frequency_frequency_plot', rf.frequency_plot)
        self.pass_object('frequency_region_plot', rf.region_plot)
