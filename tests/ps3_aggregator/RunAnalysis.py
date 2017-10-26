from collections import OrderedDict

import numpy as np

from PlotUtils import PlotData, BarPlotData
from ramutils.pipeline import RamTask


class RegionFrequencyAnalysis(object):
    def __init__(self, output_param):
        self.output_param = output_param
        self.ps_table = None
        self.regions = self.areas = None
        self.freqs = None
        self.plots = None
        self.n_experiments = None
        self.region_plot = None
        self.frequency_plot = None

    def run(self, ps_table, regions, areas):
        self.ps_table = ps_table
        self.plots = OrderedDict()
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
                table_line = [area]
                for j,freq in enumerate(self.freqs):
                    freq_sel = (self.ps_table['Burst_Frequency']==freq)
                    ps_table_area_freq = self.ps_table[area_sel & freq_sel]
                    means[j] = ps_table_area_freq[self.output_param].mean()
                    sems[j] = ps_table_area_freq[self.output_param].sem()
                    table_line.append(len(ps_table_area_freq[['Subject','stimAnodeTag','stimCathodeTag']].drop_duplicates()))
                self.plots[area] = PlotData(x=np.arange(1,len(self.freqs)+1)-i*0.05,
                                y=means, yerr=sems,
                                x_tick_labels=self.freqs,
                                label=area
                                )
                self.n_experiments.append(table_line)
                area_mean[i] = self.ps_table[area_sel][self.output_param].mean()
                area_sem[i] = self.ps_table[area_sel][self.output_param].sem()

        for i,region in enumerate(regions):
            region_sel = (ps_table['Region']==region)
            means = np.empty(len(self.freqs), dtype=float)
            sems = np.empty(len(self.freqs), dtype=float)
            table_line = [region]
            for j,freq in enumerate(self.freqs):
                freq_sel = (self.ps_table['Burst_Frequency']==freq)
                ps_table_region_freq = self.ps_table[region_sel & freq_sel]
                means[j] = ps_table_region_freq[self.output_param].mean()
                sems[j] = ps_table_region_freq[self.output_param].sem()
                table_line.append(len(ps_table_region_freq[['Subject','stimAnodeTag','stimCathodeTag']].drop_duplicates()))
            self.plots[region] = PlotData(x=np.arange(1,len(self.freqs)+1)-i*0.05,
                            y=means, yerr=sems,
                            x_tick_labels=self.freqs,
                            label=region
                            )
            self.n_experiments.append(table_line)
            area_mean[len(self.areas)+i] = self.ps_table[region_sel][self.output_param].mean()
            area_sem[len(self.areas)+i] = self.ps_table[region_sel][self.output_param].sem()

        burst_frequency_groups = self.ps_table.groupby('Burst_Frequency')
        freq_mean = burst_frequency_groups[self.output_param].mean().values
        freq_sem = burst_frequency_groups[self.output_param].sem().values

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

        ps_table_100 = self.ps_table[self.ps_table['Pulse_Frequency']==100]
        self.analyze(ps_table_100, output_param='prob_diff', name_prefix='all_100_', name_suffix='')
        self.analyze(ps_table_100, output_param='perf_diff', name_prefix='all_100_', name_suffix='')

        ps_table_200 = self.ps_table[self.ps_table['Pulse_Frequency']==200]
        self.analyze(ps_table_200, output_param='prob_diff', name_prefix='all_200_', name_suffix='')
        self.analyze(ps_table_200, output_param='perf_diff', name_prefix='all_200_', name_suffix='')

        self.analyze(self.ps_table, output_param='prob_diff', name_prefix='all_', name_suffix='')
        self.analyze(self.ps_table, output_param='perf_diff', name_prefix='all_', name_suffix='')

    def analyze(self, ps_subtable, output_param, name_prefix, name_suffix):
        rf = RegionFrequencyAnalysis(output_param)
        rf.run(ps_subtable, self.params.frequency_plot_regions, self.params.frequency_plot_areas)
        self.pass_object(name_prefix+output_param+'_frequency_plot', rf.plots)
        self.pass_object('n_region_frequency_experiment', rf.n_experiments)
        self.pass_object(name_prefix+output_param+'_frequency_frequency_plot', rf.frequency_plot)
        self.pass_object(name_prefix+output_param+'_frequency_region_plot', rf.region_plot)
