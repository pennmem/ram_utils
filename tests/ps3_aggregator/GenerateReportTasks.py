__author__ = 'm'

from RamPipeline import *

from PlotUtils import PlotData, BarPlotData, PlotDataCollection, PanelPlot
import TextTemplateUtils

import datetime
import numpy as np

from latex_table import latex_table


def pvalue_formatting(p):
    return '\leq 0.001' if p<=0.001 else ('%.3f'%p)


class GenerateTex(RamTask):
    def __init__(self, mark_as_completed=True): RamTask.__init__(self, mark_as_completed)

    def run(self):
        tex_template = 'ps3_aggregator.tex.tpl'

        report_tex_file_name = 'ps3_aggregator.tex'
        self.pass_object('report_tex_file_name',report_tex_file_name)

        self.set_file_resources_to_move(report_tex_file_name, dst='reports')

        region_total = self.get_passed_object('region_session_total')
        area_total = self.get_passed_object('area_session_total')

        region_total = sorted(zip(region_total.keys(), region_total.values()))
        region_total.append(('Total',sum([v[1] for v in region_total])))

        area_total = sorted(zip(area_total.keys(), area_total.values()))

        replace_dict = {
            '<DATE>': datetime.date.today(),
            '<LOW_QUANTILE_FREQUENCY_PLOT_FILE>': 'ps3_low_quantile_frequency_plot.pdf',
            '<HIGH_QUANTILE_FREQUENCY_PLOT_FILE>': 'ps3_high_quantile_frequency_plot.pdf',
            '<ALL_FREQUENCY_PLOT_FILE>': 'ps3_all_frequency_plot.pdf',
            '<LOW_QUANTILE_100_FREQUENCY_PLOT_FILE>': 'ps3_low_quantile_100_frequency_plot.pdf',
            '<HIGH_QUANTILE_100_FREQUENCY_PLOT_FILE>': 'ps3_high_quantile_100_frequency_plot.pdf',
            '<ALL_100_FREQUENCY_PLOT_FILE>': 'ps3_all_100_frequency_plot.pdf',
            '<LOW_QUANTILE_200_FREQUENCY_PLOT_FILE>': 'ps3_low_quantile_200_frequency_plot.pdf',
            '<HIGH_QUANTILE_200_FREQUENCY_PLOT_FILE>': 'ps3_high_quantile_200_frequency_plot.pdf',
            '<ALL_200_FREQUENCY_PLOT_FILE>': 'ps3_all_200_frequency_plot.pdf',
            '<REGION_FREQUENCY_EXPERIMENT_COUNT_TABLE>': latex_table(self.get_passed_object('n_region_frequency_experiment')),
            '<REGION_SESSION_TOTAL_DATA>': latex_table(region_total, hlines=False)
        }

        TextTemplateUtils.replace_template(template_file_name=tex_template, out_file_name=report_tex_file_name, replace_dict=replace_dict)


class GeneratePlots(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    def run(self):
        self.create_dir_in_workspace('reports')

        low_quantile_frequency_plot_data = self.get_passed_object('low_quantile_frequency_plot')
        low_quantile_frequency_region_plot_data = self.get_passed_object('low_quantile_frequency_region_plot')
        low_quantile_frequency_frequency_plot_data = self.get_passed_object('low_quantile_frequency_frequency_plot')

        high_quantile_frequency_plot_data = self.get_passed_object('high_quantile_frequency_plot')
        high_quantile_frequency_region_plot_data = self.get_passed_object('high_quantile_frequency_region_plot')
        high_quantile_frequency_frequency_plot_data = self.get_passed_object('high_quantile_frequency_frequency_plot')

        all_frequency_plot_data = self.get_passed_object('all_frequency_plot')
        all_frequency_region_plot_data = self.get_passed_object('all_frequency_region_plot')
        all_frequency_frequency_plot_data = self.get_passed_object('all_frequency_frequency_plot')


        pdc = PlotDataCollection(legend_on=True, xlabel='Burst Frequency (Hz)', xlabel_fontsize=15)
        for v,p in low_quantile_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, ytitle=self.params.output_title, labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r = pdc.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = low_quantile_frequency_region_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = low_quantile_frequency_frequency_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        y_min = np.min(min_y_list)
        y_max = np.max(max_y_list)
        r = y_max - y_min
        y_min -= 0.05*r
        y_max += 0.40*r
        if y_min>0.0: y_min=0.0
        if y_max<0.0: y_max=0.0

        pdc.ylim=[y_min,y_max]
        low_quantile_frequency_region_plot_data.ylim=[y_min,y_max]
        low_quantile_frequency_frequency_plot_data.ylim=[y_min,y_max]

        pdc.xhline_pos = 0.0
        low_quantile_frequency_region_plot_data.xhline_pos = 0.0
        low_quantile_frequency_frequency_plot_data.xhline_pos = 0.0

        # label fontsize
        pdc.xlabel_fontsize = 16
        pdc.ylabel_fontsize = 16
        low_quantile_frequency_region_plot_data.xlabel_fontsize = 16
        low_quantile_frequency_region_plot_data.ylabel_fontsize = 16
        low_quantile_frequency_frequency_plot_data.xlabel_fontsize = 16
        low_quantile_frequency_frequency_plot_data.ylabel_fontsize = 16

        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)
        panel_plot.add_plot_data(0, 1, plot_data=low_quantile_frequency_region_plot_data)
        panel_plot.add_plot_data(0, 2, plot_data=low_quantile_frequency_frequency_plot_data)

        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_low_quantile_frequency_plot.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        pdc = PlotDataCollection(legend_on=True, xlabel='Burst Frequency (Hz)', xlabel_fontsize=15)
        for v,p in high_quantile_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, ytitle=self.params.output_title, labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r = pdc.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = high_quantile_frequency_region_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = high_quantile_frequency_frequency_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        y_min = np.min(min_y_list)
        y_max = np.max(max_y_list)
        r = y_max - y_min
        y_min -= 0.05*r
        y_max += 0.20*r
        if y_min>0.0: y_min=0.0
        if y_max<0.0: y_max=0.0

        pdc.ylim=[y_min,y_max]
        high_quantile_frequency_region_plot_data.ylim=[y_min,y_max]
        high_quantile_frequency_frequency_plot_data.ylim=[y_min,y_max]

        pdc.xhline_pos = 0.0
        high_quantile_frequency_region_plot_data.xhline_pos = 0.0
        high_quantile_frequency_frequency_plot_data.xhline_pos = 0.0

        # label fontsize
        pdc.xlabel_fontsize = 16
        pdc.ylabel_fontsize = 16
        high_quantile_frequency_region_plot_data.xlabel_fontsize = 16
        high_quantile_frequency_region_plot_data.ylabel_fontsize = 16
        high_quantile_frequency_frequency_plot_data.xlabel_fontsize = 16
        high_quantile_frequency_frequency_plot_data.ylabel_fontsize = 16

        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)
        panel_plot.add_plot_data(0, 1, plot_data=high_quantile_frequency_region_plot_data)
        panel_plot.add_plot_data(0, 2, plot_data=high_quantile_frequency_frequency_plot_data)

        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_high_quantile_frequency_plot.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        pdc = PlotDataCollection(legend_on=True, xlabel='Burst Frequency (Hz)', xlabel_fontsize=15)
        for v,p in all_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, ytitle=self.params.output_title, labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r = pdc.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = all_frequency_region_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = all_frequency_frequency_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        y_min = np.min(min_y_list)
        y_max = np.max(max_y_list)
        r = y_max - y_min
        y_min -= 0.05*r
        y_max += 0.20*r
        if y_min>0.0: y_min=0.0
        if y_max<0.0: y_max=0.0

        pdc.ylim=[y_min,y_max]
        all_frequency_region_plot_data.ylim=[y_min,y_max]
        all_frequency_frequency_plot_data.ylim=[y_min,y_max]

        pdc.xhline_pos = 0.0
        all_frequency_region_plot_data.xhline_pos = 0.0
        all_frequency_frequency_plot_data.xhline_pos = 0.0

        # label fontsize
        pdc.xlabel_fontsize = 16
        pdc.ylabel_fontsize = 16
        all_frequency_region_plot_data.xlabel_fontsize = 16
        all_frequency_region_plot_data.ylabel_fontsize = 16
        all_frequency_frequency_plot_data.xlabel_fontsize = 16
        all_frequency_frequency_plot_data.ylabel_fontsize = 16

        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)
        panel_plot.add_plot_data(0, 1, plot_data=all_frequency_region_plot_data)
        panel_plot.add_plot_data(0, 2, plot_data=all_frequency_frequency_plot_data)

        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_all_frequency_plot.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')



        low_quantile_frequency_plot_data = self.get_passed_object('low_quantile_100_frequency_plot')
        low_quantile_frequency_region_plot_data = self.get_passed_object('low_quantile_100_frequency_region_plot')
        low_quantile_frequency_frequency_plot_data = self.get_passed_object('low_quantile_100_frequency_frequency_plot')

        high_quantile_frequency_plot_data = self.get_passed_object('high_quantile_100_frequency_plot')
        high_quantile_frequency_region_plot_data = self.get_passed_object('high_quantile_100_frequency_region_plot')
        high_quantile_frequency_frequency_plot_data = self.get_passed_object('high_quantile_100_frequency_frequency_plot')

        all_frequency_plot_data = self.get_passed_object('all_100_frequency_plot')
        all_frequency_region_plot_data = self.get_passed_object('all_100_frequency_region_plot')
        all_frequency_frequency_plot_data = self.get_passed_object('all_100_frequency_frequency_plot')


        pdc = PlotDataCollection(legend_on=True, xlabel='Burst Frequency (Hz)', xlabel_fontsize=15)
        for v,p in low_quantile_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, ytitle=self.params.output_title, labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r = pdc.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = low_quantile_frequency_region_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = low_quantile_frequency_frequency_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        y_min = np.min(min_y_list)
        y_max = np.max(max_y_list)
        r = y_max - y_min
        y_min -= 0.05*r
        y_max += 0.40*r
        if y_min>0.0: y_min=0.0
        if y_max<0.0: y_max=0.0

        pdc.ylim=[y_min,y_max]
        low_quantile_frequency_region_plot_data.ylim=[y_min,y_max]
        low_quantile_frequency_frequency_plot_data.ylim=[y_min,y_max]

        pdc.xhline_pos = 0.0
        low_quantile_frequency_region_plot_data.xhline_pos = 0.0
        low_quantile_frequency_frequency_plot_data.xhline_pos = 0.0

        # label fontsize
        pdc.xlabel_fontsize = 16
        pdc.ylabel_fontsize = 16
        low_quantile_frequency_region_plot_data.xlabel_fontsize = 16
        low_quantile_frequency_region_plot_data.ylabel_fontsize = 16
        low_quantile_frequency_frequency_plot_data.xlabel_fontsize = 16
        low_quantile_frequency_frequency_plot_data.ylabel_fontsize = 16

        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)
        panel_plot.add_plot_data(0, 1, plot_data=low_quantile_frequency_region_plot_data)
        panel_plot.add_plot_data(0, 2, plot_data=low_quantile_frequency_frequency_plot_data)

        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_low_quantile_100_frequency_plot.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        pdc = PlotDataCollection(legend_on=True, xlabel='Burst Frequency (Hz)', xlabel_fontsize=15)
        for v,p in high_quantile_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, ytitle=self.params.output_title, labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r = pdc.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = high_quantile_frequency_region_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = high_quantile_frequency_frequency_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        y_min = np.min(min_y_list)
        y_max = np.max(max_y_list)
        r = y_max - y_min
        y_min -= 0.05*r
        y_max += 0.20*r
        if y_min>0.0: y_min=0.0
        if y_max<0.0: y_max=0.0

        pdc.ylim=[y_min,y_max]
        high_quantile_frequency_region_plot_data.ylim=[y_min,y_max]
        high_quantile_frequency_frequency_plot_data.ylim=[y_min,y_max]

        pdc.xhline_pos = 0.0
        high_quantile_frequency_region_plot_data.xhline_pos = 0.0
        high_quantile_frequency_frequency_plot_data.xhline_pos = 0.0

        # label fontsize
        pdc.xlabel_fontsize = 16
        pdc.ylabel_fontsize = 16
        high_quantile_frequency_region_plot_data.xlabel_fontsize = 16
        high_quantile_frequency_region_plot_data.ylabel_fontsize = 16
        high_quantile_frequency_frequency_plot_data.xlabel_fontsize = 16
        high_quantile_frequency_frequency_plot_data.ylabel_fontsize = 16

        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)
        panel_plot.add_plot_data(0, 1, plot_data=high_quantile_frequency_region_plot_data)
        panel_plot.add_plot_data(0, 2, plot_data=high_quantile_frequency_frequency_plot_data)

        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_high_quantile_100_frequency_plot.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        pdc = PlotDataCollection(legend_on=True, xlabel='Burst Frequency (Hz)', xlabel_fontsize=15)
        for v,p in all_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, ytitle=self.params.output_title, labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r = pdc.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = all_frequency_region_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = all_frequency_frequency_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        y_min = np.min(min_y_list)
        y_max = np.max(max_y_list)
        r = y_max - y_min
        y_min -= 0.05*r
        y_max += 0.20*r
        if y_min>0.0: y_min=0.0
        if y_max<0.0: y_max=0.0

        pdc.ylim=[y_min,y_max]
        all_frequency_region_plot_data.ylim=[y_min,y_max]
        all_frequency_frequency_plot_data.ylim=[y_min,y_max]

        pdc.xhline_pos = 0.0
        all_frequency_region_plot_data.xhline_pos = 0.0
        all_frequency_frequency_plot_data.xhline_pos = 0.0

        # label fontsize
        pdc.xlabel_fontsize = 16
        pdc.ylabel_fontsize = 16
        all_frequency_region_plot_data.xlabel_fontsize = 16
        all_frequency_region_plot_data.ylabel_fontsize = 16
        all_frequency_frequency_plot_data.xlabel_fontsize = 16
        all_frequency_frequency_plot_data.ylabel_fontsize = 16

        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)
        panel_plot.add_plot_data(0, 1, plot_data=all_frequency_region_plot_data)
        panel_plot.add_plot_data(0, 2, plot_data=all_frequency_frequency_plot_data)

        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_all_100_frequency_plot.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')



        low_quantile_frequency_plot_data = self.get_passed_object('low_quantile_200_frequency_plot')
        low_quantile_frequency_region_plot_data = self.get_passed_object('low_quantile_200_frequency_region_plot')
        low_quantile_frequency_frequency_plot_data = self.get_passed_object('low_quantile_200_frequency_frequency_plot')

        high_quantile_frequency_plot_data = self.get_passed_object('high_quantile_200_frequency_plot')
        high_quantile_frequency_region_plot_data = self.get_passed_object('high_quantile_200_frequency_region_plot')
        high_quantile_frequency_frequency_plot_data = self.get_passed_object('high_quantile_200_frequency_frequency_plot')

        all_frequency_plot_data = self.get_passed_object('all_200_frequency_plot')
        all_frequency_region_plot_data = self.get_passed_object('all_200_frequency_region_plot')
        all_frequency_frequency_plot_data = self.get_passed_object('all_200_frequency_frequency_plot')


        pdc = PlotDataCollection(legend_on=True, xlabel='Burst Frequency (Hz)', xlabel_fontsize=15)
        for v,p in low_quantile_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, ytitle=self.params.output_title, labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r = pdc.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = low_quantile_frequency_region_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = low_quantile_frequency_frequency_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        y_min = np.min(min_y_list)
        y_max = np.max(max_y_list)
        r = y_max - y_min
        y_min -= 0.05*r
        y_max += 0.40*r
        if y_min>0.0: y_min=0.0
        if y_max<0.0: y_max=0.0

        pdc.ylim=[y_min,y_max]
        low_quantile_frequency_region_plot_data.ylim=[y_min,y_max]
        low_quantile_frequency_frequency_plot_data.ylim=[y_min,y_max]

        pdc.xhline_pos = 0.0
        low_quantile_frequency_region_plot_data.xhline_pos = 0.0
        low_quantile_frequency_frequency_plot_data.xhline_pos = 0.0

        # label fontsize
        pdc.xlabel_fontsize = 16
        pdc.ylabel_fontsize = 16
        low_quantile_frequency_region_plot_data.xlabel_fontsize = 16
        low_quantile_frequency_region_plot_data.ylabel_fontsize = 16
        low_quantile_frequency_frequency_plot_data.xlabel_fontsize = 16
        low_quantile_frequency_frequency_plot_data.ylabel_fontsize = 16

        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)
        panel_plot.add_plot_data(0, 1, plot_data=low_quantile_frequency_region_plot_data)
        panel_plot.add_plot_data(0, 2, plot_data=low_quantile_frequency_frequency_plot_data)

        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_low_quantile_200_frequency_plot.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        pdc = PlotDataCollection(legend_on=True, xlabel='Burst Frequency (Hz)', xlabel_fontsize=15)
        for v,p in high_quantile_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, ytitle=self.params.output_title, labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r = pdc.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = high_quantile_frequency_region_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = high_quantile_frequency_frequency_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        y_min = np.min(min_y_list)
        y_max = np.max(max_y_list)
        r = y_max - y_min
        y_min -= 0.05*r
        y_max += 0.20*r
        if y_min>0.0: y_min=0.0
        if y_max<0.0: y_max=0.0

        pdc.ylim=[y_min,y_max]
        high_quantile_frequency_region_plot_data.ylim=[y_min,y_max]
        high_quantile_frequency_frequency_plot_data.ylim=[y_min,y_max]

        pdc.xhline_pos = 0.0
        high_quantile_frequency_region_plot_data.xhline_pos = 0.0
        high_quantile_frequency_frequency_plot_data.xhline_pos = 0.0

        # label fontsize
        pdc.xlabel_fontsize = 16
        pdc.ylabel_fontsize = 16
        high_quantile_frequency_region_plot_data.xlabel_fontsize = 16
        high_quantile_frequency_region_plot_data.ylabel_fontsize = 16
        high_quantile_frequency_frequency_plot_data.xlabel_fontsize = 16
        high_quantile_frequency_frequency_plot_data.ylabel_fontsize = 16

        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)
        panel_plot.add_plot_data(0, 1, plot_data=high_quantile_frequency_region_plot_data)
        panel_plot.add_plot_data(0, 2, plot_data=high_quantile_frequency_frequency_plot_data)

        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_high_quantile_200_frequency_plot.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        pdc = PlotDataCollection(legend_on=True, xlabel='Burst Frequency (Hz)', xlabel_fontsize=15)
        for v,p in all_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, ytitle=self.params.output_title, labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r = pdc.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = all_frequency_region_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = all_frequency_frequency_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        y_min = np.min(min_y_list)
        y_max = np.max(max_y_list)
        r = y_max - y_min
        y_min -= 0.05*r
        y_max += 0.20*r
        if y_min>0.0: y_min=0.0
        if y_max<0.0: y_max=0.0

        pdc.ylim=[y_min,y_max]
        all_frequency_region_plot_data.ylim=[y_min,y_max]
        all_frequency_frequency_plot_data.ylim=[y_min,y_max]

        pdc.xhline_pos = 0.0
        all_frequency_region_plot_data.xhline_pos = 0.0
        all_frequency_frequency_plot_data.xhline_pos = 0.0

        # label fontsize
        pdc.xlabel_fontsize = 16
        pdc.ylabel_fontsize = 16
        all_frequency_region_plot_data.xlabel_fontsize = 16
        all_frequency_region_plot_data.ylabel_fontsize = 16
        all_frequency_frequency_plot_data.xlabel_fontsize = 16
        all_frequency_frequency_plot_data.ylabel_fontsize = 16

        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)
        panel_plot.add_plot_data(0, 1, plot_data=all_frequency_region_plot_data)
        panel_plot.add_plot_data(0, 2, plot_data=all_frequency_frequency_plot_data)

        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_all_200_frequency_plot.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')





class GenerateReportPDF(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        from subprocess import call

        output_directory = self.get_path_to_resource_in_workspace('reports')

        texinputs_set_str = r'export TEXINPUTS="' + output_directory + '":$TEXINPUTS;'

        report_tex_file_name = self.get_passed_object('report_tex_file_name')

        # + '/Library/TeX/texbin/pdflatex '\
        pdflatex_command_str = texinputs_set_str \
                               + 'module load Tex;pdflatex '\
                               + ' -output-directory '+output_directory\
                               + ' -shell-escape ' \
                               + self.get_path_to_resource_in_workspace('reports/'+report_tex_file_name)

        call([pdflatex_command_str], shell=True)
