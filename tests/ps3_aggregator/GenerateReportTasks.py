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
            '<ALL_FREQUENCY_CLASSIFIER_DELTA_PLOT_FILE>': 'ps3_prob_diff_frequency_plot.pdf',
            '<ALL_FREQUENCY_RECALL_CHANGE_PLOT_FILE>': 'ps3_perf_diff_frequency_plot.pdf',
            '<ALL_100_FREQUENCY_CLASSIFIER_DELTA_PLOT_FILE>': 'ps3_prob_diff_100_frequency_plot.pdf',
            '<ALL_100_FREQUENCY_RECALL_CHANGE_PLOT_FILE>': 'ps3_perf_diff_100_frequency_plot.pdf',
            '<ALL_200_FREQUENCY_CLASSIFIER_DELTA_PLOT_FILE>': 'ps3_prob_diff_200_frequency_plot.pdf',
            '<ALL_200_FREQUENCY_RECALL_CHANGE_PLOT_FILE>': 'ps3_perf_diff_200_frequency_plot.pdf',
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

        self.create_plots('')
        self.create_plots('_100')
        self.create_plots('_200')

    def create_plots(self, pf):
        prob_diff_frequency_plot_data = self.get_passed_object('all%s_prob_diff_frequency_plot' % pf)
        prob_diff_frequency_region_plot_data = self.get_passed_object('all%s_prob_diff_frequency_region_plot' % pf)
        prob_diff_frequency_frequency_plot_data = self.get_passed_object('all%s_prob_diff_frequency_frequency_plot' % pf)

        perf_diff_frequency_plot_data = self.get_passed_object('all%s_perf_diff_frequency_plot' % pf)
        perf_diff_frequency_region_plot_data = self.get_passed_object('all%s_perf_diff_frequency_region_plot' % pf)
        perf_diff_frequency_frequency_plot_data = self.get_passed_object('all%s_perf_diff_frequency_frequency_plot' % pf)


        pdc = PlotDataCollection(legend_on=True, legend_loc=3, xlabel='Burst Frequency (Hz)', xlabel_fontsize=15)
        for v,p in prob_diff_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, ytitle='$\Delta$ Post-Pre Classifier Output', labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r = pdc.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = prob_diff_frequency_region_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = prob_diff_frequency_frequency_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        y_min = np.min(min_y_list)
        y_max = np.max(max_y_list)
        r = y_max - y_min
        y_min -= 0.05*r
        y_max += 0.05*r
        if y_min>0.0: y_min=0.0
        if y_max<0.0: y_max=0.0

        pdc.ylim=[y_min,y_max]
        prob_diff_frequency_region_plot_data.ylim=[y_min,y_max]
        prob_diff_frequency_frequency_plot_data.ylim=[y_min,y_max]

        pdc.xhline_pos = 0.0
        prob_diff_frequency_region_plot_data.xhline_pos = 0.0
        prob_diff_frequency_frequency_plot_data.xhline_pos = 0.0

        # label fontsize
        pdc.xlabel_fontsize = 16
        pdc.ylabel_fontsize = 16
        prob_diff_frequency_region_plot_data.xlabel_fontsize = 16
        prob_diff_frequency_region_plot_data.ylabel_fontsize = 16
        prob_diff_frequency_frequency_plot_data.xlabel_fontsize = 16
        prob_diff_frequency_frequency_plot_data.ylabel_fontsize = 16

        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)
        panel_plot.add_plot_data(0, 1, plot_data=prob_diff_frequency_region_plot_data)
        panel_plot.add_plot_data(0, 2, plot_data=prob_diff_frequency_frequency_plot_data)

        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_prob_diff%s_frequency_plot.pdf' % pf)
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        pdc = PlotDataCollection(legend_on=True, legend_loc=3, xlabel='Burst Frequency (Hz)', xlabel_fontsize=15)
        for v,p in perf_diff_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, ytitle='Expected Recall Change (%)', labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r = pdc.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = perf_diff_frequency_region_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        r = perf_diff_frequency_frequency_plot_data.get_yrange()
        min_y_list.append(r[0])
        max_y_list.append(r[1])

        y_min = np.min(min_y_list)
        y_max = np.max(max_y_list)
        r = y_max - y_min
        y_min -= 0.05*r
        y_max += 0.05*r
        if y_min>0.0: y_min=0.0
        if y_max<0.0: y_max=0.0

        pdc.ylim=[y_min,y_max]
        perf_diff_frequency_region_plot_data.ylim=[y_min,y_max]
        perf_diff_frequency_frequency_plot_data.ylim=[y_min,y_max]

        pdc.xhline_pos = 0.0
        perf_diff_frequency_region_plot_data.xhline_pos = 0.0
        perf_diff_frequency_frequency_plot_data.xhline_pos = 0.0

        # label fontsize
        pdc.xlabel_fontsize = 16
        pdc.ylabel_fontsize = 16
        perf_diff_frequency_region_plot_data.xlabel_fontsize = 16
        perf_diff_frequency_region_plot_data.ylabel_fontsize = 16
        perf_diff_frequency_frequency_plot_data.xlabel_fontsize = 16
        perf_diff_frequency_frequency_plot_data.ylabel_fontsize = 16

        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)
        panel_plot.add_plot_data(0, 1, plot_data=perf_diff_frequency_region_plot_data)
        panel_plot.add_plot_data(0, 2, plot_data=perf_diff_frequency_frequency_plot_data)

        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_perf_diff%s_frequency_plot.pdf' % pf)
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
