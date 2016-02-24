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
            '<FREQUENCY_PLOT_FILE>': 'ps3_frequency_aggregate_plots.pdf',
            '<FREQUENCY_PROJECTION_PLOT_FILE>': 'ps3_frequency_projection_plots.pdf',
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

        frequency_plot_data = self.get_passed_object('frequency_plot')
        centralized_frequency_plot_data = self.get_passed_object('centralized_frequency_plot')
        frequency_region_plot_data = self.get_passed_object('frequency_region_plot')
        frequency_frequency_plot_data = self.get_passed_object('frequency_frequency_plot')

        panel_plot = PanelPlot(xfigsize=11, yfigsize=11, i_max=1, j_max=1, title='', ytitle=self.params.output_title, ytitle_fontsize=16, wspace=0.3, hspace=0.3)

        pdc = PlotDataCollection(legend_on=True)
        pdc.xlabel = 'Burst Frequency (Hz)'
        pdc.xlabel_fontsize = 16
        for v,p in frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_frequency_aggregate_plots.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, title='', ytitle=self.params.output_title, wspace=0.3, hspace=0.3)
        panel_plot.add_plot_data(0, 0, plot_data=frequency_region_plot_data)
        panel_plot.add_plot_data(0, 1, plot_data=frequency_frequency_plot_data)
        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps3_frequency_projection_plots.pdf')
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
