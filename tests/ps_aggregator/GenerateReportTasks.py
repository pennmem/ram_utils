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
        tex_template = 'ps_aggregator.tex.tpl'

        report_tex_file_name = 'ps_aggregator.tex'
        self.pass_object('report_tex_file_name',report_tex_file_name)

        self.set_file_resources_to_move(report_tex_file_name, dst='reports')

        region_total = self.get_passed_object('region_session_total')
        area_total = self.get_passed_object('area_session_total')

        region_total = sorted(zip(region_total.keys(), region_total.values()))
        area_total = sorted(zip(area_total.keys(), area_total.values()))

        replace_dict = {
            '<DATE>': datetime.date.today(),
            '<FREQUENCY_PLOT_FILE>': 'ps_frequency_aggregate_plots.pdf',
            '<FVALUERF1>': np.nan,
            '<FVALUERF2>': np.nan,
            '<FVALUERF12>': np.nan,
            '<PVALUERF1>': pvalue_formatting(np.nan),
            '<PVALUERF2>': pvalue_formatting(np.nan),
            '<PVALUERF12>': pvalue_formatting(np.nan),
            '<AMPLITUDE_LOW_PLOT_FILE>': 'ps_amplitude_low_aggregate_plots.pdf',
            '<FVALUERALOW1>': np.nan,
            '<FVALUERALOW2>': np.nan,
            '<FVALUERALOW12>': np.nan,
            '<PVALUERALOW1>': pvalue_formatting(np.nan),
            '<PVALUERALOW2>': pvalue_formatting(np.nan),
            '<PVALUERALOW12>': pvalue_formatting(np.nan),
            '<AMPLITUDE_HIGH_PLOT_FILE>': 'ps_amplitude_high_aggregate_plots.pdf',
            '<FVALUERAHIGH1>': np.nan,
            '<FVALUERAHIGH2>': np.nan,
            '<FVALUERAHIGH12>': np.nan,
            '<PVALUERAHIGH1>': pvalue_formatting(np.nan),
            '<PVALUERAHIGH2>': pvalue_formatting(np.nan),
            '<PVALUERAHIGH12>': pvalue_formatting(np.nan),
            '<DURATION_LOW_PLOT_FILE>': 'ps_duration_low_aggregate_plots.pdf',
            '<FVALUERDLOW1>': np.nan,
            '<FVALUERDLOW2>': np.nan,
            '<FVALUERDLOW12>': np.nan,
            '<PVALUERDLOW1>': pvalue_formatting(np.nan),
            '<PVALUERDLOW2>': pvalue_formatting(np.nan),
            '<PVALUERDLOW12>': pvalue_formatting(np.nan),
            '<DURATION_HIGH_PLOT_FILE>': 'ps_duration_high_aggregate_plots.pdf',
            '<FVALUERDHIGH1>': np.nan,
            '<FVALUERDHIGH2>': np.nan,
            '<FVALUERDHIGH12>': np.nan,
            '<PVALUERDHIGH1>': pvalue_formatting(np.nan),
            '<PVALUERDHIGH2>': pvalue_formatting(np.nan),
            '<PVALUERDHIGH12>': pvalue_formatting(np.nan),
            '<REGION_SESSION_TOTAL_DATA>': latex_table(region_total, hlines=False)
        }

        TextTemplateUtils.replace_template(template_file_name=tex_template, out_file_name=report_tex_file_name, replace_dict=replace_dict)


class GeneratePlots(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        self.create_dir_in_workspace('reports')

        frequency_plot_data = self.get_passed_object('frequency_plot')
        low_freq_duration_plot_data = self.get_passed_object('low_freq_duration_plot')
        high_freq_duration_plot_data = self.get_passed_object('high_freq_duration_plot')
        low_freq_amplitude_plot_data = self.get_passed_object('low_freq_amplitude_plot')
        high_freq_amplitude_plot_data = self.get_passed_object('high_freq_amplitude_plot')


        panel_plot = PanelPlot(i_max=1, j_max=1, title='', ytitle='$\Delta$ Post-Pre Classifier Output', ytitle_fontsize=24, wspace=0.3, hspace=0.3)

        pdc = PlotDataCollection(legend_on=True)
        pdc.xlabel = 'Pulse Frequency (Hz)'
        pdc.xlabel_fontsize = 24
        for v,p in frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps_frequency_aggregate_plots.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        panel_plot = PanelPlot(i_max=1, j_max=1, title='', ytitle='$\Delta$ Post-Pre Classifier Output', ytitle_fontsize=24, wspace=0.3, hspace=0.3)

        pdc = PlotDataCollection(legend_on=True)
        pdc.xlabel = 'Duration (ms)'
        pdc.xlabel_fontsize = 24
        for v,p in low_freq_duration_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps_duration_low_aggregate_plots.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        panel_plot = PanelPlot(i_max=1, j_max=1, title='', ytitle='$\Delta$ Post-Pre Classifier Output', ytitle_fontsize=24, wspace=0.3, hspace=0.3)

        pdc = PlotDataCollection(legend_on=True)
        pdc.xlabel = 'Duration (ms)'
        pdc.xlabel_fontsize = 24
        for v,p in high_freq_duration_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps_duration_high_aggregate_plots.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        panel_plot = PanelPlot(i_max=1, j_max=1, title='', ytitle='$\Delta$ Post-Pre Classifier Output', ytitle_fontsize=24, wspace=0.3, hspace=0.3)

        pdc = PlotDataCollection(legend_on=True)
        pdc.xlabel = 'Amplitude (mA)'
        pdc.xlabel_fontsize = 24
        for v,p in low_freq_amplitude_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps_amplitude_low_aggregate_plots.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        panel_plot = PanelPlot(i_max=1, j_max=1, title='', ytitle='$\Delta$ Post-Pre Classifier Output', ytitle_fontsize=24, wspace=0.3, hspace=0.3)

        pdc = PlotDataCollection(legend_on=True)
        pdc.xlabel = 'Amplitude (mA)'
        pdc.xlabel_fontsize = 24
        for v,p in high_freq_amplitude_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps_amplitude_high_aggregate_plots.pdf')

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
