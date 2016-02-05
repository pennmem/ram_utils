__author__ = 'm'

from RamPipeline import *

from PlotUtils import PlotData, BarPlotData, PlotDataCollection, PanelPlot
import TextTemplateUtils


def pvalue_formatting(p):
    return '\leq 0.001' if p<=0.001 else ('%.3f'%p)


class GeneratePlots(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        #experiment = self.pipeline.experiment

        self.create_dir_in_workspace('reports')

        frequency_plot_data = self.get_passed_object('frequency_plot')

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


        low_freq_duration_plot_data = self.get_passed_object('low_freq_duration_plot')
        high_freq_duration_plot_data = self.get_passed_object('high_freq_duration_plot')
        low_freq_amplitude_plot_data = self.get_passed_object('low_freq_amplitude_plot')
        high_freq_amplitude_plot_data = self.get_passed_object('high_freq_amplitude_plot')

        panel_plot = PanelPlot(i_max=2, j_max=2, title='', ytitle='$\Delta$ Post-Pre Classifier Output', ytitle_fontsize=24, wspace=0.3, hspace=0.3)

        pdc = PlotDataCollection(legend_on=True)
        pdc.xlabel = 'Duration (ms)'
        pdc.xlabel_fontsize = 24
        for v,p in low_freq_duration_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

        pdc = PlotDataCollection(legend_on=True)
        pdc.xlabel = 'Duration (ms)'
        pdc.xlabel_fontsize = 24
        for v,p in high_freq_duration_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 1, plot_data_collection=pdc)

        pdc = PlotDataCollection(legend_on=True)
        pdc.xlabel = 'Amplitude (mA)'
        pdc.xlabel_fontsize = 24
        for v,p in low_freq_amplitude_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(1, 0, plot_data_collection=pdc)

        pdc = PlotDataCollection(legend_on=True)
        pdc.xlabel = 'Amplitude (mA)'
        pdc.xlabel_fontsize = 24
        for v,p in high_freq_amplitude_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(1, 1, plot_data_collection=pdc)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps_amplitude_duration_aggregate_plots.pdf')

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
