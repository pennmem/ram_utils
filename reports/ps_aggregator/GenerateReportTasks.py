__author__ = 'm'

from RamPipeline import *

from PlotUtils import PlotData, BarPlotData, PlotDataCollection, PanelPlot
import TextTemplateUtils

import datetime
import numpy as np

from latex_table import latex_table
import shutil

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
        region_total.append(('Total',sum([v[1] for v in region_total])))

        area_total = sorted(zip(area_total.keys(), area_total.values()))

        # fvalue_rf = self.get_passed_object('fvalue_rf')
        # pvalue_rf = self.get_passed_object('pvalue_rf')
        #
        # fvalue_ra_low = self.get_passed_object('fvalue_ra_low')
        # pvalue_ra_low = self.get_passed_object('pvalue_ra_low')
        #
        # fvalue_ra_high = self.get_passed_object('fvalue_ra_high')
        # pvalue_ra_high = self.get_passed_object('pvalue_ra_high')
        #
        # fvalue_rd_low = self.get_passed_object('fvalue_rd_low')
        # pvalue_rd_low = self.get_passed_object('pvalue_rd_low')
        #
        # fvalue_rd_high = self.get_passed_object('fvalue_rd_high')
        # pvalue_rd_high = self.get_passed_object('pvalue_rd_high')

        # replace_dict = {
        #     '<DATE>': datetime.date.today(),
        #     '<FREQUENCY_PLOT_FILE>': 'ps_frequency_aggregate_plots.pdf',
        #     '<FVALUERF1>': '%.2f' % fvalue_rf[0],
        #     '<FVALUERF2>': '%.2f' % fvalue_rf[1],
        #     '<FVALUERF12>': '%.2f' % fvalue_rf[2],
        #     '<PVALUERF1>': pvalue_formatting(pvalue_rf[0]),
        #     '<PVALUERF2>': pvalue_formatting(pvalue_rf[1]),
        #     '<PVALUERF12>': pvalue_formatting(pvalue_rf[2]),
        #     '<AMPLITUDE_LOW_PLOT_FILE>': 'ps_amplitude_low_aggregate_plots.pdf',
        #     '<FVALUERALOW1>': '%.2f' % fvalue_ra_low[0],
        #     '<FVALUERALOW2>': '%.2f' % fvalue_ra_low[1],
        #     '<FVALUERALOW12>': '%.2f' % fvalue_ra_low[2],
        #     '<PVALUERALOW1>': pvalue_formatting(pvalue_ra_low[0]),
        #     '<PVALUERALOW2>': pvalue_formatting(pvalue_ra_low[1]),
        #     '<PVALUERALOW12>': pvalue_formatting(pvalue_ra_low[2]),
        #     '<AMPLITUDE_HIGH_PLOT_FILE>': 'ps_amplitude_high_aggregate_plots.pdf',
        #     '<FVALUERAHIGH1>': '%.2f' % fvalue_ra_high[0],
        #     '<FVALUERAHIGH2>': '%.2f' % fvalue_ra_high[1],
        #     '<FVALUERAHIGH12>': '%.2f' % fvalue_ra_high[2],
        #     '<PVALUERAHIGH1>': pvalue_formatting(pvalue_ra_high[0]),
        #     '<PVALUERAHIGH2>': pvalue_formatting(pvalue_ra_high[1]),
        #     '<PVALUERAHIGH12>': pvalue_formatting(pvalue_ra_high[2]),
        #     '<DURATION_LOW_PLOT_FILE>': 'ps_duration_low_aggregate_plots.pdf',
        #     '<FVALUERDLOW1>': '%.2f' % fvalue_rd_low[0],
        #     '<FVALUERDLOW2>': '%.2f' % fvalue_rd_low[1],
        #     '<FVALUERDLOW12>': '%.2f' % fvalue_rd_low[2],
        #     '<PVALUERDLOW1>': pvalue_formatting(pvalue_rd_low[0]),
        #     '<PVALUERDLOW2>': pvalue_formatting(pvalue_rd_low[1]),
        #     '<PVALUERDLOW12>': pvalue_formatting(pvalue_rd_low[2]),
        #     '<DURATION_HIGH_PLOT_FILE>': 'ps_duration_high_aggregate_plots.pdf',
        #     '<FVALUERDHIGH1>': '%.2f' % fvalue_rd_high[0],
        #     '<FVALUERDHIGH2>': '%.2f' % fvalue_rd_high[1],
        #     '<FVALUERDHIGH12>': '%.2f' % fvalue_rd_high[2],
        #     '<PVALUERDHIGH1>': pvalue_formatting(pvalue_rd_high[0]),
        #     '<PVALUERDHIGH2>': pvalue_formatting(pvalue_rd_high[1]),
        #     '<PVALUERDHIGH12>': pvalue_formatting(pvalue_rd_high[2]),
        #     '<REGION_SESSION_TOTAL_DATA>': latex_table(region_total, hlines=False)
        # }

        replace_dict = {
            '<DATE>': datetime.date.today(),
            '<ALL_FREQUENCY_CLASSIFIER_DELTA_PLOT_FILE>': 'ps_prob_diff_frequency_plot.pdf',
            '<ALL_FREQUENCY_RECALL_CHANGE_PLOT_FILE>': 'ps_perf_diff_frequency_plot.pdf',
            '<REGION_FREQUENCY_EXPERIMENT_COUNT_TABLE>': latex_table(self.get_passed_object('n_region_frequency_experiment')),
            '<FVALUERF1>': '%.2f' % np.nan,
            '<FVALUERF2>': '%.2f' % np.nan,
            '<FVALUERF12>': '%.2f' % np.nan,
            '<PVALUERF1>': pvalue_formatting(np.nan),
            '<PVALUERF2>': pvalue_formatting(np.nan),
            '<PVALUERF12>': pvalue_formatting(np.nan),
            '<ALL_AMPLITUDE_CLASSIFIER_DELTA_PLOT_FILE>': 'ps_prob_diff_amplitude_plot.pdf',
            '<ALL_AMPLITUDE_RECALL_CHANGE_PLOT_FILE>': 'ps_perf_diff_amplitude_plot.pdf',
            '<FVALUERALOW1>': '%.2f' % np.nan,
            '<FVALUERALOW2>': '%.2f' % np.nan,
            '<FVALUERALOW12>': '%.2f' % np.nan,
            '<PVALUERALOW1>': pvalue_formatting(np.nan),
            '<PVALUERALOW2>': pvalue_formatting(np.nan),
            '<PVALUERALOW12>': pvalue_formatting(np.nan),
            '<FVALUERAHIGH1>': '%.2f' % np.nan,
            '<FVALUERAHIGH2>': '%.2f' % np.nan,
            '<FVALUERAHIGH12>': '%.2f' % np.nan,
            '<PVALUERAHIGH1>': pvalue_formatting(np.nan),
            '<PVALUERAHIGH2>': pvalue_formatting(np.nan),
            '<PVALUERAHIGH12>': pvalue_formatting(np.nan),
            '<ALL_DURATION_CLASSIFIER_DELTA_PLOT_FILE>': 'ps_prob_diff_duration_plot.pdf',
            '<ALL_DURATION_RECALL_CHANGE_PLOT_FILE>': 'ps_perf_diff_duration_plot.pdf',
            '<FVALUERDLOW1>': '%.2f' % np.nan,
            '<FVALUERDLOW2>': '%.2f' % np.nan,
            '<FVALUERDLOW12>': '%.2f' % np.nan,
            '<PVALUERDLOW1>': pvalue_formatting(np.nan),
            '<PVALUERDLOW2>': pvalue_formatting(np.nan),
            '<PVALUERDLOW12>': pvalue_formatting(np.nan),
            '<FVALUERDHIGH1>': '%.2f' % np.nan,
            '<FVALUERDHIGH2>': '%.2f' % np.nan,
            '<FVALUERDHIGH12>': '%.2f' % np.nan,
            '<PVALUERDHIGH1>': pvalue_formatting(np.nan),
            '<PVALUERDHIGH2>': pvalue_formatting(np.nan),
            '<PVALUERDHIGH12>': pvalue_formatting(np.nan),
            '<REGION_SESSION_TOTAL_DATA>': latex_table(region_total, hlines=False)
        }

        TextTemplateUtils.replace_template(template_file_name=tex_template, out_file_name=report_tex_file_name, replace_dict=replace_dict)


class GeneratePlots(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    def run(self):
        self.create_dir_in_workspace('reports')

        prob_diff_frequency_plot_data = self.get_passed_object('all_prob_diff_frequency_plot')
        prob_diff_frequency_region_plot_data = self.get_passed_object('all_prob_diff_frequency_region_plot')
        prob_diff_frequency_frequency_plot_data = self.get_passed_object('all_prob_diff_frequency_frequency_plot')

        prob_diff_low_freq_duration_plot_data = self.get_passed_object('all_prob_diff_low_freq_duration_plot')
        prob_diff_high_freq_duration_plot_data = self.get_passed_object('all_prob_diff_high_freq_duration_plot')
        prob_diff_low_freq_amplitude_plot_data = self.get_passed_object('all_prob_diff_low_freq_amplitude_plot')
        prob_diff_high_freq_amplitude_plot_data = self.get_passed_object('all_prob_diff_high_freq_amplitude_plot')

        perf_diff_frequency_plot_data = self.get_passed_object('all_perf_diff_frequency_plot')
        perf_diff_frequency_region_plot_data = self.get_passed_object('all_perf_diff_frequency_region_plot')
        perf_diff_frequency_frequency_plot_data = self.get_passed_object('all_perf_diff_frequency_frequency_plot')

        perf_diff_low_freq_duration_plot_data = self.get_passed_object('all_perf_diff_low_freq_duration_plot')
        perf_diff_high_freq_duration_plot_data = self.get_passed_object('all_perf_diff_high_freq_duration_plot')
        perf_diff_low_freq_amplitude_plot_data = self.get_passed_object('all_perf_diff_low_freq_amplitude_plot')
        perf_diff_high_freq_amplitude_plot_data = self.get_passed_object('all_perf_diff_high_freq_amplitude_plot')


        pdc = PlotDataCollection(legend_on=True, legend_loc=3, xlabel='Pulse Frequency (Hz)', xlabel_fontsize=15)
        for v,p in prob_diff_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, title='', ytitle='$\Delta$ Post-Pre Classifier Output', labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r =pdc.get_yrange()
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
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps_prob_diff_frequency_plot.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        pdc = PlotDataCollection(legend_on=True, legend_loc=3, xlabel='Pulse Frequency (Hz)', xlabel_fontsize=15)
        for v,p in perf_diff_frequency_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=3, title='', ytitle='Expected Recall Change (%)', labelsize=16, ytitle_fontsize=18)
        min_y_list = []
        max_y_list = []

        r =pdc.get_yrange()
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
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps_perf_diff_frequency_plot.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=2, ytitle='$\Delta$ Post-Pre Classifier Output', labelsize=20, ytitle_fontsize=20)

        pdc = PlotDataCollection(legend_on=True, legend_loc=3)
        pdc.xlabel = 'Duration (ms)'
        pdc.xlabel_fontsize = 16
        for v,p in prob_diff_low_freq_duration_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

        pdc = PlotDataCollection(legend_on=True, legend_loc=3)
        pdc.xlabel = 'Duration (ms)'
        pdc.xlabel_fontsize = 16
        for v,p in prob_diff_high_freq_duration_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 1, plot_data_collection=pdc)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps_prob_diff_duration_plot.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=2, ytitle='Expected Recall Change (%)', labelsize=20, ytitle_fontsize=20)

        pdc = PlotDataCollection(legend_on=True, legend_loc=3)
        pdc.xlabel = 'Duration (ms)'
        pdc.xlabel_fontsize = 16
        for v,p in perf_diff_low_freq_duration_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

        pdc = PlotDataCollection(legend_on=True, legend_loc=3)
        pdc.xlabel = 'Duration (ms)'
        pdc.xlabel_fontsize = 16
        for v,p in perf_diff_high_freq_duration_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 1, plot_data_collection=pdc)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps_perf_diff_duration_plot.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')



        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=2, title='', ytitle='$\Delta$ Post-Pre Classifier Output', labelsize=20, ytitle_fontsize=20)

        pdc = PlotDataCollection(legend_on=True, legend_loc=3)
        pdc.xlabel = 'Amplitude (mA)'
        pdc.xlabel_fontsize = 16
        for v,p in prob_diff_low_freq_amplitude_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

        pdc = PlotDataCollection(legend_on=True, legend_loc=3)
        pdc.xlabel = 'Amplitude (mA)'
        pdc.xlabel_fontsize = 16
        for v,p in prob_diff_high_freq_amplitude_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 1, plot_data_collection=pdc)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps_prob_diff_amplitude_plot.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


        panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=2, title='', ytitle='Expected Recall Change (%)', labelsize=20, ytitle_fontsize=20)

        pdc = PlotDataCollection(legend_on=True, legend_loc=3)
        pdc.xlabel = 'Amplitude (mA)'
        pdc.xlabel_fontsize = 16
        for v,p in perf_diff_low_freq_amplitude_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

        pdc = PlotDataCollection(legend_on=True, legend_loc=3)
        pdc.xlabel = 'Amplitude (mA)'
        pdc.xlabel_fontsize = 16
        for v,p in perf_diff_high_freq_amplitude_plot_data.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)
        panel_plot.add_plot_data_collection(0, 1, plot_data_collection=pdc)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/ps_perf_diff_amplitude_plot.pdf')

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


        report_core_file_name, ext = splitext(report_tex_file_name)


        src = join(output_directory,report_core_file_name + '.pdf')
        dst = '/protocols/r1/reports/ps1-2_dashboard.pdf'

        shutil.copy(src,dst)
        # self.pipeline.deploy_report(report_path=report_file)


