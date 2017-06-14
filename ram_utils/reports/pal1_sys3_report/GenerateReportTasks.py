import datetime
from os import path
import shutil
from subprocess import call

from ... import TextTemplateUtils
import numpy as np
from ...RamPipeline import *
from ...ReportUtils import ReportRamTask, ReportDeployer

from latex_table import latex_table
from ram_utils.PlotUtils import PlotData, BarPlotData, PanelPlot


def get_template(template):
    return path.join(path.dirname(path.abspath(__file__)),template)

class GenerateTex(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GenerateTex, self).__init__(mark_as_completed)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        # tex_session_template = task + '_session.tex.tpl'

        n_sess = self.get_passed_object('NUMBER_OF_SESSIONS')
        n_bps = self.get_passed_object('NUMBER_OF_ELECTRODES')


        #
        # self.pass_object('report_tex_file_names', report_tex_file_names)

        tex_combined_template = get_template(task + '_combined.tex.tpl')
        combined_report_tex_file_name = '%s_%s_report.tex' % (subject, task)

        self.set_file_resources_to_move(combined_report_tex_file_name, dst='reports')

        cumulative_summary = self.get_passed_object('cumulative_summary')

        cumulative_data_tex_table = latex_table(self.get_passed_object('SESSION_DATA'))

        cumulative_ttest_tex_table = latex_table(self.get_passed_object('cumulative_ttest_data'))

        combined_cumulative_summary = self.get_passed_object('combined_cumulative_summary')

        pal_cumulative_summary = self.get_passed_object('pal_cumulative_summary')

        params = self.get_passed_object('params')

        replace_dict = {
            '<PROB_RECALL_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-prob_recall_plot_combined.pdf',
            '<DATE>': datetime.date.today(),
            '<SESSION_DATA>': cumulative_data_tex_table,
            '<SUBJECT>': subject.replace('_', '\\textunderscore'),
            '<NUMBER_OF_SESSIONS>': n_sess,
            '<NUMBER_OF_ELECTRODES>': n_bps,
            '<N_PAIRS>': cumulative_summary.n_pairs,
            '<N_CORRECT_PAIRS>': cumulative_summary.n_correct_pairs,
            '<PC_CORRECT_PAIRS>': '%.2f' % cumulative_summary.pc_correct_pairs,
            '<WILSON1>': '%.2f' % cumulative_summary.wilson1,
            '<WILSON2>': '%.2f' % cumulative_summary.wilson2,
            '<N_VOC_PASS>': cumulative_summary.n_voc_pass,
            '<PC_VOC_PASS>': '%.2f' % cumulative_summary.pc_voc_pass,
            '<N_NONVOC_PASS>': cumulative_summary.n_nonvoc_pass,
            '<PC_NONVOC_PASS>': '%.2f' % cumulative_summary.pc_nonvoc_pass,
            '<N_PLI>': cumulative_summary.n_pli,
            '<PC_PLI>': '%.2f' % cumulative_summary.pc_pli,
            '<N_ELI>': cumulative_summary.n_eli,
            '<PC_ELI>': '%.2f' % cumulative_summary.pc_eli,
            '<N_MATH>': cumulative_summary.n_math,
            '<N_CORRECT_MATH>': cumulative_summary.n_correct_math,
            '<PC_CORRECT_MATH>': '%.2f' % cumulative_summary.pc_correct_math,
            '<MATH_PER_LIST>': '%.2f' % cumulative_summary.math_per_list,
            '<SIGNIFICANT_ELECTRODES>': cumulative_ttest_tex_table,
            '<AUC>': cumulative_summary.auc,
            '<PERM-P-VALUE>': cumulative_summary.perm_test_pvalue,
            '<J-THRESH>': cumulative_summary.jstat_thresh,
            '<J-PERC>': cumulative_summary.jstat_percentile,
            '<ROC_AND_TERC_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-roc_and_terc_plot_combined.pdf',

            '<ROC_AND_TERC_PLOT_FILE_FR_PAL_ENC>': self.pipeline.task + '-' + self.pipeline.subject + '-roc_and_terc_plot_fr_pal_test_enc.pdf',
            '<PERM-P-VALUE_FR_PAL_ENC>':combined_cumulative_summary.perm_test_pvalue,
            '<J-THRESH_FR_PAL_ENC>':combined_cumulative_summary.jstat_thresh,
            '<AUC_FR_PAL_ENC>': combined_cumulative_summary.auc,

            '<ROC_AND_TERC_PLOT_FILE_PAL_ENC>': self.pipeline.task + '-' + self.pipeline.subject + '-roc_and_terc_plot_pal_test_enc.pdf',
            '<PERM-P-VALUE_PAL_ENC>':pal_cumulative_summary.perm_test_pvalue,
            '<J-THRESH_PAL_ENC>':pal_cumulative_summary.jstat_thresh,
            '<AUC_PAL_ENC>': pal_cumulative_summary.auc,
            '<ENCODING_SAMPLES_WEIGHT>':params.encoding_samples_weight,
            '<PAL_SAMPLES_WEIGHT>':params.pal_samples_weight,

            }

        TextTemplateUtils.replace_template(template_file_name=tex_combined_template,
                                           out_file_name=combined_report_tex_file_name, replace_dict=replace_dict)

        self.pass_object('combined_report_tex_file_name', combined_report_tex_file_name)


class GeneratePlots(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GeneratePlots, self).__init__(mark_as_completed)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.create_dir_in_workspace('reports')

        # session_summary_array = self.get_passed_object('session_summary_array')

        # serial_positions = np.arange(1,7)
        # lag_positions = np.arange(1,12)

        cumulative_summary = self.get_passed_object('cumulative_summary')

        panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, labelsize=18, wspace=20.0)

        pd1 = PlotData(x=cumulative_summary.positions, y=cumulative_summary.prob_recall,
                       xlim=(0, len(cumulative_summary.positions)), ylim=(0.0, 1.0), xlabel='Serial position\n(a)',
                       ylabel='Probability of recall', xlabel_fontsize=18, ylabel_fontsize=18)
        panel_plot.add_plot_data(0, 0, plot_data=pd1)

        pd2 = PlotData(x=cumulative_summary.study_lag_values, y=cumulative_summary.prob_study_lag,
                       xlim=(0, len(cumulative_summary.study_lag_values)), ylim=(0.0, 1.0),
                       xlabel='Study-test Lag\n(a)', ylabel='Probability of recall', xlabel_fontsize=18,
                       ylabel_fontsize=18)
        panel_plot.add_plot_data(0, 1, plot_data=pd2)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace(
            'reports/' + task + '-' + subject + '-prob_recall_plot_combined.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        # --------------------------------------------------------------------------------------------------------
        # -------------------------  classifier plot - trained on PAL1 encoding tested on PAL1 encoding --------------------

        panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, title='', labelsize=18)

        pd1 = PlotData(x=cumulative_summary.fpr, y=cumulative_summary.tpr, xlim=[0.0, 1.0], ylim=[0.0, 1.0],
                       xlabel='False Alarm Rate\n(a)', ylabel='Hit Rate', levelline=((0.001, 0.999), (0.001, 0.999)),
                       color='k', markersize=1.0, xlabel_fontsize=18, ylabel_fontsize=18)

        ylim = np.max(np.abs(cumulative_summary.pc_diff_from_mean)) + 5.0
        if ylim > 100.0:
            ylim = 100.0
        pd2 = BarPlotData(x=(0, 1, 2), y=cumulative_summary.pc_diff_from_mean, ylim=[-ylim, ylim],
                          xlabel='Tercile of Classifier Estimate\n(b)', ylabel='Recall Change From Mean (%)',
                          x_tick_labels=['Low', 'Middle', 'High'], xhline_pos=0.0, barcolors=['grey', 'grey', 'grey'],
                          xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)

        panel_plot.add_plot_data(0, 0, plot_data=pd1)
        panel_plot.add_plot_data(0, 1, plot_data=pd2)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace(
            'reports/' + task + '-' + subject + '-roc_and_terc_plot_combined.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        # --------------------------------------------------------------------------------------------------------
        # -----------------  classifier plot - trained on PAL1/FR1 encoding/retrieval tested on encoding -
        combined_cumulative_summary = self.get_passed_object('combined_cumulative_summary')
        panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, title='', labelsize=18)

        pd1 = PlotData(x=combined_cumulative_summary.fpr, y=combined_cumulative_summary.tpr, xlim=[0.0, 1.0], ylim=[0.0, 1.0],
                       xlabel='False Alarm Rate\n(a)', ylabel='Hit Rate', levelline=((0.001, 0.999), (0.001, 0.999)),
                       color='k', markersize=1.0, xlabel_fontsize=18, ylabel_fontsize=18)

        ylim = np.max(np.abs(combined_cumulative_summary.pc_diff_from_mean)) + 5.0
        if ylim > 100.0:
            ylim = 100.0
        pd2 = BarPlotData(x=(0, 1, 2), y=combined_cumulative_summary.pc_diff_from_mean, ylim=[-ylim, ylim],
                          xlabel='Tercile of Classifier Estimate\n(b)', ylabel='Recall Change From Mean (%)',
                          x_tick_labels=['Low', 'Middle', 'High'], xhline_pos=0.0, barcolors=['grey', 'grey', 'grey'],
                          xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)

        panel_plot.add_plot_data(0, 0, plot_data=pd1)
        panel_plot.add_plot_data(0, 1, plot_data=pd2)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace(
            'reports/' + task + '-' + subject + '-roc_and_terc_plot_fr_pal_test_enc.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        # --------------------------------------------------------------------------------------------------------
        # -----------------  classifier plot - trained on PAL1/FR1 encoding/retrieval tested on encoding -
        pal_cumulative_summary = self.get_passed_object('pal_cumulative_summary')
        panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, title='', labelsize=18)

        pd1 = PlotData(x=pal_cumulative_summary.fpr, y=pal_cumulative_summary.tpr, xlim=[0.0, 1.0], ylim=[0.0, 1.0],
                       xlabel='False Alarm Rate\n(a)', ylabel='Hit Rate', levelline=((0.001, 0.999), (0.001, 0.999)),
                       color='k', markersize=1.0, xlabel_fontsize=18, ylabel_fontsize=18)

        ylim = np.max(np.abs(pal_cumulative_summary.pc_diff_from_mean)) + 5.0
        if ylim > 100.0:
            ylim = 100.0
        pd2 = BarPlotData(x=(0, 1, 2), y=pal_cumulative_summary.pc_diff_from_mean, ylim=[-ylim, ylim],
                          xlabel='Tercile of Classifier Estimate\n(b)', ylabel='Recall Change From Mean (%)',
                          x_tick_labels=['Low', 'Middle', 'High'], xhline_pos=0.0, barcolors=['grey', 'grey', 'grey'],
                          xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)

        panel_plot.add_plot_data(0, 0, plot_data=pd1)
        panel_plot.add_plot_data(0, 1, plot_data=pd2)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace(
            'reports/' + task + '-' + subject + '-roc_and_terc_plot_pal_test_enc.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


class GenerateReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GenerateReportPDF, self).__init__(mark_as_completed)

    def run(self):
        output_directory = self.get_path_to_resource_in_workspace('reports')

        texinputs_set_str = r'export TEXINPUTS="' + output_directory + '":$TEXINPUTS;'

        # report_tex_file_names = self.get_passed_object('report_tex_file_names')
        # for f in report_tex_file_names:
        #     # + '/Library/TeX/texbin/pdflatex '\
        #     pdflatex_command_str = texinputs_set_str \
        #                            + 'module load Tex; pdflatex '\
        #                            + ' -output-directory '+output_directory\
        #                            + ' -shell-escape ' \
        #                            + self.get_path_to_resource_in_workspace('reports/'+f)
        #
        #     call([pdflatex_command_str], shell=True)

        combined_report_tex_file_name = self.get_passed_object('combined_report_tex_file_name')

        pdflatex_command_str = texinputs_set_str \
                               + 'module load Tex; pdflatex ' \
                               + ' -output-directory ' + output_directory \
                               + ' -shell-escape ' \
                               + self.get_path_to_resource_in_workspace('reports/' + combined_report_tex_file_name)

        call([pdflatex_command_str], shell=True)

        report_core_file_name, ext = splitext(combined_report_tex_file_name)
        report_file = join(output_directory, report_core_file_name + '.pdf')
        self.pass_object('report_file', report_file)


class DeployReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(DeployReportPDF, self).__init__(mark_as_completed)

    def run(self):
        report_file = self.get_passed_object('report_file')
        self.pipeline.deploy_report(report_path=report_file)
        SME_file = self.get_passed_object('SME_file')

        rd = ReportDeployer(pipeline=self.pipeline)
        ssc = rd.split_subject_code(self.pipeline.subject)
        report_base_dir = rd.report_base_dir(ssc)
        shutil.copyfile(SME_file, os.path.join(self.pipeline.mount_point, report_base_dir, os.path.basename(SME_file)))
