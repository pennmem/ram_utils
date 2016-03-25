from RamPipeline import *

import TextTemplateUtils
from PlotUtils import PlotData, BarPlotData, PanelPlot
from latex_table import latex_table

import numpy as np
import datetime
from subprocess import call
from ReportUtils import ReportRamTask

import re
from collections import namedtuple
SplitSubjectCode = namedtuple(typename='SplitSubjectCode',field_names=['protocol','id','site','montage'])
import os
import shutil


class GenerateTex(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GenerateTex,self).__init__(mark_as_completed)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        # tex_session_template = task + '_session.tex.tpl'

        n_sess = self.get_passed_object('NUMBER_OF_SESSIONS')
        n_bps = self.get_passed_object('NUMBER_OF_ELECTRODES')

        # session_summary_array = self.get_passed_object('session_summary_array')
        # session_ttest_data = self.get_passed_object('session_ttest_data')

        ttable_format = self.get_passed_object('ttable_format')
        ttable_header = self.get_passed_object('ttable_header')

        # report_tex_file_names = []
        # for i_sess in xrange(n_sess):
        #     session_summary = session_summary_array[i_sess]
        #     sess = session_summary.number
        #     report_tex_file_name = '%s-%s-s%02d-report.tex' % (task,subject,sess)
        #     report_tex_file_names.append(report_tex_file_name)
        #
        #     self.set_file_resources_to_move(report_tex_file_name, dst='reports')
        #
        #     session_ttest_tex_table = latex_table(session_ttest_data[i_sess])
        #
        #     replace_dict = {'<PROB_RECALL_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-prob_recall_plot_' + session_summary.name + '.pdf',
        #                     '<IRT_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-irt_plot_' + session_summary.name + '.pdf',
        #                     '<DATE>': datetime.date.today(),
        #                     '<SUBJECT>': subject.replace('_','\\textunderscore'),
        #                     '<NUMBER_OF_ELECTRODES>': n_bps,
        #                     '<SESSION_NUMBER>': sess,
        #                     '<SESSION_DATE>': session_summary.date,
        #                     '<SESSION_LENGTH>': session_summary.length,
        #                     '<N_WORDS>': session_summary.n_words,
        #                     '<N_CORRECT_WORDS>': session_summary.n_correct_words,
        #                     '<PC_CORRECT_WORDS>': '%.2f' % session_summary.pc_correct_words,
        #                     '<N_PLI>': session_summary.n_pli,
        #                     '<PC_PLI>': '%.2f' % session_summary.pc_pli,
        #                     '<N_ELI>': session_summary.n_eli,
        #                     '<PC_ELI>': '%.2f' % session_summary.pc_eli,
        #                     '<N_MATH>': session_summary.n_math,
        #                     '<N_CORRECT_MATH>': session_summary.n_correct_math,
        #                     '<PC_CORRECT_MATH>': '%.2f' % session_summary.pc_correct_math,
        #                     '<MATH_PER_LIST>': '%.2f' % session_summary.math_per_list,
        #                     '<TABLE_FORMAT>': ttable_format,
        #                     '<TABLE_HEADER>': ttable_header,
        #                     '<SIGNIFICANT_ELECTRODES>': session_ttest_tex_table,
        #                     '<AUC>': session_summary.auc,
        #                     '<ROC_AND_TERC_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-roc_and_terc_plot_' + session_summary.name + '.pdf'
        #                     }
        #
        #     TextTemplateUtils.replace_template(template_file_name=tex_session_template, out_file_name=report_tex_file_name, replace_dict=replace_dict)
        #
        # self.pass_object('report_tex_file_names', report_tex_file_names)

        tex_combined_template = task + '_combined.tex.tpl'
        combined_report_tex_file_name = '%s_RAM_FR1_CatFR1_joined_report.tex' % subject

        self.set_file_resources_to_move(combined_report_tex_file_name, dst='reports')

        cumulative_summary = self.get_passed_object('cumulative_summary')

        cumulative_data_tex_table = latex_table(self.get_passed_object('SESSION_DATA'))

        cumulative_ttest_tex_table = latex_table(self.get_passed_object('cumulative_ttest_data'))

        replace_dict = {'<PROB_RECALL_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-prob_recall_plot_combined.pdf',
                        '<IRT_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-irt_plot_combined.pdf',
                        '<DATE>': datetime.date.today(),
                        '<SESSION_DATA>': cumulative_data_tex_table,
                        '<SUBJECT>': subject.replace('_','\\textunderscore'),
                        '<NUMBER_OF_SESSIONS>': n_sess,
                        '<NUMBER_OF_ELECTRODES>': n_bps,
                        '<N_WORDS>': cumulative_summary.n_words,
                        '<N_CORRECT_WORDS>': cumulative_summary.n_correct_words,
                        '<PC_CORRECT_WORDS>': '%.2f' % cumulative_summary.pc_correct_words,
                        '<N_PLI>': cumulative_summary.n_pli,
                        '<PC_PLI>': '%.2f' % cumulative_summary.pc_pli,
                        '<N_ELI>': cumulative_summary.n_eli,
                        '<PC_ELI>': '%.2f' % cumulative_summary.pc_eli,
                        '<N_MATH>': cumulative_summary.n_math,
                        '<N_CORRECT_MATH>': cumulative_summary.n_correct_math,
                        '<PC_CORRECT_MATH>': '%.2f' % cumulative_summary.pc_correct_math,
                        '<MATH_PER_LIST>': '%.2f' % cumulative_summary.math_per_list,
                        '<TABLE_FORMAT>': ttable_format,
                        '<TABLE_HEADER>': ttable_header,
                        '<SIGNIFICANT_ELECTRODES>': cumulative_ttest_tex_table,
                        '<AUC>': cumulative_summary.auc,
                        '<PERM-P-VALUE>': cumulative_summary.perm_test_pvalue,
                        '<J-THRESH>': cumulative_summary.jstat_thresh,
                        '<J-PERC>': cumulative_summary.jstat_percentile,
                        '<ROC_AND_TERC_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-roc_and_terc_plot_combined.pdf'
                        }

        TextTemplateUtils.replace_template(template_file_name=tex_combined_template, out_file_name=combined_report_tex_file_name, replace_dict=replace_dict)

        self.pass_object('combined_report_tex_file_name', combined_report_tex_file_name)


class GeneratePlots(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(ReportRamTask,self).__init__(mark_as_completed)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.create_dir_in_workspace('reports')

        # session_summary_array = self.get_passed_object('session_summary_array')

        serial_positions = np.arange(1,13)

        # for session_summary in session_summary_array:
        #     panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, labelsize=18, wspace=20.0)
        #
        #     pd1 = PlotData(x=serial_positions, y=session_summary.prob_recall, xlim=(0,12), ylim=(0.0, 1.0), xlabel='Serial position\n(a)', ylabel='Probability of recall', xlabel_fontsize=18, ylabel_fontsize=18)
        #     pd2 = PlotData(x=serial_positions, y=session_summary.prob_first_recall, xlim=(0,12), ylim=(0.0, 1.0), xlabel='Serial position\n(b)', ylabel='Probability of first recall', xlabel_fontsize=18, ylabel_fontsize=18)
        #
        #     panel_plot.add_plot_data(0, 0, plot_data=pd1)
        #     panel_plot.add_plot_data(0, 1, plot_data=pd2)
        #
        #     plot = panel_plot.generate_plot()
        #
        #     plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-prob_recall_plot_' + session_summary.name + '.pdf')
        #
        #     plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')
        #
        #     if task == 'RAM_CatFR1':
        #         panel_plot = PanelPlot(xfigsize=6.0, yfigsize=6.0, i_max=1, j_max=1, title='', xtitle='', labelsize=18)
        #         pd = BarPlotData(x=[0,1], y=[session_summary.irt_within_cat,  session_summary.irt_between_cat], ylabel='IRT (msec)',xlabel='', x_tick_labels=['Within Cat', 'Between Cat'], barcolors=['grey','grey'], barwidth=0.5, xlabel_fontsize=18, ylabel_fontsize=18)
        #         panel_plot.add_plot_data(0, 0, plot_data=pd)
        #         plot = panel_plot.generate_plot()
        #         plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-irt_plot_' + session_summary.name + '.pdf')
        #         plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')
        #
        #     panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, title='', labelsize=18)
        #
        #     pd1 = PlotData(x=session_summary.fpr, y=session_summary.tpr, xlim=[0.0,1.0], ylim=[0.0,1.0], xlabel='False Alarm Rate\n(a)', ylabel='Hit Rate', levelline=((0.0,1.0),(0.0,1.0)), color='k', markersize=1.0, xlabel_fontsize=18, ylabel_fontsize=18)
        #
        #     ylim = np.max(np.abs(session_summary.pc_diff_from_mean)) + 5.0
        #     if ylim > 100.0:
        #         ylim = 100.0
        #     pd2 = BarPlotData(x=(0,1,2), y=session_summary.pc_diff_from_mean, ylim=[-ylim,ylim], xlabel='Tercile of Classifier Estimate\n(b)', ylabel='Recall Change From Mean (%)', x_tick_labels=['Low', 'Middle', 'High'], xhline_pos=0.0, barcolors=['grey','grey', 'grey'], xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)
        #
        #     panel_plot.add_plot_data(0, 0, plot_data=pd1)
        #     panel_plot.add_plot_data(0, 1, plot_data=pd2)
        #
        #     plot = panel_plot.generate_plot()
        #
        #     plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-roc_and_terc_plot_' + session_summary.name + '.pdf')
        #
        #     plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        cumulative_summary = self.get_passed_object('cumulative_summary')

        panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, labelsize=18, wspace=20.0)

        pd1 = PlotData(x=serial_positions, y=cumulative_summary.prob_recall, xlim=(0, 12), ylim=(0.0, 1.0), xlabel='Serial position\n(a)', ylabel='Probability of recall', xlabel_fontsize=18, ylabel_fontsize=18)
        pd2 = PlotData(x=serial_positions, y=cumulative_summary.prob_first_recall, xlim=(0, 12), ylim=(0.0, 1.0), xlabel='Serial position\n(b)', ylabel='Probability of first recall', xlabel_fontsize=18, ylabel_fontsize=18)

        panel_plot.add_plot_data(0, 0, plot_data=pd1)
        panel_plot.add_plot_data(0, 1, plot_data=pd2)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-prob_recall_plot_combined.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        if task == 'RAM_CatFR1':
            panel_plot = PanelPlot(xfigsize=6.0, yfigsize=6.0, i_max=1, j_max=1, title='',xtitle='', labelsize=18)
            pd = BarPlotData(x=[0,1], y=[cumulative_summary.irt_within_cat, cumulative_summary.irt_between_cat], ylabel='IRT (msec)', xlabel='',x_tick_labels=['Within Cat', 'Between Cat'], barcolors=['grey','grey'], barwidth=0.5, xlabel_fontsize=18, ylabel_fontsize=18)
            panel_plot.add_plot_data(0, 0, plot_data=pd)
            plot = panel_plot.generate_plot()
            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-irt_plot_combined.pdf')
            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, title='', labelsize=18)

        pd1 = PlotData(x=cumulative_summary.fpr, y=cumulative_summary.tpr, xlim=[0.0,1.0], ylim=[0.0,1.0], xlabel='False Alarm Rate\n(a)', ylabel='Hit Rate', levelline=((0.001,0.999),(0.001,0.999)), color='k', markersize=1.0, xlabel_fontsize=18, ylabel_fontsize=18)

        ylim = np.max(np.abs(cumulative_summary.pc_diff_from_mean)) + 5.0
        if ylim > 100.0:
            ylim = 100.0
        pd2 = BarPlotData(x=(0,1,2), y=cumulative_summary.pc_diff_from_mean, ylim=[-ylim,ylim], xlabel='Tercile of Classifier Estimate\n(b)', ylabel='Recall Change From Mean (%)', x_tick_labels=['Low', 'Middle', 'High'], xhline_pos=0.0, barcolors=['grey','grey', 'grey'], xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)

        panel_plot.add_plot_data(0, 0, plot_data=pd1)
        panel_plot.add_plot_data(0, 1, plot_data=pd2)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-roc_and_terc_plot_combined.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


class GenerateReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GenerateReportPDF,self).__init__(mark_as_completed)

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
                               + 'module load Tex; pdflatex '\
                               + ' -output-directory '+output_directory\
                               + ' -shell-escape ' \
                               + self.get_path_to_resource_in_workspace('reports/'+combined_report_tex_file_name)

        call([pdflatex_command_str], shell=True)


        report_core_file_name, ext = splitext(combined_report_tex_file_name)
        report_file = join(output_directory,report_core_file_name+'.pdf')

        self.pass_object('report_file',report_file)



class DeployReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(DeployReportPDF,self).__init__(mark_as_completed)

        self.protocol = 'R1'
        self.convert_subject_code_regex = re.compile('('+self.protocol+')'+'([0-9]*)([a-zA-Z]{1,1})([\S]*)')

    def split_subject_code(self,subject_code):
        match = re.match(self.convert_subject_code_regex,subject_code)
        if match:
            groups = match.groups()

            ssc = SplitSubjectCode(protocol=groups[0], id=groups[1],site=groups[2],montage=groups[3])
            return ssc
        return None


    def deploy_report(self,report_path):
        subject = self.pipeline.subject

        ssc = self.split_subject_code(subject)

        report_basename = basename(report_path)
        report_dir = join('/protocols',ssc.protocol.lower(),'subjects',str(ssc.id)+ssc.montage,'reports',self.pipeline.experiment)

        if not isdir(report_dir):
            try:
                os.makedirs(report_dir)
            except OSError:
                return

        standard_report_basename = subject+'_'+self.pipeline.experiment+'_report.pdf'
        standard_report_path = join(report_dir,standard_report_basename)
        # shutil.copy(report_path,join(report_dir,report_basename))
        shutil.copy(report_path,standard_report_path)

        self.add_report_file(file=standard_report_path)


    def run(self):
        report_file = self.get_passed_object('report_file')
        self.deploy_report(report_path=report_file)
