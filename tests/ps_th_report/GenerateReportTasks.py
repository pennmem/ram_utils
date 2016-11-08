__author__ = 'm'

from RamPipeline import *

import datetime
import numpy as np

from PlotUtils import PlotData, BarPlotData, PlotDataCollection, PanelPlot
import TextTemplateUtils

from latex_table import latex_table
from ReportUtils import ReportRamTask

import re
from collections import namedtuple
SplitSubjectCode = namedtuple(typename='SplitSubjectCode',field_names=['protocol','id','site','montage'])
import os
import shutil


def pvalue_formatting(p):
    return '\leq 0.001' if p<=0.001 else ('%.3f'%p)


class GenerateTex(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GenerateTex,self).__init__(mark_as_completed)

    def run(self):
        tex_template = 'ps_report.tex.tpl'
        tex_session_template = 'ps_session.tex.tpl'
        tex_sham_plots_template = 'ps_sham_plots.tex.tpl'
        tex_nosham_plots_template = 'ps_nosham_plots.tex.tpl'
        tex_ttest_table1_template = 'ttest_table1.tex.tpl'
        tex_ttest_table2_template = 'ttest_table2.tex.tpl'
        tex_ttest_against_sham_template = 'ttest_against_sham.tex.tpl'

        report_tex_file_name = self.pipeline.experiment + '-TH1-' + self.pipeline.subject + '-' + 'report.tex'
        self.pass_object('report_tex_file_name',report_tex_file_name)

        self.set_file_resources_to_move(report_tex_file_name, dst='reports')

        param1_name = self.get_passed_object('param1_name')
        param1_unit = self.get_passed_object('param1_unit')

        param2_name = self.get_passed_object('param2_name')
        param2_unit = self.get_passed_object('param2_unit')

        const_param_name = self.get_passed_object('const_param_name')
        const_unit = self.get_passed_object('const_unit')

        session_summary_array = self.get_passed_object('session_summary_array')

        tex_session_pages_str = ''

        for session_summary in session_summary_array:
            n_ttest_tables = 0

            param1_ttest_table = ''
            if session_summary.param1_ttest_table is not None:
                ttest_replace_dict = {'<PARAMETER>': param1_name,
                                      '<UNIT>': param1_unit,
                                      '<TABLE>': latex_table(session_summary.param1_ttest_table, hlines=False)
                                     }
                param1_ttest_table = TextTemplateUtils.replace_template_to_string(tex_ttest_table1_template, ttest_replace_dict)
                n_ttest_tables += 1

            param2_ttest_table = ''
            if session_summary.param2_ttest_table is not None:
                ttest_replace_dict = {'<PARAMETER>': param2_name,
                                      '<UNIT>': param2_unit,
                                      '<TABLE>': latex_table(session_summary.param2_ttest_table, hlines=False)
                                      }
                param2_ttest_table = TextTemplateUtils.replace_template_to_string(tex_ttest_table1_template, ttest_replace_dict)
                n_ttest_tables += 1

            param12_ttest_table = ''
            if session_summary.param12_ttest_table is not None:
                ttest_replace_dict = {'<PARAMETER1>': param1_name,
                                      '<UNIT1>': param1_unit,
                                      '<PARAMETER2>': param2_name,
                                      '<UNIT2>': param2_unit,
                                      '<TABLE>': latex_table(session_summary.param12_ttest_table, hlines=False)
                                      }
                param12_ttest_table = TextTemplateUtils.replace_template_to_string(tex_ttest_table2_template, ttest_replace_dict)
                n_ttest_tables += 1

            adhoc_page_title = ''
            if n_ttest_tables > 0:
                adhoc_page_title = '\n\\subsection*{\\hfil Post hoc significance analysis \\hfil}\n\n'

            n_significantly_above_zero_params = len(session_summary.ttest_against_zero_table)
            #ttest_against_zero_title = '\n\\subsection*{\\hfil $t$-test against zero \\hfil}\n\n' if n_significantly_above_zero_params>0 else ''
            ttest_against_zero_table = '\n{\em No significant parameters for $t$-test against zero}.\n'
            if n_significantly_above_zero_params > 0:
                ttest_replace_dict = {'<PARAMETER1>': param1_name,
                                      '<UNIT1>': param1_unit,
                                      '<PARAMETER2>': param2_name,
                                      '<UNIT2>': param2_unit,
                                      '<TABLE>': latex_table(session_summary.ttest_against_zero_table, hlines=False)
                                      }
                ttest_against_zero_table = TextTemplateUtils.replace_template_to_string(tex_ttest_table2_template, ttest_replace_dict)

            n_significantly_above_sham_params = len(session_summary.ttest_against_sham_table) if session_summary.ttest_against_sham_table is not None else 0
            ttest_against_sham_table = '' if self.pipeline.experiment!='PS2.1' else '\n{\em No significant parameters for $t$-test against sham for the lower half}.\n\\vspace{1pc}\n'
            if n_significantly_above_sham_params > 0:
                ttest_replace_dict = {'<PARAMETER1>': param1_name,
                                      '<UNIT1>': param1_unit,
                                      '<PARAMETER2>': param2_name,
                                      '<UNIT2>': param2_unit,
                                      '<TABLE>': latex_table(session_summary.ttest_against_sham_table, hlines=False)
                                      }
                ttest_against_sham_table = TextTemplateUtils.replace_template_to_string(tex_ttest_against_sham_template, ttest_replace_dict)

            plot_replace_dict = {'<LOW_PLOT_FILE>': self.pipeline.experiment + '-' + self.pipeline.subject + '-low_plot_' + session_summary.stimtag + '.pdf',
                                '<ALL_PLOT_FILE>': self.pipeline.experiment + '-' + self.pipeline.subject + '-all_plot_' + session_summary.stimtag + '.pdf'}

            ps_plots = TextTemplateUtils.replace_template_to_string(tex_sham_plots_template if self.pipeline.experiment=='PS2.1' else tex_nosham_plots_template, plot_replace_dict)

            replace_dict = {'<PS_PLOTS>': ps_plots,
                            '<STIMTAG>': session_summary.stimtag,
                            '<REGION>': session_summary.region_of_interest,
                            '<SESSIONS>': session_summary.sessions,
                            '<CONSTANT_NAME>': const_param_name,
                            '<CONSTANT_VALUE>': session_summary.const_param_value,
                            '<CONSTANT_UNIT>': const_unit,
                            '<ISI_MID>': session_summary.isi_mid,
                            '<ISI_HALF_RANGE>': session_summary.isi_half_range,
                            '<PARAMETER1>': param1_name,
                            '<PARAMETER2>': param2_name,
                            '<FVALUE1>': '%.2f' % session_summary.anova_fvalues[0],
                            '<FVALUE2>': '%.2f' % session_summary.anova_fvalues[1],
                            '<FVALUE12>': '%.2f' % session_summary.anova_fvalues[2],
                            '<PVALUE1>': pvalue_formatting(session_summary.anova_pvalues[0]),
                            '<PVALUE2>': pvalue_formatting(session_summary.anova_pvalues[1]),
                            '<PVALUE12>': pvalue_formatting(session_summary.anova_pvalues[2]),
                            '<TTEST_AGAINST_SHAM_TABLE>': ttest_against_sham_table,
                            '<TTEST_AGAINST_ZERO_TABLE>': ttest_against_zero_table,
                            '<ADHOC_PAGE_TITLE>': adhoc_page_title,
                            '<PARAM1_TTEST_TABLE>': param1_ttest_table,
                            '<PARAM2_TTEST_TABLE>': param2_ttest_table,
                            '<PARAM12_TTEST_TABLE>': param12_ttest_table
                            }

            tex_session_pages_str += TextTemplateUtils.replace_template_to_string(tex_session_template, replace_dict)
            tex_session_pages_str += '\n'


        session_data_tex_table = latex_table(self.get_passed_object('SESSION_DATA'))

        xval_output = self.get_passed_object('xval_output')
        perm_test_pvalue = self.get_passed_object('pvalue')

        replace_dict = {
            '<SUBJECT>': self.pipeline.subject.replace('_', '\\textunderscore'),
            '<EXPERIMENT>': self.pipeline.experiment,
            '<DATE>': datetime.date.today(),
            '<SESSION_DATA>': session_data_tex_table,
            '<NUMBER_OF_SESSIONS>': self.get_passed_object('NUMBER_OF_SESSIONS'),
            '<NUMBER_OF_ELECTRODES>': self.get_passed_object('NUMBER_OF_ELECTRODES'),
            '<REPORT_PAGES>': tex_session_pages_str,
            '<AUC>': '%.2f' % (100*xval_output[-1].auc),
            '<PERM-P-VALUE>': pvalue_formatting(perm_test_pvalue),
            '<ROC_AND_TERC_PLOT_FILE>': self.pipeline.subject + '-roc_and_terc_plot_combined.pdf'
        }

        TextTemplateUtils.replace_template(template_file_name=tex_template, out_file_name=report_tex_file_name, replace_dict=replace_dict)




class GeneratePlots(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GeneratePlots,self).__init__(mark_as_completed)

    def run(self):
        self.create_dir_in_workspace('reports')

        xval_output = self.get_passed_object('xval_output')
        th1_summary = xval_output[-1]

        panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, labelsize=16, wspace=5.0)

        pd1 = PlotData(x=th1_summary.fpr, y=th1_summary.tpr, xlim=[0.0,1.0], ylim=[0.0,1.0], xlabel='False Alarm Rate\n(a)', ylabel='Hit Rate', xlabel_fontsize=20, ylabel_fontsize=20, levelline=((0.0,1.0),(0.0,1.0)), color='k', markersize=1.0)

        pc_diff_from_mean = (th1_summary.low_pc_diff_from_mean, th1_summary.mid_pc_diff_from_mean, th1_summary.high_pc_diff_from_mean)

        ylim = np.max(np.abs(pc_diff_from_mean)) + 5.0
        if ylim > 100.0:
            ylim = 100.0
        pd2 = BarPlotData(x=(0,1,2), y=pc_diff_from_mean, ylim=[-ylim,ylim], xlabel='Tercile of Classifier Estimate\n(b)', ylabel='Recall Change From Mean (%)', x_tick_labels=['Low', 'Middle', 'High'], xlabel_fontsize=20, ylabel_fontsize=20, xhline_pos=0.0, barcolors=['grey','grey', 'grey'], barwidth=0.5)

        panel_plot.add_plot_data(0, 0, plot_data=pd1)
        panel_plot.add_plot_data(0, 1, plot_data=pd2)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + self.pipeline.subject + '-roc_and_terc_plot_combined.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        session_summary_array = self.get_passed_object('session_summary_array')
        param1_name = self.get_passed_object('param1_name')
        param1_unit = self.get_passed_object('param1_unit')
        param1_title = '%s (%s)' % (param1_name,param1_unit)


        for session_summary in session_summary_array:
            panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=2, labelsize=16, wspace=5.0)

            pdc = PlotDataCollection(legend_on=True, legend_loc=3, xlabel=param1_title+'\n(a)', ylabel='$\Delta$ Post-Pre Classifier Output', xlabel_fontsize=20, ylabel_fontsize=20)
            for v,p in session_summary.low_classifier_delta_plot.iteritems():
                p.xhline_pos=0.0
                pdc.add_plot_data(p)

            panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

            pdc = PlotDataCollection(legend_on=True, legend_loc=3, xlabel=param1_title+'\n(b)', ylabel='Expected Recall Change (%)', xlabel_fontsize=20, ylabel_fontsize=20)
            for v,p in session_summary.low_recall_delta_plot.iteritems():
                p.xhline_pos=0.0
                pdc.add_plot_data(p)

            panel_plot.add_plot_data_collection(0, 1, plot_data_collection=pdc)

            plot = panel_plot.generate_plot()

            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + self.pipeline.experiment + '-' + self.pipeline.subject + '-low_plot_' + session_summary.stimtag + '.pdf')

            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


            panel_plot = PanelPlot(xfigsize=16, yfigsize=6.5, i_max=1, j_max=2, labelsize=16, wspace=5.0)

            pdc = PlotDataCollection(legend_on=True, legend_loc=3, xlabel=param1_title+'\n(a)', ylabel='$\Delta$ Post-Pre Classifier Output', xlabel_fontsize=20, ylabel_fontsize=20)
            for v,p in session_summary.all_classifier_delta_plot.iteritems():
                p.xhline_pos=0.0
                pdc.add_plot_data(p)

            panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

            pdc = PlotDataCollection(legend_on=True, legend_loc=3, xlabel=param1_title+'\n(b)', ylabel='Expected Recall Change (%)', xlabel_fontsize=20, ylabel_fontsize=20)
            for v,p in session_summary.all_recall_delta_plot.iteritems():
                p.xhline_pos=0.0
                pdc.add_plot_data(p)

            panel_plot.add_plot_data_collection(0, 1, plot_data_collection=pdc)

            plot = panel_plot.generate_plot()

            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + self.pipeline.experiment + '-' + self.pipeline.subject + '-all_plot_' + session_summary.stimtag + '.pdf')

            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


class GenerateReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GenerateReportPDF,self).__init__(mark_as_completed)


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
        report_file = join(output_directory,report_core_file_name+'.pdf')

        self.pass_object('report_file', report_file)

class DeployReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(DeployReportPDF, self).__init__(mark_as_completed)

    def run(self):
        report_file = self.get_passed_object('report_file')
        self.pipeline.deploy_report(report_path=report_file, classifier_experiment='TH1',suffix='json')

# class DeployReportPDF(ReportRamTask):
#     def __init__(self, mark_as_completed=True):
#         super(DeployReportPDF,self).__init__(mark_as_completed)
#
#         self.protocol = 'R1'
#         self.convert_subject_code_regex = re.compile('('+self.protocol+')'+'([0-9]*)([a-zA-Z]{1,1})([\S]*)')
#
#     def split_subject_code(self,subject_code):
#         match = re.match(self.convert_subject_code_regex,subject_code)
#         if match:
#             groups = match.groups()
#
#             ssc = SplitSubjectCode(protocol=groups[0], id=groups[1],site=groups[2],montage=groups[3])
#             return ssc
#         return None
#
#
#     def deploy_report(self,report_path):
#         subject = self.pipeline.subject
#
#         ssc = self.split_subject_code(subject)
#
#         report_basename = basename(report_path)
#         report_base_dir = join('protocols',ssc.protocol.lower(),'subjects',str(ssc.id),'reports')
#
#         report_dir = join(self.pipeline.mount_point,report_base_dir)
#
#         if not isdir(report_dir):
#             try:
#                 os.makedirs(report_dir)
#             except OSError:
#                 return
#
#         standard_report_basename = subject+'_RAM_'+self.pipeline.experiment+'_report.pdf'
#         standard_report_path = join(report_dir,standard_report_basename)
#         # shutil.copy(report_path,join(report_dir,report_basename))
#         shutil.copy(report_path,standard_report_path)
#
#         self.add_report_file(file=standard_report_path)
#
#         standard_report_link = join(self.pipeline.report_site_URL, report_base_dir, standard_report_basename)
#         self.add_report_link(link=standard_report_link)
#
#
#
#     # def deploy_report(self,report_path):
#     #     subject = self.pipeline.subject
#     #
#     #     ssc = self.split_subject_code(subject)
#     #
#     #     report_basename = basename(report_path)
#     #     report_dir = join('/protocols',ssc.protocol.lower(),'subjects',str(ssc.id)+ssc.montage,'reports',self.pipeline.experiment)
#     #
#     #     if not isdir(report_dir):
#     #         try:
#     #             os.makedirs(report_dir)
#     #         except OSError:
#     #             return
#     #
#     #     standard_report_basename = subject+'_RAM_'+self.pipeline.experiment+'_report.pdf'
#     #     standard_report_path = join(report_dir,standard_report_basename)
#     #     # shutil.copy(report_path,join(report_dir,report_basename))
#     #     shutil.copy(report_path,standard_report_path)
#     #
#     #     self.add_report_file(file=standard_report_path)
#
#
#     def run(self):
#         report_file = self.get_passed_object('report_file')
#         self.deploy_report(report_path=report_file)
