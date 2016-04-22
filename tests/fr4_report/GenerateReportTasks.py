from RamPipeline import *

import TextTemplateUtils
from PlotUtils import PlotData, BarPlotData, PlotDataCollection, PanelPlot
from latex_table import latex_table

import numpy as np
import datetime
from subprocess import call

from ReportUtils import ReportRamTask


class GenerateTex(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GenerateTex,self).__init__(mark_as_completed)

    def run(self):
        tex_template = 'fr4_report.tex.tpl'
        tex_session_template = 'fr4_session.tex.tpl'

        report_tex_file_name = self.pipeline.task + '-' + self.pipeline.subject + '-report.tex'
        self.pass_object('report_tex_file_name',report_tex_file_name)

        self.set_file_resources_to_move(report_tex_file_name, dst='reports')

        n_sess = self.get_passed_object('NUMBER_OF_SESSIONS')
        n_elecs = self.get_passed_object('NUMBER_OF_ELECTRODES')

        session_summary_array = self.get_passed_object('session_summary_array')

        tex_session_pages_str = ''

        for session_summary in session_summary_array:
            replace_dict = {'<STIMTAG>': session_summary.stimtag,
                            '<REGION>': session_summary.region_of_interest,
                            '<FREQUENCY>': session_summary.frequency,
                            '<SESSIONS>': ','.join([str(s) for s in session_summary.sessions]),
                            '<PROB_RECALL_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-prob_recall_plot_' + session_summary.stimtag + '-' + str(session_summary.frequency) + '.pdf',
                            '<STIM_AND_RECALL_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-stim_and_recall_plot_' + session_summary.stimtag + '-' + str(session_summary.frequency) + '.pdf',
                            '<N_WORDS>': session_summary.n_words,
                            '<N_CORRECT_WORDS>': session_summary.n_correct_words,
                            '<PC_CORRECT_WORDS>': '%.2f' % session_summary.pc_correct_words,
                            '<N_PLI>': session_summary.n_pli,
                            '<PC_PLI>': '%.2f' % session_summary.pc_pli,
                            '<N_ELI>': session_summary.n_eli,
                            '<PC_ELI>': '%.2f' % session_summary.pc_eli,
                            '<N_MATH>': session_summary.n_math,
                            '<N_CORRECT_MATH>': session_summary.n_correct_math,
                            '<PC_CORRECT_MATH>': '%.2f' % session_summary.pc_correct_math,
                            '<MATH_PER_LIST>': '%.2f' % session_summary.math_per_list,
                            '<N_CORRECT_STIM>': session_summary.n_correct_stim,
                            '<N_TOTAL_STIM>': session_summary.n_total_stim,
                            '<PC_FROM_STIM>': '%.2f' % session_summary.pc_from_stim,
                            '<N_CORRECT_NONSTIM>': session_summary.n_correct_nonstim,
                            '<N_TOTAL_NONSTIM>': session_summary.n_total_nonstim,
                            '<PC_FROM_NONSTIM>': '%.2f' % session_summary.pc_from_nonstim,
                            '<CHISQR>': '%.2f' % session_summary.chisqr,
                            '<PVALUE>': '%.2f' % session_summary.pvalue,
                            '<N_STIM_INTR>': session_summary.n_stim_intr,
                            '<PC_FROM_STIM_INTR>': '%.2f' % session_summary.pc_from_stim_intr,
                            '<N_NONSTIM_INTR>': session_summary.n_nonstim_intr,
                            '<PC_FROM_NONSTIM_INTR>': '%.2f' % session_summary.pc_from_nonstim_intr
                            }

            tex_session_pages_str += TextTemplateUtils.replace_template_to_string(tex_session_template, replace_dict)
            tex_session_pages_str += '\n'

        session_data_tex_table = latex_table(self.get_passed_object('SESSION_DATA'))

        replace_dict = {'<DATE>': datetime.date.today(),
                        '<SESSION_DATA>': session_data_tex_table,
                        '<SUBJECT>': self.pipeline.subject.replace('_','\\textunderscore'),
                        '<NUMBER_OF_SESSIONS>': n_sess,
                        '<NUMBER_OF_ELECTRODES>': n_elecs,
                        '<REPORT_PAGES>': tex_session_pages_str
                        }

        TextTemplateUtils.replace_template(template_file_name=tex_template, out_file_name=report_tex_file_name, replace_dict=replace_dict)

        self.pass_object('report_tex_file_name', report_tex_file_name)


class GeneratePlots(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GeneratePlots,self).__init__(mark_as_completed)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.create_dir_in_workspace('reports')

        session_summary_array = self.get_passed_object('session_summary_array')

        serial_positions = np.arange(1,13)

        #session_separator_pos = np.array([],dtype=np.float)
        #pos_counter = 0

        for session_summary in session_summary_array:
            panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, title='', wspace=0.3, hspace=0.3, labelsize=20)

            pd1 = PlotData(x=serial_positions, y=session_summary.prob_recall, xlim=(0,12), ylim=(0.0, 1.0), xlabel='Serial position\n(a)', ylabel='Probability of recall')
            pd1.xlabel_fontsize = 20
            pd1.ylabel_fontsize = 20
            pd2 = PlotData(x=serial_positions, y=session_summary.prob_first_recall, xlim=(0,12), ylim=(0.0, 1.0), xlabel='Serial position\n(b)', ylabel='Probability of first recall')
            pd2.xlabel_fontsize = 20
            pd2.ylabel_fontsize = 20

            panel_plot.add_plot_data(0, 0, plot_data=pd1)

            panel_plot.add_plot_data(0, 1, plot_data=pd2)

            plot = panel_plot.generate_plot()

            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-prob_recall_plot_' + session_summary.stimtag + '-' + str(session_summary.frequency) + '.pdf')

            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

            panel_plot = PanelPlot(xfigsize=10.0, yfigsize=10.0, i_max=1, j_max=1, title='', xlabel='List', ylabel='# of items', labelsize=20)

            pdc = PlotDataCollection()
            #----------------- FORMATTING
            pdc.xlabel = 'List number'
            pdc.xlabel_fontsize = 20
            pdc.ylabel ='#items'
            pdc.ylabel_fontsize = 20

            n_lists = len(session_summary.n_stims_per_list)

            print 'Number of lists', n_lists

            bpd_1 = BarPlotData(x=np.arange(n_lists), y=session_summary.n_stims_per_list, title='', alpha=0.3)
            stim_x = np.where(session_summary.is_stim_list)[0]
            stim_y = session_summary.n_recalls_per_list[session_summary.is_stim_list]
            pd_1 = PlotData(x=stim_x, y=stim_y, ylim=(0,12),
                    title='', linestyle='', color='red', marker='o',markersize=20)

            nostim_x = np.where(~session_summary.is_stim_list)[0]
            nostim_y = session_summary.n_recalls_per_list[~session_summary.is_stim_list]
            pd_2 = PlotData(x=nostim_x , y=nostim_y , ylim=(0,12),
                    title='', linestyle='', color='blue', marker='o',markersize=20)

            pdc.add_plot_data(pd_1)
            pdc.add_plot_data(pd_2)
            pdc.add_plot_data(bpd_1)

            panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

            plot = panel_plot.generate_plot()

            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-stim_and_recall_plot_' + session_summary.stimtag + '-' + str(session_summary.frequency) + '.pdf')

            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


class GenerateReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GenerateReportPDF,self).__init__(mark_as_completed)

    def run(self):
        output_directory = self.get_path_to_resource_in_workspace('reports')

        texinputs_set_str = r'export TEXINPUTS="' + output_directory + '":$TEXINPUTS;'

        report_tex_file_name = self.get_passed_object('report_tex_file_name')

        pdflatex_command_str = texinputs_set_str \
                               + 'module load Tex; pdflatex '\
                               + ' -output-directory '+output_directory\
                               + ' -shell-escape ' \
                               + self.get_path_to_resource_in_workspace('reports/'+report_tex_file_name)

        call([pdflatex_command_str], shell=True)
