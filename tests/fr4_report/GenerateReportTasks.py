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
        subject = self.pipeline.subject
        task = self.pipeline.task

        tex_session_template = task + '_session.tex.tpl'

        n_sess = self.get_passed_object('NUMBER_OF_SESSIONS')
        n_bps = self.get_passed_object('NUMBER_OF_ELECTRODES')

        session_summary_array = self.get_passed_object('session_summary_array')

        report_tex_file_names = []
        for i_sess in xrange(n_sess):
            session_summary = session_summary_array[i_sess]
            sess = session_summary.number
            report_tex_file_name = '%s-%s-s%02d-report.tex' % (task,subject,sess)
            report_tex_file_names.append(report_tex_file_name)

            self.set_file_resources_to_move(report_tex_file_name, dst='reports')

            replace_dict = {'<PROB_RECALL_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-prob_recall_plot_' + session_summary.name + '.pdf',
                            '<STIM_AND_RECALL_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-stim_and_recall_plot_' + session_summary.name + '.pdf',
                            '<IRT_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-irt_plot_' + session_summary.name + '.pdf',
                            '<DATE>': datetime.date.today(),
                            '<SUBJECT>': subject.replace('_','\\textunderscore'),
                            '<NUMBER_OF_ELECTRODES>': n_bps,
                            '<SESSION_NUMBER>': sess,
                            '<SESSION_DATE>': session_summary.date,
                            '<SESSION_LENGTH>': session_summary.length,
                            '<N_LISTS>': session_summary.n_lists,
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
                            '<PC_FROM_NONSTIM_INTR>': '%.2f' % session_summary.pc_from_nonstim_intr,
                            '<CHISQR_INTR>': '%.2f' % session_summary.chisqr_intr,
                            '<PVALUE_INTR>': '%.2f' % session_summary.pvalue_intr
                            }

            TextTemplateUtils.replace_template(template_file_name=tex_session_template, out_file_name=report_tex_file_name, replace_dict=replace_dict)

        self.pass_object('report_tex_file_names', report_tex_file_names)

        tex_combined_template = task + '_combined.tex.tpl'
        combined_report_tex_file_name = '%s-%s-combined-report.tex' % (task,subject)

        self.set_file_resources_to_move(combined_report_tex_file_name, dst='reports')

        cumulative_summary = self.get_passed_object('cumulative_summary')

        cumulative_data_tex_table = latex_table(self.get_passed_object('SESSION_DATA'))

        replace_dict = {'<PROB_RECALL_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-prob_recall_plot_combined.pdf',
                        '<STIM_AND_RECALL_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-stim_and_recall_plot_combined.pdf',
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
                        '<N_CORRECT_STIM>': cumulative_summary.n_correct_stim,
                        '<N_TOTAL_STIM>': cumulative_summary.n_total_stim,
                        '<PC_FROM_STIM>': '%.2f' % cumulative_summary.pc_from_stim,
                        '<N_CORRECT_NONSTIM>': cumulative_summary.n_correct_nonstim,
                        '<N_TOTAL_NONSTIM>': cumulative_summary.n_total_nonstim,
                        '<PC_FROM_NONSTIM>': '%.2f' % cumulative_summary.pc_from_nonstim,
                        '<CHISQR>': '%.2f' % cumulative_summary.chisqr,
                        '<PVALUE>': '%.2f' % cumulative_summary.pvalue,
                        '<N_STIM_INTR>': cumulative_summary.n_stim_intr,
                        '<PC_FROM_STIM_INTR>': '%.2f' % cumulative_summary.pc_from_stim_intr,
                        '<N_NONSTIM_INTR>': cumulative_summary.n_nonstim_intr,
                        '<PC_FROM_NONSTIM_INTR>': '%.2f' % cumulative_summary.pc_from_nonstim_intr,
                        '<CHISQR_INTR>': '%.2f' % cumulative_summary.chisqr_intr,
                        '<PVALUE_INTR>': '%.2f' % cumulative_summary.pvalue_intr
                        }

        TextTemplateUtils.replace_template(template_file_name=tex_combined_template, out_file_name=combined_report_tex_file_name, replace_dict=replace_dict)

        self.pass_object('combined_report_tex_file_name', combined_report_tex_file_name)


class GeneratePlots(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GeneratePlots,self).__init__(mark_as_completed)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.create_dir_in_workspace('reports')

        session_summary_array = self.get_passed_object('session_summary_array')

        serial_positions = np.arange(1,13)



        combined_stim_pd_list = []
        combined_nostim_pd_list = []
        combined_number_stim_pd_list = []

        session_separator_pos = np.array([],dtype=np.float)

        combined_stim_x = np.array([],dtype=np.int)
        combined_nostim_x = np.array([], dtype=np.int)


        combined_stim_y = np.array([], dtype=np.int)
        combined_nostim_y = np.array([], dtype=np.int)


        combined_list_number_label = np.array([],dtype='|S32')
        combined_number_stims = np.array([], dtype=np.int)

        pos_counter = 0
        for session_summary in session_summary_array:
            panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, title='', wspace=0.3, hspace=0.3, labelsize=20)

            pd1 = PlotData(x=serial_positions, y=session_summary.prob_recall, xlim=(0,12), ylim=(0.0, 1.0), xlabel='Serial position\n(a)', ylabel='Probability of recall')
            pd1.xlabel_fontsize = 20
            pd1.ylabel_fontsize = 20
            pd2 = PlotData(x=serial_positions, y=session_summary.prob_first_recall, xlim=(0,12), ylim=(0.0, 1.0), xlabel='Serial position\n(b)', ylabel='Probability of first recall')
            pd2.xlabel_fontsize = 20
            pd2.ylabel_fontsize = 20

            # panel_plot.add_plot_data(0, 0, plot_data=pd1)
            # panel_plot.add_plot_data(0, 1, plot_data=pd2)
            #
            # plot = panel_plot.generate_plot()

            panel_plot.add_plot_data(0, 0, plot_data=pd1)

            panel_plot.add_plot_data(0, 1, plot_data=pd2)

            plot = panel_plot.generate_plot()

            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-prob_recall_plot_' + session_summary.name + '.pdf')

            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

            if task == 'RAM_CatFR1':
                panel_plot = PanelPlot(xfigsize=6.0, yfigsize=6.0, i_max=1, j_max=1, title='', wspace=0.3, hspace=0.3)
                pd = BarPlotData(x=[0,1], y=[session_summary.irt_within_cat,  session_summary.irt_between_cat], ylabel='IRT (msec)', x_tick_labels=['Within Cat', 'Between Cat'], barcolors=['grey','grey'], barwidth=0.5)
                panel_plot.add_plot_data(0, 0, plot_data=pd)
                plot = panel_plot.generate_plot()
                plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-irt_plot_' + session_summary.name + '.pdf')
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

            bpd_1 = BarPlotData(x=np.arange(n_lists), y=session_summary.n_stims_per_list, title='', alpha=0.2)
            stim_x = np.where(session_summary.is_stim_list)[0]
            stim_y = session_summary.n_recalls_per_list[session_summary.is_stim_list]
            pd_1 = PlotData(x=stim_x, y=stim_y, ylim=(0,12),
                    title='', linestyle='', color='red', marker='o',markersize=20)

            nostim_x = np.where(~session_summary.is_stim_list)[0]
            nostim_y = session_summary.n_recalls_per_list[~session_summary.is_stim_list]
            pd_2 = PlotData(x=nostim_x , y=nostim_y , ylim=(0,12),
                    title='', linestyle='', color='blue', marker='o',markersize=20)

            combined_number_stim_pd_list.append(bpd_1)
            combined_stim_pd_list.append(pd_1)
            combined_nostim_pd_list.append(pd_2)

            pdc.add_plot_data(pd_1)
            pdc.add_plot_data(pd_2)
            pdc.add_plot_data(bpd_1)

            panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

            plot = panel_plot.generate_plot()



            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-stim_and_recall_plot_' + session_summary.name + '.pdf')

            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

            # combined session  summary

            # labels = np.array(['']*len(bpd_1.x),dtype='|S32')

            # combined_list_number_label = np.hstack((combined_list_number_label, labels))
            combined_list_number_label = np.hstack((combined_list_number_label, np.arange(len(bpd_1.x))))

            combined_list_number_label[1::5]=''
            combined_list_number_label[2::5] = ''
            combined_list_number_label[3::5] = ''
            combined_list_number_label[4::5] = ''

            combined_stim_x = np.hstack((combined_stim_x,stim_x+pos_counter))
            combined_nostim_x = np.hstack((combined_nostim_x, nostim_x + pos_counter))

            combined_number_stims = np.hstack((combined_number_stims,session_summary.n_stims_per_list))

            combined_stim_y = np.hstack((combined_stim_y, stim_y))
            combined_nostim_y = np.hstack((combined_nostim_y, nostim_y))

            session_separator_pos = np.hstack((session_separator_pos,np.array([len(combined_list_number_label)-0.5],dtype=np.float)))


            pos_counter+=n_lists

            # end of combined session  summary

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # combined plot over sessions

        # empirical size of the figure based on the number of lists
        xfigsize = 7*len(combined_list_number_label)/25.0

        panel_plot_combined = PanelPlot(xfigsize=xfigsize, yfigsize=10.0, i_max=1, j_max=1, title='', xlabel='List',
                               ylabel='# of items', labelsize=20)

        bpd_combined = BarPlotData(x=np.arange(len(combined_list_number_label)), y=combined_number_stims, x_tick_labels=combined_list_number_label , title='', alpha=0.2)


        stim_pd_combined = PlotData(x=combined_stim_x,
                        y=combined_stim_y, ylim=(0, 12),
                        title='', linestyle='', color='red', marker='o', markersize=12)

        nostim_pd_combined = PlotData(x=combined_nostim_x,
                                    y=combined_nostim_y, ylim=(0, 12),
                                    title='', linestyle='', color='blue', marker='o', markersize=12)

        pdc_combined = PlotDataCollection()
        # ----------------- FORMATTING
        pdc_combined.xlabel = 'List number'
        pdc_combined.xlabel_fontsize = 20
        pdc_combined.ylabel = '#items'
        pdc_combined.ylabel_fontsize = 20

        n_lists = len(session_summary.n_stims_per_list)

        print 'Number of lists', n_lists

        pdc_combined.add_plot_data(stim_pd_combined)
        pdc_combined.add_plot_data(nostim_pd_combined)
        pdc_combined.add_plot_data(bpd_combined)

        for separator_pos  in session_separator_pos:

            x = np.arange(len(combined_list_number_label))
            y = [0]*len(x)
            sep_plot_data = PlotData(x=[0],y=[0],levelline=[[separator_pos, separator_pos], [0, 12]], color='white', alpha=0.0)
            pdc_combined.add_plot_data(sep_plot_data)



        panel_plot_combined.add_plot_data_collection(0, 0, plot_data_collection=pdc_combined)
        #
        plot_combined = panel_plot_combined.generate_plot()

        # print plot_combined

        # fig, ax = plot_combined.subplots()

        # for label in plot_combined.axes().xaxis.get_ticklabels():
        #     print label

        # for label in ax.get_ticklabels()[::2]:
        #     label.set_visible(False)

        #
        plot_out_fname = self.get_path_to_resource_in_workspace(
            'reports/' + task + '-' + subject + '-stim_and_recall_plot_combined.pdf')
        #
        plot_combined.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        # end of combined plot over sessions
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        cumulative_summary = self.get_passed_object('cumulative_summary')

        panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, title='', wspace=0.3, hspace=0.3)

        pd1 = PlotData(x=serial_positions, y=cumulative_summary.prob_recall, xlim=(0, 12), ylim=(0.0, 1.0), xlabel='Serial position\n(a)', ylabel='Probability of recall')
        pd1.xlabel_fontsize = 20
        pd1.ylabel_fontsize = 20
        pd2 = PlotData(x=serial_positions, y=cumulative_summary.prob_first_recall, xlim=(0, 12), ylim=(0.0, 1.0), xlabel='Serial position\n(b)', ylabel='Probability of first recall')
        pd2.xlabel_fontsize = 20
        pd2.ylabel_fontsize = 20

        panel_plot.add_plot_data(0, 0, plot_data=pd1)
        panel_plot.add_plot_data(0, 1, plot_data=pd2)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-prob_recall_plot_combined.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        if task == 'RAM_CatFR1':
            panel_plot = PanelPlot(xfigsize=6.0, yfigsize=6.0, i_max=1, j_max=1, title='', wspace=0.3, hspace=0.3)
            pd = BarPlotData(x=[0,1], y=[cumulative_summary.irt_within_cat, cumulative_summary.irt_between_cat], ylabel='IRT (msec)', x_tick_labels=['Within Cat', 'Between Cat'], barcolors=['grey','grey'], barwidth=0.5)
            panel_plot.add_plot_data(0, 0, plot_data=pd)
            plot = panel_plot.generate_plot()
            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-irt_plot_combined.pdf')
            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


class GenerateReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GenerateReportPDF,self).__init__(mark_as_completed)

    def run(self):
        output_directory = self.get_path_to_resource_in_workspace('reports')

        texinputs_set_str = r'export TEXINPUTS="' + output_directory + '":$TEXINPUTS;'

        report_tex_file_names = self.get_passed_object('report_tex_file_names')
        for f in report_tex_file_names:
            # + '/Library/TeX/texbin/pdflatex '\
            pdflatex_command_str = texinputs_set_str \
                                   + 'module load Tex; pdflatex '\
                                   + ' -output-directory '+output_directory\
                                   + ' -shell-escape ' \
                                   + self.get_path_to_resource_in_workspace('reports/'+f)

            call([pdflatex_command_str], shell=True)

        combined_report_tex_file_name = self.get_passed_object('combined_report_tex_file_name')

        pdflatex_command_str = texinputs_set_str \
                               + 'module load Tex; pdflatex '\
                               + ' -output-directory '+output_directory\
                               + ' -shell-escape ' \
                               + self.get_path_to_resource_in_workspace('reports/'+combined_report_tex_file_name)

        call([pdflatex_command_str], shell=True)
