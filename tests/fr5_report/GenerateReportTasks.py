# -*- coding: utf-8 -*-

from RamPipeline import *

from PlotUtils import PlotData, BarPlotData, PlotDataCollection, PanelPlot

import numpy as np
import datetime
from subprocess import call

from ReportUtils import ReportRamTask
import jinja2

class GeneratePlots(ReportRamTask):
    def __init__(self):
        super(GeneratePlots,self).__init__(mark_as_completed=False)



    def run(self):
        self.create_dir_in_workspace('reports')
        task = self.pipeline.task
        subject= self.pipeline.subject


        ps_events = self.get_passed_object('ps_events')
        ps_sessions = np.unique(ps_events.session)
        ps4_session_summaries = self.get_passed_object('ps4_session_summaries')
        if ps4_session_summaries:
            for session in ps_sessions:

                session_summary = ps4_session_summaries[session]


                panel_plot  = PanelPlot(i_max = 2, j_max = 1)
                panel_plot.add_plot_data(0,1,x=session_summary.loc_1_amplitudes,y=session_summary.loc_1_delta_classifiers,
                                         linestyle='',marker='x',color='black', title=session_summary.LOC1)
                panel_plot.add_plot_data(1,1,x=session_summary.loc_2_amplitudes, y=session_summary.loc_2_delta_classifiers,
                                         linestyle = '',marker='x',color='black',title=session_summary.LOC2)
                plt = panel_plot.generate_plot()
                session_summary.PS_PLOT_FILE = self.get_path_to_resource_in_workspace('reports','PS4_%d_dc_plot.pdf'%session)
                plt.savefig(session_summary.PS_PLOT_FILE,dpi=300,bbox_inches='tight')


        fr5_events = self.get_passed_object('FR5_events')
        fr5_session_summaries = self.get_passed_object('fr5_session_summaries')
        if fr5_session_summaries:

            xval_output = self.get_passed_object('xval_output')
            fr1_summary = xval_output[-1]

            panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, labelsize=16, wspace=5.0)

            pd1 = PlotData(x=fr1_summary.fpr, y=fr1_summary.tpr, xlim=[0.0, 1.0], ylim=[0.0, 1.0],
                           xlabel='False Alarm Rate\n(a)', ylabel='Hit Rate', xlabel_fontsize=20, ylabel_fontsize=20,
                           levelline=((0.0, 1.0), (0.0, 1.0)), color='k', markersize=1.0)

            pc_diff_from_mean = (
            fr1_summary.low_pc_diff_from_mean, fr1_summary.mid_pc_diff_from_mean, fr1_summary.high_pc_diff_from_mean)

            ylim = np.max(np.abs(pc_diff_from_mean)) + 5.0
            if ylim > 100.0:
                ylim = 100.0
            pd2 = BarPlotData(x=(0, 1, 2), y=pc_diff_from_mean, ylim=[-ylim, ylim],
                              xlabel='Tercile of Classifier Estimate\n(b)', ylabel='Recall Change From Mean (%)',
                              x_tick_labels=['Low', 'Middle', 'High'], xlabel_fontsize=20, ylabel_fontsize=20,
                              xhline_pos=0.0, barcolors=['grey', 'grey', 'grey'], barwidth=0.5)

            panel_plot.add_plot_data(0, 0, plot_data=pd1)
            panel_plot.add_plot_data(0, 1, plot_data=pd2)

            plot = panel_plot.generate_plot()

            plot_out_fname = self.get_path_to_resource_in_workspace(
                'reports/' + self.pipeline.subject + '-roc_and_terc_plot.pdf')

            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

            sessions = np.unique(fr5_events.session)

            serial_positions = np.arange(1, 13)
            for session in sessions:

                session_summary = fr5_session_summaries[session]

                # P_REC and PFR

                pdca = PlotDataCollection(xlim=(0,12), ylim=(0.0, 1.0), xlabel='(a)', ylabel='Stim vs Non-Stim Items', xlabel_fontsize=20,ylabel_fontsize=20)
                pd1a = PlotData(x=serial_positions, y=session_summary.prob_stim_recall,linestyle='-',label='Stim')
                pdca.add_plot_data(pd1a)
                pd2a = PlotData(x = serial_positions, y=session_summary.prob_nostim_recall,linestyle = '--',label='No Stim')
                pdca.add_plot_data(pd2a)
                panel_plot.add_plot_data_collection(1,0,plot_data_collection=pdca)

                pdcb = PlotDataCollection(xlim=(0,12), ylim=(0.0, 1.0), xlabel='(b)', ylabel='', xlabel_fontsize=20,ylabel_fontsize=20)
                pd1b = PlotData(x=serial_positions, y = session_summary.prob_first_stim_recall, linestyle = '-', label = 'Stim')
                pd2b = PlotData(x=serial_positions, y=session_summary.prob_first_nostim_recall, linestyle = '--', label = 'No Stim')
                pdcb.add_plot_data(pd1b)
                pdcb.add_plot_data(pd2b)
                panel_plot.add_plot_data_collection(1,1,plot_data_collection=pdcb)
                plot = panel_plot.generate_plot()
                plot.legend()
                plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-split_prob_recall_plot_' + session_summary.STIMTAG + '-' + str(session_summary.frequency) + '.pdf')
                plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')
                session_summary.PROB_RECALL_PLOT_FILE = plot_out_fname


                # Change in recall

                panel_plot = PanelPlot(xfigsize=6, yfigsize=7.5, i_max=1, j_max=1, title='', labelsize=18)

                ylim = np.max(np.abs(session_summary.pc_diff_from_mean)) + 5.0
                if ylim > 100.0:
                    ylim = 100.0
                pd = BarPlotData(x=(0,1), y=session_summary.pc_diff_from_mean, ylim=[-ylim,ylim], xlabel='Items', ylabel='% Recall Difference (Stim-NoStim)', x_tick_labels=['Stim', 'PostStim'], xhline_pos=0.0, barcolors=['grey', 'grey'], xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)
                panel_plot.add_plot_data(0, 0, plot_data=pd)

                plot = panel_plot.generate_plot()

                session_summary.STIM_VS_NON_STIM_HALVES_PLOT_FILE = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-stim_vs_non_stim_halves_plot_' + session_summary.STIMTAG + '-' + str(session_summary.frequency) + '.pdf')

                plot.savefig(session_summary.STIM_VS_NON_STIM_HALVES_PLOT_FILE, dpi=300, bboxinches='tight')


                # number of stims and number of recalls

                n_lists = len(session_summary.n_stims_per_list)

                xfigsize = 7 * n_lists / 25.0
                if xfigsize < 10.0:
                    xfigsize = 10.0
                elif xfigsize > 18.0:
                    xfigsize = 18.0
                panel_plot = PanelPlot(xfigsize=xfigsize, yfigsize=10.0, i_max=1, j_max=1, title='', xlabel='List',
                                       ylabel='# of items', labelsize=20)

                pdc = PlotDataCollection()
                pdc.xlabel = 'List number'
                pdc.xlabel_fontsize = 20
                pdc.ylabel = '#items'
                pdc.ylabel_fontsize = 20

                x_tick_labels = np.array([str(k) for k in session_summary.list_number])
                x_tick_labels[1::5] = ''
                x_tick_labels[2::5] = ''
                x_tick_labels[3::5] = ''
                x_tick_labels[4::5] = ''

                bpd_1 = BarPlotData(x=np.arange(n_lists), y=session_summary.n_stims_per_list,
                                    x_tick_labels=x_tick_labels, title='', alpha=0.3)
                stim_x = np.where(session_summary.is_stim_list)[0]
                stim_y = session_summary.n_recalls_per_list[session_summary.is_stim_list]
                pd_1 = PlotData(x=stim_x, y=stim_y, ylim=(0, 12),
                                title='', linestyle='', color='red', marker='o', markersize=12)

                nostim_x = np.where(session_summary.is_ps_list)[0]
                nostim_y = session_summary.n_recalls_per_list[session_summary.is_ps_list]
                pd_2 = PlotData(x=nostim_x, y=nostim_y, ylim=(0, 12),
                                title='', linestyle='', color='grey', marker='o', markersize=12)

                baseline_x = np.where(session_summary.is_baseline_list)[0]
                baseline_y = session_summary.n_recalls_per_list[session_summary.is_baseline_list]
                pd_3 = PlotData(x=baseline_x, y=baseline_y, ylim=(0, 12),
                                title='', linestyle='', color='blue', marker='o', markersize=12)

                pdc.add_plot_data(pd_1)
                pdc.add_plot_data(pd_2)
                pdc.add_plot_data(pd_3)
                pdc.add_plot_data(bpd_1)

                for i in xrange(len(session_summary.list_number) - 1):
                    if session_summary.list_number[i] > session_summary.list_number[i + 1]:
                        sep_pos = i + 0.5
                        sep_plot_data = PlotData(x=[0], y=[0], levelline=[[sep_pos, sep_pos], [0, 12]], color='white',
                                                 alpha=0.0)
                        pdc.add_plot_data(sep_plot_data)

                panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

                plot = panel_plot.generate_plot()

                session_summary.STIM_AND_RECALL_PLOT_FILE = self.get_path_to_resource_in_workspace(
                    'reports/' + task + '-' + subject + '-stim_and_recall_plot_' + session_summary.STIMTAG + '-' + str(
                        session_summary.frequency) + '.pdf')

                plot.savefig(session_summary.STIM_AND_RECALL_PLOT_FILE, dpi=300, bboxinches='tight')

                panel_plot = PanelPlot(xfigsize=8,yfigsize=5,i_max=1,j_max=1)
                pd = PlotData(x=range(1,len(session_summary.prob_stim)+1),y=session_summary.prob_stim,ylim=[0,1],label_size=18,
                              xlabel='Serial Position',ylabel='Probability of stim',color='black')
                panel_plot.add_plot_data(0,0,plot_data=pd)
                plot = panel_plot.generate_plot()
                session_summary.PROB_STIM_PLOT_FILE = self.get_path_to_resource_in_workspace('reports/'+subject+'p_stim_plot_'+session_summary.STIMTAG+'-'+str(session_summary.frequency)+'.pdf')
                plot.savefig(session_summary.PROB_STIM_PLOT_FILE,dpi=300,bboxinches='tight')


class GenerateTex(ReportRamTask):
    def run(self):

        subject = self.pipeline.subject
        experiment = self.pipeline.task
        date = datetime.date.today()

        ps_events = self.get_passed_object('ps_events')
        fr5_events  = self.get_passed_object('fr5_events')
        monopolar_channels = self.get_passed_object('monopolar_channels')

        xval_output = self.get_passed_object('xval_output')
        fr5_auc = xval_output[-1].auc
        fr5_perm_pvalue = xval_output[-1].pvalue


        template_objects = ['ps4_session_data','ps4_session_summaries','preferred_location','preferred_amplitude',
                            'tstat','pvalue','auc','perm_p_value','roc_and_terc_plot_file','fr5_session_summaries']
        template_dict = {}
        for name in template_objects:
            template_dict[name.capitalize()] = self.get_passed_object(name)

        jinja_env = jinja2.Environment()
        template = jinja_env.get_template('ps4_fr5_report_base.tex.tpl')
        tex_output = template.render(
            SUBJECT= subject,
            EXPERIMENT=experiment,
            DATE = date,
            NUMBER_OF_ELECTRODES = len(monopolar_channels),
            NUMBER_OF_PS4_SESSIONS = len(np.unique(ps_events.session)),
            NUMBER_OF_FR5_SESSIONS = len(np.unique(fr5_events.session)),
            HAS_PS4  = ps_events is not None,
            HAS_FR5 = fr5_events is not None,
            FR5_AUC = fr5_auc,
            FR5_PERM_PVALUE = fr5_perm_pvalue,
            **template_dict
        )

        report_tex_file_name = self.pipeline.task + '-' + self.pipeline.subject + '-report.tex'

        with open(self.get_path_to_resource_in_workspace(report_tex_file_name),'w') as report_tex_file:
            report_tex_file.write(tex_output)

        self.pass_object('report_tex_file_name',report_tex_file_name)

class GenerateReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=False):
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

        report_core_file_name, ext = splitext(report_tex_file_name)
        report_file = join(output_directory,report_core_file_name+'.pdf')
        self.pass_object('report_file',report_file)


class DeployReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(DeployReportPDF,self).__init__(mark_as_completed)

    def run(self):
        report_file = self.get_passed_object('report_file')
        self.pipeline.deploy_report(report_path=report_file)














