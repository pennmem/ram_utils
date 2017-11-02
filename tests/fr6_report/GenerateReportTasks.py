# -*- coding: utf-8 -*-
import numpy as np
import datetime

from subprocess import call
from os.path import join, splitext
from PlotUtils import PlotData, BarPlotData, PlotDataCollection, PanelPlot

from RamPipeline import *
from ReportUtils import ReportRamTask
from ramutils.models.hmm import HierarchicalModel, HierarchicalModelPlots
from ramutils.plotting import plots
from TextTemplateUtils import replace_template,replace_template_to_string
from TexUtils.latex_table import latex_table


class GeneratePlots(ReportRamTask):
    def __init__(self):
        super(GeneratePlots,self).__init__(mark_as_completed=False)

    def run(self):
        self.create_dir_in_workspace('reports')
        task = self.pipeline.task
        subject= self.pipeline.subject
        events = self.get_passed_object(task+'_events')
        session_summaries = self.get_passed_object('fr_session_summary')
        xval_output = self.get_passed_object(task+'_xval_output')
        xval = xval_output[-1]
        #FIXME: Actually identify pairs instead
        pairs = ["LB6-LB7", "LB4-LB5", "Both-Both", "None-None"] # hard code for now until there is a good way to get them from elsewhere

        # Cross-session classifier performance
        panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, labelsize=16, wspace=5.0)
        pd1 = PlotData(x=xval.fpr, y=xval.tpr, xlim=[0.0, 1.0], ylim=[0.0, 1.0],
                       xlabel='False Alarm Rate\n(a)', ylabel='Hit Rate', xlabel_fontsize=20, ylabel_fontsize=20,
                       levelline=((0.0, 1.0), (0.0, 1.0)), color='k', markersize=1.0)
        pc_diff_from_mean = (
        xval.low_pc_diff_from_mean, xval.mid_pc_diff_from_mean, xval.high_pc_diff_from_mean)
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
        self.pass_object('ROC_AND_TERC_PLOT_FILE',plot_out_fname)


        ## Biomarker histograms
        panel_plot = PanelPlot(xfigsize = 12,yfigsize=5,i_max=1,j_max=2)

        # pre-stim
        pre_stim_probs = self.get_passed_object('pre_stim_probs')
        hist,bin_edges = np.histogram(pre_stim_probs,range=[np.round(pre_stim_probs.min(),1),np.round(pre_stim_probs.max(),1)])
        x_tick_labels = ['{:.2f}-\n{:.2f}'.format(x, y) for (x, y) in zip(bin_edges[:-1], bin_edges[1:])]
        pd = BarPlotData(x = np.arange(len(hist))-0.25,y=hist,xlabel = 'Pre-stim classifier output',ylabel='',xlabel_fontsize=20,
                         x_tick_labels=x_tick_labels,
                         ylim=[0, len(pre_stim_probs)/2])
        panel_plot.add_plot_data(0,0,plot_data=pd)

        # post-stim
        post_stim_probs = self.get_passed_object('post_stim_probs')
        hist,bin_edges = np.histogram(post_stim_probs,range=[np.round(post_stim_probs.min(),1),np.round(post_stim_probs.max(),1)])
        x_tick_labels= ['{:.2f}-\n{:.2f}'.format(x, y) for (x, y) in zip(bin_edges[:-1], bin_edges[1:])]

        pd = BarPlotData(x = np.arange(len(hist))-0.25,y=hist,xlabel = 'Post-stim classifier output',ylabel='',xlabel_fontsize=20,
                         x_tick_labels=x_tick_labels,
                         ylim =[0,len(post_stim_probs)/2])
        panel_plot.add_plot_data(0,1,plot_data=pd)

        plt = panel_plot.generate_plot()

        figname = self.get_path_to_resource_in_workspace('reports/'+self.pipeline.subject+'-biomarker-histograms.pdf')
        plt.savefig(figname,dpi=300,bboxinches='tight')
        self.pass_object('BIOMARKER_HISTOGRAM',figname)
        plt.close()

        # Post stim EEG plot
        post_stim_eeg = self.get_passed_object('post_stim_eeg')
        plt.figure(figsize=(9,5.5))
        plt.imshow(post_stim_eeg,cmap='bwr',aspect='auto',origin='lower')
        cbar = plt.colorbar()
        plt.clim([-500,500])
        plt.xlabel('Time (ms)')
        plt.ylabel('Channel (bipolar reference)')
        cbar.set_label('Avg voltage ($\mu$V)')

        figname = self.get_path_to_resource_in_workspace(join('reports',self.pipeline.subject+'-post-stim-eeg.pdf'))
        plt.savefig(figname)
        self.pass_object('post_stim_eeg_plot',figname)
        plt.close()

        sessions = np.unique(events.session)

        serial_positions = np.arange(1, 13)
        for session_summary in session_summaries:
            for pair in pairs:
                if pair == "None-None":
                    continue
                # P_REC and PFR
                panel_plot = PanelPlot(i_max=1,j_max=2,xfigsize=15, yfigsize=7.5, title='', labelsize=18)
                pdca = PlotDataCollection(xlim=(0,12), ylim=(0.0, 1.0), xlabel='(a)', ylabel='Stim vs Non-Stim Items', xlabel_fontsize=20,ylabel_fontsize=20)
                pd1a = PlotData(x=session_summary.prob_stim_recall[pair].index.values, y=session_summary.prob_stim_recall[pair].values,linestyle='-',label='Stim items')
                pdca.add_plot_data(pd1a)
                pd2a = PlotData(x = session_summary.prob_nostim_recall[pair].index.values, y=session_summary.prob_nostim_recall[pair].values,linestyle = '--',label='Non-stim Items')
                pdca.add_plot_data(pd2a)
                panel_plot.add_plot_data_collection(0,0,plot_data_collection=pdca)

                pdcb = PlotDataCollection(xlim=(0,12), ylim=(0.0, 1.0), xlabel='(b)', ylabel='', xlabel_fontsize=20,ylabel_fontsize=20)
                pd1b = PlotData(x=serial_positions, y = session_summary.prob_first_stim_recall[pair], linestyle = '-', label = 'Stim Items')
                pd2b = PlotData(x=serial_positions, y=session_summary.prob_first_nostim_recall[pair], linestyle = '--', label = 'Non-stim Items')
                pdcb.add_plot_data(pd1b)
                pdcb.add_plot_data(pd2b)
                panel_plot.add_plot_data_collection(0,1,plot_data_collection=pdcb)
                plot = panel_plot.generate_plot()
                plot.legend()
                plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-split_prob_recall_plot_'+ str(session_summary.session) + "-" + session_summary.stimtag[pair] + '_' + str(session_summary.frequency[pair]) + '.pdf')
                plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')
                session_summary.PROB_RECALL_PLOT_FILE[pair] = plot_out_fname

                # Change in recall
                panel_plot = PanelPlot(xfigsize=6, yfigsize=7.5, i_max=1, j_max=1, title='', labelsize=18)
                ylim = np.nanmax(np.abs(session_summary.pc_diff_from_mean[pair])) + 5.0
                if ylim > 100.0:
                    ylim = 100.0
                if ylim<10.0:
                    ylim=10.0
                pd = BarPlotData(x=(0,1), y=session_summary.pc_diff_from_mean[pair], ylim=[-ylim,ylim], xlabel='Items', ylabel='% Recall Difference (Stim-NoStim)', x_tick_labels=['Stim', 'PostStim'], xhline_pos=0.0, barcolors=['grey', 'grey'], xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)
                panel_plot.add_plot_data(0, 0, plot_data=pd)
                plot = panel_plot.generate_plot()
                session_summary.STIM_VS_NON_STIM_HALVES_PLOT_FILE[pair] = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-stim_vs_non_stim_halves_plot_'+ str(session_summary.session) + "-" + session_summary.stimtag[pair] + '_' + str(session_summary.frequency[pair]) + '.pdf')
                plot.savefig(session_summary.STIM_VS_NON_STIM_HALVES_PLOT_FILE[pair], dpi=300, bboxinches='tight')

                # number of stims and number of recalls
                n_lists = len(session_summary.n_stims_per_list[pair])
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
                x_tick_labels = [str(k) for k in range(1, 26)]
                bpd_1 = BarPlotData(x=np.arange(n_lists), y=session_summary.n_stims_per_list[pair],
                                    x_tick_labels=x_tick_labels, title='', alpha=0.3)
                stim_x = np.where(session_summary.is_stim_list[pair])[0]
                stim_y = session_summary.n_recalls_per_list[pair][session_summary.is_stim_list[pair]]
                pd_1 = PlotData(x=stim_x, y=stim_y, ylim=(0, 12),
                                title='', linestyle='', color='red', marker='o', markersize=12)
                plot_data_list = [bpd_1,pd_1]
                if session_summary.is_stim_list[pair].all():
                    nostim_x = np.where(session_summary.is_ps_list[pair])[0]
                    nostim_y = session_summary.n_recalls_per_list[pair][session_summary.is_ps_list[pair]]
                    pd_2 = PlotData(x=nostim_x, y=nostim_y, ylim=(0, 12),
                                    title='', linestyle='', color='grey', marker='o', markersize=12)
                    baseline_x = np.where(session_summary.is_baseline_list[pair])[0]
                    baseline_y = session_summary.n_recalls_per_list[pair][np.array(session_summary.is_baseline_list[pair])]
                    pd_3 = PlotData(x=baseline_x, y=baseline_y, ylim=(0, 12),
                                    title='', linestyle='', color='blue', marker='o', markersize=12)
                    plot_data_list.extend([pd_2,pd_3])
                else:
                    nostim_x = np.where(session_summary.is_nonstim_list[pair])[0]
                    nostim_y = session_summary.n_recalls_per_list[pair][session_summary.is_nonstim_list[pair]]
                    pd_2 = PlotData(x=nostim_x, y=nostim_y, ylim=(0, 12),
                                    title='', linestyle='', color='blue', marker='o', markersize=12)
                    plot_data_list.append(pd_2)
                for pd in plot_data_list:
                    if (pd.x.shape and pd.y.shape) and all(pd.x.shape) and all(pd.y.shape):
                        print pd.x.shape
                        print pd.y.shape
                        pdc.add_plot_data(pd)
                panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)
                plot = panel_plot.generate_plot()
                session_summary.STIM_AND_RECALL_PLOT_FILE[pair] = self.get_path_to_resource_in_workspace(
                    'reports/' + task + '-' + subject + '-stim_and_recall_plot_'+ str(session_summary.session) + "-" + session_summary.stimtag[pair] + '-' + str(
                        session_summary.frequency[pair]) + '.pdf')
                plot.savefig(session_summary.STIM_AND_RECALL_PLOT_FILE[pair], dpi=300, bboxinches='tight')

                # Probability of stim plots
                panel_plot = PanelPlot(xfigsize=8,yfigsize=5,i_max=1,j_max=1)
                pd = PlotData(x=range(1,len(session_summary.prob_stim[pair])+1),y=session_summary.prob_stim[pair],ylim=[0,1],label_size=18,
                            xlabel='Serial Position',ylabel='Probability of stim',color='black')
                panel_plot.add_plot_data(0,0,plot_data=pd)
                plot = panel_plot.generate_plot()
                session_summary.PROB_STIM_PLOT_FILE[pair] = self.get_path_to_resource_in_workspace('reports/'+subject+'_stim_plot_session_'+ str(session_summary.session) + "-" + session_summary.stimtag[pair]+'-'+str(session_summary.frequency[pair])+'.pdf')
                plot.savefig(session_summary.PROB_STIM_PLOT_FILE[pair],dpi=300,bboxinches='tight')


class GenerateTex(ReportRamTask):
    def run(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.task
        date = datetime.date.today()
        pairs = ["LB6-LB7", "LB4-LB5", "Both-Both", "None-None"] # hard code for now until there is a good way to get them from elsewhere

        latex = self.generate_latex()
        report_tex_file_name = '%s-%s_report.tex'%(subject,experiment)

        replace_template('ps4_fr6_report_base.tex.tpl',self.get_path_to_resource_in_workspace('reports',report_tex_file_name),
                         {
                             '<SUBJECT>':subject,
                             '<EXPERIMENT>':experiment,
                             '<DATE>':date,
                             '<EXPERIMENT_SECTION>':latex})
        self.pass_object('report_tex_file_name',report_tex_file_name)


    def generate_latex(self):
        subject =self.pipeline.subject
        task = self.pipeline.task
        monopolar_channels = self.get_passed_object('monopolar_channels')
        xval_output = self.get_passed_object(task+'_xval_output')
        fr1_xval_output = self.get_passed_object('xval_output_all_electrodes')
        fr1_auc = fr1_xval_output[-1].auc
        fr1_pvalue = self.get_passed_object('pvalue_full')
        session_data =self.get_passed_object('session_table')
        pairs = ["LB6-LB7", "LB4-LB5", "Both-Both", "None-None"] # hard code for now until there is a good way to get them from elsewhere

        if xval_output:
            auc = '%2.2f'%xval_output[-1].auc
            perm_pvalue = self.get_passed_object(task+'_pvalue')
            roc_title = 'Classifier generalization to FR6'
            jstat_thresh = '%2.2f'%xval_output[-1].jstat_thresh
        else:
            auc = fr1_auc
            perm_pvalue = fr1_pvalue
            roc_title = 'FR1 Classifier Performance'
            jstat_thresh = fr1_xval_output[-1].jstat_thresh


        events  = self.get_passed_object(task+'_events')
        n_sessions = len(np.unique(events.session))
        session_summary = self.get_passed_object('fr_session_summary')
        
        all_session_tex = ''

        biomarker_histogram = self.get_passed_object('BIOMARKER_HISTOGRAM')

        if events is not None and all(events.shape):
            # TODO: With the multi stim-sites do a single
            # pass over session and just look up the appropriate data based on the tag
            for session_summary in session_summary:
                sessions = session_summary.session
                sessions = str(sessions)

                biomarker_tex = replace_template_to_string('biomarker_plots.tex.tpl',
                                                           {'<STIM_VS_NON_STIM_HALVES_PLOT_FILE_A>':session_summary.STIM_VS_NON_STIM_HALVES_PLOT_FILE[pairs[0]],
                                                            '<STIM_VS_NON_STIM_HALVES_PLOT_FILE_B>':session_summary.STIM_VS_NON_STIM_HALVES_PLOT_FILE[pairs[1]],
                                                            '<STIM_VS_NON_STIM_HALVES_PLOT_FILE_AB>':session_summary.STIM_VS_NON_STIM_HALVES_PLOT_FILE[pairs[2]],
                                                            ' <COMPARISON_LIST_TYPE>': 'non-stim' if ((events[np.in1d(events.session,session_summary.session)].phase=='NON-STIM')).any() else 'FR1'
                                                           })
                item_level_comparison = '' 
                session_tex = replace_template_to_string('fr6_session.tex.tpl',
                             {
                                 '<SESSIONS>': sessions,
                                 '<STIMTAG_A>': session_summary.stimtag[pairs[0]],
                                 '<REGION_A>': session_summary.region_of_interest[pairs[0]],
                                 '<STIMTAG_B>': session_summary.stimtag[pairs[1]],
                                 '<REGION_B>': session_summary.region_of_interest[pairs[1]],
                                 '<PULSE_FREQ_A>': session_summary.frequency[pairs[0]],
                                 '<PULSE_FREQ_B>': session_summary.frequency[pairs[1]],
                                 '<AMPLITUDE_A>': session_summary.amplitude[pairs[0]],
                                 '<AMPLITUDE_B>': session_summary.amplitude[pairs[1]],
                                 '<N_WORDS>': session_summary.n_words,
                                 '<N_CORRECT_WORDS>': session_summary.n_correct_words,
                                 '<PC_CORRECT_WORDS>': '%2.2f'%session_summary.pc_correct_words,
                                 '<N_PLI>': session_summary.n_pli,
                                 '<PC_PLI>': '%2.2f'%session_summary.pc_pli,
                                 '<N_ELI>': session_summary.n_eli,
                                 '<PC_ELI>': '%2.2f'% session_summary.pc_eli,
                                 '<N_MATH>': session_summary.n_math,
                                 '<N_CORRECT_MATH>':session_summary.n_correct_math,
                                 '<PC_CORRECT_MATH>':'%2.2f'%session_summary.pc_correct_math,
                                 '<MATH_PER_LIST>':'%2.2f'%session_summary.math_per_list,
                                 '<PROB_RECALL_PLOT_FILE_A>':session_summary.PROB_RECALL_PLOT_FILE[pairs[0]],
                                 '<PROB_RECALL_PLOT_FILE_B>':session_summary.PROB_RECALL_PLOT_FILE[pairs[1]],
                                 '<PROB_RECALL_PLOT_FILE_AB>':session_summary.PROB_RECALL_PLOT_FILE[pairs[2]],
                                 '<COMPARISON_LIST_TYPE>': 'non-stim' if ((events[np.in1d(events.session,session_summary.session)].phase=='NON-STIM')).any() else 'FR1',
                                 '<ITEMLEVEL_COMPARISON>': item_level_comparison,
                                 '<N_CORRECT_NONSTIM>':session_summary.n_correct_nonstim,
                                 '<N_TOTAL_NONSTIM>':session_summary.n_total_nonstim,
                                 '<PC_FROM_NONSTIM>':'%2.2f'%session_summary.pc_from_nonstim,
                                 '<N_NONSTIM_INTR>':session_summary.n_nonstim_intr,
                                 '<PC_FROM_NONSTIM_INTR>':'%2.2f'%session_summary.pc_from_nonstim_intr,
                                 '<N_CORRECT_STIM_A>':session_summary.n_correct_stim[pairs[0]],
                                 '<N_TOTAL_STIM_A>':session_summary.n_total_stim[pairs[0]],
                                 '<PC_FROM_STIM_A>':'%2.2f'%session_summary.pc_from_stim[pairs[0]],
                                 '<CHISQR_A>':'%.4f'%session_summary.chisqr[pairs[0]],
                                 '<PVALUE_A>':'%.4f'%session_summary.pvalue[pairs[0]],
                                 '<N_STIM_INTR_A>': session_summary.n_stim_intr[pairs[0]],
                                 '<PC_FROM_STIM_INTR_A>':'%2.2f'%session_summary.pc_from_stim_intr[pairs[0]],
                                 '<N_CORRECT_STIM_B>':session_summary.n_correct_stim[pairs[1]],
                                 '<N_TOTAL_STIM_B>':session_summary.n_total_stim[pairs[1]],
                                 '<PC_FROM_STIM_B>':'%2.2f'%session_summary.pc_from_stim[pairs[1]],
                                 '<CHISQR_B>':'%.4f'%session_summary.chisqr[pairs[1]],
                                 '<PVALUE_B>':'%.4f'%session_summary.pvalue[pairs[1]],
                                 '<N_STIM_INTR_B>': session_summary.n_stim_intr[pairs[1]],
                                 '<PC_FROM_STIM_INTR_B>':'%2.2f'%session_summary.pc_from_stim_intr[pairs[1]],
                                 '<N_CORRECT_STIM_AB>':session_summary.n_correct_stim[pairs[2]],
                                 '<N_TOTAL_STIM_AB>':session_summary.n_total_stim[pairs[2]],
                                 '<PC_FROM_STIM_AB>':'%2.2f'%session_summary.pc_from_stim[pairs[2]],
                                 '<CHISQR_AB>':'%.4f'%session_summary.chisqr[pairs[2]],
                                 '<PVALUE_AB>':'%.4f'%session_summary.pvalue[pairs[2]],
                                 '<N_STIM_INTR_AB>': session_summary.n_stim_intr[pairs[2]],
                                 '<PC_FROM_STIM_INTR_AB>':'%2.2f'%session_summary.pc_from_stim_intr[pairs[2]],
                                 '<STIM_AND_RECALL_PLOT_FILE_A>':session_summary.STIM_AND_RECALL_PLOT_FILE[pairs[0]],
                                 '<STIM_AND_RECALL_PLOT_FILE_B>':session_summary.STIM_AND_RECALL_PLOT_FILE[pairs[1]],
                                 '<STIM_AND_RECALL_PLOT_FILE_AB>':session_summary.STIM_AND_RECALL_PLOT_FILE[pairs[2]],
                                 '<STIM_AND_RECALL_PLOT_FILE_NOSTIM>':session_summary.STIM_AND_RECALL_PLOT_FILE[pairs[2]], #FIXME: Actually build this one
                                 '<PROB_STIM_PLOT_FILE_A>':session_summary.PROB_STIM_PLOT_FILE[pairs[0]],
                                 '<PROB_STIM_PLOT_FILE_B>':session_summary.PROB_STIM_PLOT_FILE[pairs[1]],
                                 '<PROB_STIM_PLOT_FILE_AB>':session_summary.PROB_STIM_PLOT_FILE[pairs[2]],
                                 '<BIOMARKER_PLOTS>':biomarker_tex,
                             }
                )
                all_session_tex += session_tex
        tex = replace_template_to_string(
            'FR6_section.tex.tpl',
            {
                '<SUBJECT>':subject,
                '<TASK>':task,
                '<NUMBER_OF_ELECTRODES>':len(monopolar_channels),
                '<NUMBER_OF_SESSIONS>':n_sessions,
                '<AUC>':'%2.2f'%fr1_auc,
                '<PERM-P-VALUE>':fr1_pvalue if fr1_pvalue>0 else '<0.01',
                '<SESSION_DATA>':latex_table(session_data),
                '<ROC_TITLE>':roc_title,
                '<STIM_TITLE>': 'Estimated Effects of Stim',
                '<AUC-1>':auc,
                '<PERM-P-VALUE-1>':perm_pvalue if perm_pvalue>0 else '<0.01',
                '<JSTAT-THRESH-1>':jstat_thresh,
                '<ROC_AND_TERC_PLOT_FILE_1>':self.get_passed_object('ROC_AND_TERC_PLOT_FILE'),
                '<AUC-2>':auc,
                '<PERM-P-VALUE-2>':perm_pvalue if perm_pvalue>0 else '<0.01',
                '<JSTAT-THRESH-2>':jstat_thresh,
                '<ROC_AND_TERC_PLOT_FILE_2>':self.get_passed_object('ROC_AND_TERC_PLOT_FILE'),
#                '<ESTIMATED_STIM_EFFECT_PLOT_FILE_list>': self.get_path_to_resource_in_workspace('reports/' + '_'.join([self.pipeline.subject, 'list', 'forestplot.pdf'])),
#                '<ESTIMATED_STIM_EFFECT_PLOT_FILE_stim>': self.get_path_to_resource_in_workspace('reports/' + '_'.join([self.pipeline.subject, 'stim', 'forestplot.pdf'])),
#                '<ESTIMATED_STIM_EFFECT_PLOT_FILE_post_stim>': self.get_path_to_resource_in_workspace('reports/' + '_'.join([self.pipeline.subject, 'post_stim', 'forestplot.pdf'])),
                '<REPORT_PAGES>':all_session_tex,
                '<BIOMARKER_HISTOGRAM>':biomarker_histogram,
                '<POST_STIM_EEG>':self.get_passed_object('post_stim_eeg_plot')
            }
        )
        return tex


class GenerateReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=False):
        super(GenerateReportPDF,self).__init__(mark_as_completed)

    def run(self):
        output_directory = self.get_path_to_resource_in_workspace('reports')

        texinputs_set_str = r'export TEXINPUTS="' + output_directory + '":$TEXINPUTS;'

        report_tex_file_name = self.get_passed_object('report_tex_file_name')

        pdflatex_command_str = texinputs_set_str + 'module load Tex; pdflatex' \
                               + ' -output-directory '+output_directory\
                               + ' -shell-escape ' \
                               + self.get_path_to_resource_in_workspace('reports/'+report_tex_file_name)

        call([pdflatex_command_str], shell=True)

        report_core_file_name, ext = splitext(report_tex_file_name)
        report_file = join(output_directory,report_core_file_name+'.pdf')
        self.pass_object('report_file',report_file)


class DeployReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=False):
        super(DeployReportPDF,self).__init__(mark_as_completed)

    def run(self):
        report_file = self.get_passed_object('report_file')
        self.pipeline.deploy_report(report_path=report_file)

