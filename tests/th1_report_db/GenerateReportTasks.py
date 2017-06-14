import datetime
from subprocess import call

import TextTemplateUtils
import matplotlib.pyplot as plt
import numpy as np
from RamPipeline import *
from ReportUtils import ReportRamTask

from latex_table import latex_table
from ram_utils.PlotUtils import PlotData, BarPlotData, PanelPlot

class GenerateTex(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(GenerateTex,self).__init__(mark_as_completed)
        self.params = params
        
    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        n_sess = self.get_passed_object('NUMBER_OF_SESSIONS')
        n_bps = self.get_passed_object('NUMBER_OF_ELECTRODES')

        tex_combined_template = task + '_combined.tex.tpl'        
        if self.params.doConf_classification & self.get_passed_object('conf_decode_success'):
            if self.params.doClass_wTranspose:
                tex_combined_template = task + '_combined_wConf_and_transpose.tex.tpl'        
            else:
                tex_combined_template = task + '_combined_wConf.tex.tpl'        
        combined_report_tex_file_name = '%s_%s_report.tex' % (subject,task)

        self.set_file_resources_to_move(combined_report_tex_file_name, dst='reports')

        cumulative_summary = self.get_passed_object('cumulative_summary')
        cumulative_data_tex_table = latex_table(self.get_passed_object('SESSION_DATA'))
        cumulative_ttest_tex_table_LTA = latex_table(self.get_passed_object('cumulative_ttest_data_LTA'))
        cumulative_ttest_tex_table_HTA = latex_table(self.get_passed_object('cumulative_ttest_data_HTA'))
        cumulative_ttest_tex_table_G = latex_table(self.get_passed_object('cumulative_ttest_data_G'))
        cumulative_ttest_tex_table_HFA = latex_table(self.get_passed_object('cumulative_ttest_data_HFA'))                        

        replace_dict = {'<PROB_RECALL_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-pCorrBar.pdf',
                        '<DIST_HIST_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-distHist.pdf',
                        '<ERR_BLOCK_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-errByBlock.pdf',
                        '<AUC_THRESH_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-auc_by_threshold.pdf',                       
                        '<DATE>': datetime.date.today(),
                        '<SESSION_DATA>': cumulative_data_tex_table,
                        '<SUBJECT>': subject.replace('_','\\textunderscore'),
                        '<NUMBER_OF_SESSIONS>': n_sess,
                        '<NUMBER_OF_ELECTRODES>': n_bps,
                        '<N_ITEMS>': cumulative_summary.n_items,
                        '<N_CORRECT_ITEMS>': cumulative_summary.n_correct_items,
                        '<PC_CORRECT_WORDS>': '%.2f' % cumulative_summary.pc_correct_items,      
                        '<N_TRANSPOSE_ITEMS>': cumulative_summary.n_transposed_items,
                        '<PC_TRANSPOSE_WORDS>': '%.2f' % cumulative_summary.pc_transposed_items,   
                        '<MEAN_NORM_ERROR>': '%.2f' % cumulative_summary.mean_norm_err,                                
                        '<SIGNIFICANT_ELECTRODES_LTA>': cumulative_ttest_tex_table_LTA,
                        '<SIGNIFICANT_ELECTRODES_HTA>': cumulative_ttest_tex_table_HTA,
                        '<SIGNIFICANT_ELECTRODES_G>': cumulative_ttest_tex_table_G,
                        '<SIGNIFICANT_ELECTRODES_HFA>': cumulative_ttest_tex_table_HFA,                                          
                        '<TIME_WIN_START>': self.params.th1_start_time*1000,
                        '<TIME_WIN_END>': self.params.th1_end_time*1000,                        
                        '<AUC>': cumulative_summary.auc,
                        '<PERM-P-VALUE>': cumulative_summary.perm_test_pvalue,
                        '<AUC_CONF>': cumulative_summary.auc_conf,
                        '<PERM-P-VALUE_CONF>': cumulative_summary.perm_test_pvalue_conf,                        
                        '<J-THRESH>': cumulative_summary.jstat_thresh,
                        '<J-PERC>': cumulative_summary.jstat_percentile,
                        '<AUC_TRANSPOSE>': cumulative_summary.auc_transpose,
                        '<PERM-P-VALUE_TRANSPOSE>': cumulative_summary.perm_test_pvalue_transpose,
                        '<J-THRESH_TRANSPOSE>': cumulative_summary.jstat_thresh_transpose,
                        '<J-PERC_TRANSPOSE>': cumulative_summary.jstat_percentile_transpose,                        
                        '<ROC_AND_TERC_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-roc_and_terc_plot_combined.pdf',
                        '<ROC_AND_TERC_PLOT_CONF_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-roc_and_terc_plot_combined_conf.pdf',
                        '<ROC_AND_TERC_PLOT_TRANSPOSE_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-roc_and_terc_plot_combined_transpose.pdf'                                                                        
                        }

        TextTemplateUtils.replace_template(template_file_name=tex_combined_template, out_file_name=combined_report_tex_file_name, replace_dict=replace_dict)

        self.pass_object('combined_report_tex_file_name', combined_report_tex_file_name)


class GeneratePlots(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ReportRamTask,self).__init__(mark_as_completed)
        self.params = params
                
    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.create_dir_in_workspace('reports')


        cumulative_summary = self.get_passed_object('cumulative_summary')
        
        # probabilty correct bar plot
        fig, ax = plt.subplots()
        colors  = np.array([[18., 151., 147.],[80., 80., 80.],[255., 114., 96.]])/255
        ax.grid()
        ax.set_axisbelow(True)
        for conf in xrange(3):
            ax.bar(conf+1-.35/2,cumulative_summary.prob_by_conf[conf],.35,color=colors[conf],linewidth=3)
        ax.set_xlabel('Confidence',fontsize=20)
        ax.set_ylabel('Probability Correct',fontsize=20)
        plt.xticks(range(1,4), ('Low (%.2f)'%(cumulative_summary.percent_conf[0]),'Medium (%.2f)'%(cumulative_summary.percent_conf[1]),'High (%.2f)'%(cumulative_summary.percent_conf[2])),fontsize=18)
        plt.setp(ax.get_yticklabels(), fontsize=18)
        plt.show()                
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-pCorrBar.pdf')        
        plt.savefig(plot_out_fname, dpi=300, bboxinches='tight')        

        # distance error probability hist
        binMid = np.arange(2.5,100,5);
        labels = ['Low','Medium','High']
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_axisbelow(True)
        for conf in xrange(3):
            ax.plot(binMid,cumulative_summary.dist_hist[conf],linewidth=4,color=colors[conf],label=labels[conf])
        ax.set_xlabel('Distance Error',fontsize=20)
        ax.set_ylabel('Probability',fontsize=20)
        plt.setp(ax.get_yticklabels(), fontsize=18)
        plt.setp(ax.get_xticklabels(), fontsize=18)
        legend = ax.legend(loc='upper right')
        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize(16)        
        plt.show()                
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-distHist.pdf')        
        plt.savefig(plot_out_fname, dpi=300, bboxinches='tight')           
        
        # distance error by block     
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_axisbelow(True)
        ax.bar(np.array([1,2,3,4,5])-.5/2,cumulative_summary.err_by_block,.5,color='.75',linewidth=3,yerr=cumulative_summary.err_by_block_sem,error_kw=dict(ecolor='black', lw=3, capsize=5, capthick=2))
        ax.set_xlabel('Block Number',fontsize=20)
        ax.set_ylabel('Distance Error',fontsize=20)
        plt.xticks(range(1,6),fontsize=18)
        plt.setp(ax.get_yticklabels(), fontsize=18)
        plt.xlim(.5,5.5)
        plt.show()                
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-errByBlock.pdf')        
        plt.savefig(plot_out_fname, dpi=300, bboxinches='tight')        


        # classifier auc by correct distance threshold
        if self.params.doDist_classification:
            fig, ax = plt.subplots()
            ax.grid()
            ax.set_axisbelow(True)        
            plt.plot(cumulative_summary.thresholds,cumulative_summary.aucs_by_thresh,linewidth=4,color='black')
            ax.set_xlabel('Distance Threshold (% Correct)',fontsize=20)
            ax.set_ylabel('AUC',fontsize=20)
            plt.setp(ax.get_yticklabels(), fontsize=18)
            plt.setp(ax.get_xticklabels(), fontsize=12)
            plt.xlim(cumulative_summary.thresholds[0]-1,cumulative_summary.thresholds[-1]+1)
            xlim = plt.xlim()
            plt.plot(xlim,[.5 ,.5],'--k',linewidth=2)
            ymin,ymax = plt.ylim()
            plt.ylim(ymin=ymin-.03,ymax=ymax+.03)
            plt.plot([13,13],plt.ylim(),'--k',linewidth=2)
        
            sigPos = cumulative_summary.thresholds[(cumulative_summary.pval_by_thresh < .025)]
            plt.plot(sigPos,[cumulative_summary.aucs_by_thresh.max() + .03]*len(sigPos),'ro') 
        
            sigNeg = cumulative_summary.thresholds[(cumulative_summary.pval_by_thresh > .975)]
            plt.plot(sigNeg,[cumulative_summary.aucs_by_thresh.min() - .03]*len(sigNeg),'bo')        
        
            labels = np.empty(len(cumulative_summary.thresholds), dtype="S10")
            for i, thresh in enumerate(cumulative_summary.thresholds):
                labels[i] = ('%d (%.1f)'%(int(thresh),cumulative_summary.pCorr_by_thresh[i]*100))
            ticks = np.arange(int(cumulative_summary.thresholds[0]),int(cumulative_summary.thresholds[-1])+1,3)
            plt.xticks(ticks,labels[ticks-ticks.min()],rotation=315)
            plt.gcf().subplots_adjust(bottom=0.25)
            plt.show()
            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-auc_by_threshold.pdf')        
            plt.savefig(plot_out_fname, dpi=300, bboxinches='tight')
        
        
        # original classifier auc and roc
        panel_plot = PanelPlot(xfigsize=15, yfigsize=7, i_max=1, j_max=2, title='', labelsize=18)            
        pd1 = PlotData(x=cumulative_summary.fpr, y=cumulative_summary.tpr, xlim=[0.0,1.0], ylim=[0.0,1.0], xlabel='False Alarm Rate', ylabel='Hit Rate', levelline=((0.001,0.999),(0.001,0.999)), color='k', markersize=1.0, xlabel_fontsize=18, ylabel_fontsize=18)
        ylim = np.max(np.abs(cumulative_summary.pc_diff_from_mean)) + 5.0
        if ylim > 100.0:
            ylim = 100.0
        pd2 = BarPlotData(x=(0,1,2), y=cumulative_summary.pc_diff_from_mean, ylim=[-ylim,ylim], xlabel='Tercile of Classifier Estimate', ylabel='Change From Mean (%)', x_tick_labels=['Low', 'Middle', 'High'], xhline_pos=0.0, barcolors=['grey','grey', 'grey'], xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)
        panel_plot.add_plot_data(0, 0, plot_data=pd1)
        panel_plot.add_plot_data(0, 1, plot_data=pd2)
        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-roc_and_terc_plot_combined.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')
        
                
        # confidence classifier auc and roc
        if self.params.doConf_classification & self.get_passed_object('conf_decode_success'):
            panel_plot = PanelPlot(xfigsize=15, yfigsize=7, i_max=1, j_max=2, title='', labelsize=18)        
            pd3 = PlotData(x=cumulative_summary.fpr_conf, y=cumulative_summary.tpr_conf, xlim=[0.0,1.0], ylim=[0.0,1.0], xlabel='False Alarm Rate', ylabel='Hit Rate', levelline=((0.001,0.999),(0.001,0.999)), color='k', markersize=1.0, xlabel_fontsize=18, ylabel_fontsize=18)
            ylim = np.max(np.abs(cumulative_summary.pc_diff_from_mean_conf)) + 5.0
            if ylim > 100.0:
                ylim = 100.0
            pd4 = BarPlotData(x=(0,1,2), y=cumulative_summary.pc_diff_from_mean_conf, ylim=[-ylim,ylim], xlabel='Tercile of Classifier Estimate', ylabel='Change From Mean (%)', x_tick_labels=['Low', 'Middle', 'High'], xhline_pos=0.0, barcolors=['grey','grey', 'grey'], xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)
            panel_plot.add_plot_data(0, 0, plot_data=pd3)        
            panel_plot.add_plot_data(0, 1, plot_data=pd4)                
            plot = panel_plot.generate_plot()
            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-roc_and_terc_plot_combined_conf.pdf')
            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')
            
        #  classifier auc and roc with transposed responses counted as correct
        if self.params.doClass_wTranspose:
            panel_plot = PanelPlot(xfigsize=15, yfigsize=7, i_max=1, j_max=2, title='', labelsize=18)        
            pd3 = PlotData(x=cumulative_summary.fpr_transpose, y=cumulative_summary.tpr_transpose, xlim=[0.0,1.0], ylim=[0.0,1.0], xlabel='False Alarm Rate', ylabel='Hit Rate', levelline=((0.001,0.999),(0.001,0.999)), color='k', markersize=1.0, xlabel_fontsize=18, ylabel_fontsize=18)
            ylim = np.max(np.abs(cumulative_summary.pc_diff_from_mean_transpose)) + 5.0
            if ylim > 100.0:
                ylim = 100.0
            pd4 = BarPlotData(x=(0,1,2), y=cumulative_summary.pc_diff_from_mean_transpose, ylim=[-ylim,ylim], xlabel='Tercile of Classifier Estimate', ylabel='Change From Mean (%)', x_tick_labels=['Low', 'Middle', 'High'], xhline_pos=0.0, barcolors=['grey','grey', 'grey'], xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)
            panel_plot.add_plot_data(0, 0, plot_data=pd3)        
            panel_plot.add_plot_data(0, 1, plot_data=pd4)                
            plot = panel_plot.generate_plot()
            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-roc_and_terc_plot_combined_transpose.pdf')
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

    def run(self):
        report_file = self.get_passed_object('report_file')
        self.pipeline.deploy_report(report_path=report_file,suffix='_db')
