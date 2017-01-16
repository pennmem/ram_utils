from RamPipeline import *

import TextTemplateUtils
from PlotUtils import PlotData, BarPlotData, PlotDataCollection, PanelPlot
from latex_table import latex_table

import numpy as np
import datetime
from subprocess import call

from ReportUtils import ReportRamTask


def pvalue_formatting(p):
    return '\leq 0.001' if p<=0.001 else ('%.3f'%p)


class GenerateTex(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GenerateTex,self).__init__(mark_as_completed)

    def run(self):
        task = self.pipeline.task

        tex_template = 'fr_stim_report.tex.tpl'
        tex_session_template = 'fr_stim_session.tex.tpl'

        tex_biomarker_plot_template = 'biomarker_plots.tex.tpl' if task=='RAM_FR4' else 'fr3_plots.tex.tpl'
        tex_itemlevel_comparison_template = 'itemlevel_comparison.tex.tpl'

        report_tex_file_name = self.pipeline.task + '-cat'+ self.pipeline.task+ '-joint-'+self.pipeline.subject + '-report.tex'
        self.pass_object('report_tex_file_name',report_tex_file_name)

        self.set_file_resources_to_move(report_tex_file_name, dst='reports')

        n_sess = self.get_passed_object('NUMBER_OF_SESSIONS')
        n_elecs = self.get_passed_object('NUMBER_OF_ELECTRODES')

        session_summary_array = self.get_passed_object('session_summary_array')

        tex_session_pages_str = ''

        for session_summary in session_summary_array:
            itemlevel_comparison = ''
            if task=='FR3':
                itemlevel_comparison_replace_dict = {
                    '<N_CORRECT_STIM_ITEMS>': session_summary.n_correct_stim_items,
                    '<N_TOTAL_STIM_ITEMS>': session_summary.n_total_stim_items,
                    '<PC_STIM_ITEMS>': '%.2f' % session_summary.pc_stim_items,
                    '<N_CORRECT_POST_STIM_ITEMS>': session_summary.n_correct_post_stim_items,
                    '<N_TOTAL_POST_STIM_ITEMS>': session_summary.n_total_post_stim_items,
                    '<PC_POST_STIM_ITEMS>': '%.2f' % session_summary.pc_post_stim_items,
                    '<N_CORRECT_NONSTIM_ITEMS>': session_summary.n_correct_nonstim_low_bio_items,
                    '<N_TOTAL_NONSTIM_ITEMS>': session_summary.n_total_nonstim_low_bio_items,
                    '<PC_NONSTIM_ITEMS>': '%.2f' % session_summary.pc_nonstim_low_bio_items,
                    '<N_CORRECT_NONSTIM_POST_ITEMS>': session_summary.n_correct_nonstim_post_low_bio_items,
                    '<N_TOTAL_NONSTIM_POST_ITEMS>': session_summary.n_total_nonstim_post_low_bio_items,
                    '<PC_NONSTIM_POST_ITEMS>': '%.2f' % session_summary.pc_nonstim_post_low_bio_items,
                    '<CHISQR_STIM_ITEMS>': '%.2f' % session_summary.chisqr_stim_item,
                    '<PVALUE_STIM_ITEMS>': '%.2f' % session_summary.pvalue_stim_item,
                    '<CHISQR_POST_STIM_ITEMS>': '%.2f' % session_summary.chisqr_post_stim_item,
                    '<PVALUE_POST_STIM_ITEMS>': '%.2f' % session_summary.pvalue_post_stim_item
                }

                itemlevel_comparison = TextTemplateUtils.replace_template_to_string(tex_itemlevel_comparison_template, itemlevel_comparison_replace_dict)

            biomarker_plot_replace_dict = {'<STIM_VS_NON_STIM_HALVES_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-stim_vs_non_stim_halves_plot_' + session_summary.stimtag + '-' + str(session_summary.frequency) + '.pdf',
                                          }
            biomarker_plots = TextTemplateUtils.replace_template_to_string(tex_biomarker_plot_template, biomarker_plot_replace_dict)

            replace_dict = {'<STIMTAG>': session_summary.stimtag,
                            '<REGION>': session_summary.region_of_interest,
                            '<FREQUENCY>': session_summary.frequency,
                            '<SESSIONS>': ','.join([str(s) for s in session_summary.sessions]),
                            '<PROB_RECALL_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-prob_recall_plot_' + session_summary.stimtag + '-' + str(session_summary.frequency) + '.pdf',
                            '<BIOMARKER_PLOTS>': biomarker_plots,
                            '<ITEMLEVEL_COMPARISON>': itemlevel_comparison,
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
        xval_output = self.get_passed_object('xval_output')
        perm_test_pvalue = self.get_passed_object('pvalue')
        fr3_xval_output = self.get_passed_object(task+'_xval_output')
        fr3_perm_test_pvalue = self.get_passed_object(task+'_pvalue')

        replace_dict = {'<DATE>': datetime.date.today(),
                        '<EXPERIMENT>': self.pipeline.task,
                        '<SESSION_DATA>': session_data_tex_table,
                        '<SUBJECT>': self.pipeline.subject.replace('_','\\textunderscore'),
                        '<NUMBER_OF_SESSIONS>': n_sess,
                        '<NUMBER_OF_ELECTRODES>': n_elecs,
                        '<REPORT_PAGES>': tex_session_pages_str,
                        '<AUC>': '%.2f' % (100*xval_output[-1].auc),
                        '<PERM-P-VALUE>': pvalue_formatting(perm_test_pvalue),
                        '<FR3-AUC>': '%.2f' % (100 * fr3_xval_output[-1].auc),
                        '<FR3-PERM-P-VALUE>': pvalue_formatting(fr3_perm_test_pvalue),
                        '<ROC_AND_TERC_PLOT_FILE>': self.pipeline.subject + '-roc_and_terc_plot.pdf',
                        '<IRT_PLOT_FILE>': task + '-cat' + task + '-' + self.pipeline.subject + '-irt_plot_combined.pdf',
                        '<REPETITION_PLOT_FILE>': task + '-cat' + task + '-' + self.pipeline.subject + '-repetion-ratio-plot.pdf'
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

        xval_output = self.get_passed_object(task + '_xval_output')
        fr3_summary = xval_output[-1]

        panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, labelsize=16, wspace=5.0)

        pd1 = PlotData(x=fr3_summary.fpr, y=fr3_summary.tpr, xlim=[0.0,1.0], ylim=[0.0,1.0], xlabel='False Alarm Rate\n(a)', ylabel='Hit Rate', xlabel_fontsize=20, ylabel_fontsize=20, levelline=((0.0,1.0),(0.0,1.0)), color='k', markersize=1.0)

        pc_diff_from_mean = (fr3_summary.low_pc_diff_from_mean, fr3_summary.mid_pc_diff_from_mean, fr3_summary.high_pc_diff_from_mean)

        ylim = np.max(np.abs(pc_diff_from_mean)) + 5.0
        if ylim > 100.0:
            ylim = 100.0
        pd2 = BarPlotData(x=(0,1,2), y=pc_diff_from_mean, ylim=[-ylim,ylim], xlabel='Tercile of Classifier Estimate\n(b)', ylabel='Recall Change From Mean (%)', x_tick_labels=['Low', 'Middle', 'High'], xlabel_fontsize=20, ylabel_fontsize=20, xhline_pos=0.0, barcolors=['grey','grey', 'grey'], barwidth=0.5)

        panel_plot.add_plot_data(0, 0, plot_data=pd1)
        panel_plot.add_plot_data(0, 1, plot_data=pd2)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + self.pipeline.subject + '-roc_and_terc_plot.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        session_summary_array = self.get_passed_object('session_summary_array')

        serial_positions = np.arange(1,13)

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


            if 'FR4' in task:
                panel_plot = PanelPlot(xfigsize=17, yfigsize=7.5, i_max=1, j_max=3, title='', wspace=3.5, hspace=0.3, labelsize=18)

                pd1 = BarPlotData(x=(0,1,2,3),
                               y=(session_summary.control_mean_prob_diff_all, session_summary.mean_prob_diff_all_post_stim_item, session_summary.control_mean_prob_diff_low, session_summary.mean_prob_diff_low_post_stim_item),
                               yerr=(session_summary.control_sem_prob_diff_all, session_summary.sem_prob_diff_all_post_stim_item, session_summary.control_sem_prob_diff_low, session_summary.sem_prob_diff_low_post_stim_item),
                               x_tick_labels=('Control\nAll', 'Stim\nAll', 'Control\nLow', 'Stim\nLow'),
                               xlabel='(a)', ylabel='$\Delta$ Post-Pre Classifier Output',
                               xhline_pos=0.0, #levelline=[(1.5,0),(1.5,0.1)],
                               barcolors=['grey', 'grey', 'grey', 'grey'], xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5
                              )


                ylim = np.max(np.abs(session_summary.stim_vs_non_stim_pc_diff_from_mean)) + 5.0
                if ylim > 100.0:
                    ylim = 100.0
                pd2 = BarPlotData(x=(0,1), y=session_summary.stim_vs_non_stim_pc_diff_from_mean, ylim=[-ylim,ylim], xlabel='\n(b) Stimulated Item', ylabel='% Recall Difference (Stim-NoStim)', x_tick_labels=['Low', 'High'], xhline_pos=0.0, barcolors=['grey', 'grey'], xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)

                ylim = np.max(np.abs(session_summary.post_stim_vs_non_stim_pc_diff_from_mean)) + 5.0
                if ylim > 100.0:
                    ylim = 100.0
                pd3 = BarPlotData(x=(0,1), y=session_summary.post_stim_vs_non_stim_pc_diff_from_mean, ylim=[-ylim,ylim], xlabel='\n(c) Post-Stimulated Item', ylabel='% Recall Difference (Stim-NoStim)', x_tick_labels=['Low', 'High'], xhline_pos=0.0, barcolors=['grey', 'grey'], xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)

                panel_plot.add_plot_data(0, 0, plot_data=pd1)
                panel_plot.add_plot_data(0, 1, plot_data=pd2)
                panel_plot.add_plot_data(0, 2, plot_data=pd3)

                plot = panel_plot.generate_plot()

                plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-stim_vs_non_stim_halves_plot_' + session_summary.stimtag + '-' + str(session_summary.frequency) + '.pdf')

                plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')
            elif 'FR3' in task:
                panel_plot = PanelPlot(xfigsize=6, yfigsize=7.5, i_max=1, j_max=1, title='', labelsize=18)

                ylim = np.max(np.abs(session_summary.pc_diff_from_mean)) + 5.0
                if ylim > 100.0:
                    ylim = 100.0
                pd = BarPlotData(x=(0,1), y=session_summary.pc_diff_from_mean, ylim=[-ylim,ylim], xlabel='Items', ylabel='% Recall Difference (Stim-NoStim)', x_tick_labels=['Stim', 'PostStim'], xhline_pos=0.0, barcolors=['grey', 'grey'], xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)
                panel_plot.add_plot_data(0, 0, plot_data=pd)

                plot = panel_plot.generate_plot()

                plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-stim_vs_non_stim_halves_plot_' + session_summary.stimtag + '-' + str(session_summary.frequency) + '.pdf')

                plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

            n_lists = len(session_summary.n_stims_per_list)

            xfigsize = 7*n_lists / 25.0
            if xfigsize < 10.0:
                xfigsize = 10.0
            elif xfigsize > 18.0:
                xfigsize = 18.0
            panel_plot = PanelPlot(xfigsize=xfigsize, yfigsize=10.0, i_max=1, j_max=1, title='', xlabel='List', ylabel='# of items', labelsize=20)

            pdc = PlotDataCollection()
            pdc.xlabel = 'List number'
            pdc.xlabel_fontsize = 20
            pdc.ylabel ='#items'
            pdc.ylabel_fontsize = 20

            x_tick_labels = np.array([str(k) for k in session_summary.list_number])
            x_tick_labels[1::5] = ''
            x_tick_labels[2::5] = ''
            x_tick_labels[3::5] = ''
            x_tick_labels[4::5] = ''

            bpd_1 = BarPlotData(x=np.arange(n_lists), y=session_summary.n_stims_per_list, x_tick_labels=x_tick_labels, title='', alpha=0.3)
            stim_x = np.where(session_summary.is_stim_list)[0]
            stim_y = session_summary.n_recalls_per_list[session_summary.is_stim_list]
            pd_1 = PlotData(x=stim_x, y=stim_y, ylim=(0,12),
                    title='', linestyle='', color='red', marker='o',markersize=12)

            nostim_x = np.where(~session_summary.is_stim_list)[0]
            nostim_y = session_summary.n_recalls_per_list[~session_summary.is_stim_list]
            pd_2 = PlotData(x=nostim_x , y=nostim_y , ylim=(0,12),
                    title='', linestyle='', color='blue', marker='o',markersize=12)

            pdc.add_plot_data(pd_1)
            pdc.add_plot_data(pd_2)
            pdc.add_plot_data(bpd_1)

            for i in xrange(len(session_summary.list_number)-1):
                if session_summary.list_number[i] > session_summary.list_number[i+1]:
                    sep_pos = i+0.5
                    sep_plot_data = PlotData(x=[0],y=[0],levelline=[[sep_pos, sep_pos], [0, 12]], color='white', alpha=0.0)
                    pdc.add_plot_data(sep_plot_data)

            panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

            plot = panel_plot.generate_plot()

            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-stim_and_recall_plot_' + session_summary.stimtag + '-' + str(session_summary.frequency) + '.pdf')

            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        irt_within_cat = self.get_passed_object('irt_within_cat')
        irt_between_cat = self.get_passed_object('irt_between_cat')

        panel_plot = PanelPlot(xfigsize=6.0, yfigsize=6.0, i_max=1, j_max=1, title='', xtitle='', labelsize=18)
        pd = BarPlotData(x=[0, 1], y=[np.nanmean(irt_within_cat), np.nanmean(irt_between_cat)],
                         ylabel='IRT (msec)', xlabel='', x_tick_labels=['Within Cat', 'Between Cat'],
                         barcolors=['grey', 'grey'], barwidth=0.5, xlabel_fontsize=18, ylabel_fontsize=18)
        panel_plot.add_plot_data(0, 0, plot_data=pd)
        plot = panel_plot.generate_plot()
        plot_out_fname = self.get_path_to_resource_in_workspace(
            'reports/' + task + '-cat' + task + '-' + subject + '-irt_plot_combined.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        panel_plot = PanelPlot(xfigsize=12.0, yfigsize=6.0, i_max=1, j_max=1, title='', xtitle='', labelsize=18)

        all_repetition_ratios = self.get_passed_object('all_repetition_ratios')
        all_repetition_ratios = all_repetition_ratios[np.isfinite(all_repetition_ratios)]
        all_rr_hist = np.histogram(all_repetition_ratios, range=[0., 1], bins='auto')

        mean_rr = self.get_passed_object('mean_rr')
        stim_mean_rr = self.get_passed_object('stim_mean_rr')
        nostim_mean_rr = self.get_passed_object('nostim_mean_rr')
        pdc = PlotDataCollection()
        hist = BarPlotData(y=all_rr_hist[0], x=all_rr_hist[1][1:], barcolors=['grey' for h in all_rr_hist[0]],
                           xlim=[0, 1],barwidth=0.05, xlabel='Repetition Ratio',
                           ylabel='# of lists', xlabel_fontsize=18, ylabel_fontsize=24)
        mean = PlotData(x=[mean_rr,mean_rr], y=[0, max(all_rr_hist[0])],
                             linecolor = 'black',label = 'All',linestyle='--')
        stim_mean = PlotData(x=[stim_mean_rr,stim_mean_rr], y=[0, max(all_rr_hist[0])],
                             linecolor = 'red',label = 'Stim',linestyle='--')
        nostim_mean = PlotData(x=[nostim_mean_rr,nostim_mean_rr], y=[0, max(all_rr_hist[0])],
                             linecolor = 'blue',label = 'No Stim',linestyle='--')
        pdc.add_plot_data(hist)
        pdc.add_plot_data(mean)
        pdc.add_plot_data(stim_mean)
        pdc.add_plot_data(nostim_mean)
        panel_plot.add_plot_data_collection(0,0,plot_data_collection=pdc)
        plot = panel_plot.generate_plot()
        percentile=np.nanmean(all_repetition_ratios<mean_rr)*100
        plot.annotate(s='{:2}'.format(percentile),xy=(mean_rr,max(all_rr_hist[0])))
        plot.legend()
        plot_out_fname = self.get_path_to_resource_in_workspace(
            'reports/' + task + '-cat' + task + '-' + subject + '-repetion-ratio-plot.pdf')
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

        report_core_file_name, ext = splitext(report_tex_file_name)
        report_file = join(output_directory,report_core_file_name+'.pdf')
        self.pass_object('report_file',report_file)




class DeployReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(DeployReportPDF,self).__init__(mark_as_completed)

    def run(self):
        report_file = self.get_passed_object('report_file')
        self.pipeline.deploy_report(report_path=report_file)
