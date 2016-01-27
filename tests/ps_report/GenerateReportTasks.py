__author__ = 'm'

from RamPipeline import *

import numpy as np

from PlotUtils import PlotData, BarPlotData, PlotDataCollection, PanelPlot

from latex_table import latex_table


def pvalue_formatting(p):
    return ('%.3f'%p) if p>0.001 else '\leq 0.001'


class GenerateTex(RamTask):
    def __init__(self, mark_as_completed=True): RamTask.__init__(self, mark_as_completed)

    def run(self):
        import TextTemplateUtils
        import datetime
        tex_template = 'ps_report.tex.tpl'
        tex_session_template = 'ps_session.tex.tpl'

        report_tex_file_name = self.pipeline.experiment + '-' + self.pipeline.subject + '-' + 'report.tex'
        self.pass_object('report_tex_file_name',report_tex_file_name)

        self.set_file_resources_to_move(report_tex_file_name, dst='reports')

        tex_session_pages_str = ''
        session_summary_array = self.get_passed_object('session_summary_array')

        for session_summary in session_summary_array:
            replace_dict = {'<SESS_NUM>': session_summary.sess_num,
                            '<PLOT_FILE>': self.pipeline.experiment + '-' + self.pipeline.subject + '-report_plot_' + session_summary.name + '.pdf',
                            '<STIMTAG>': session_summary.stimtag,
                            '<CONSTANT_NAME>': session_summary.constant_name,
                            '<CONSTANT_VALUE>': session_summary.constant_value,
                            '<CONSTANT_UNIT>': session_summary.constant_unit,
                            '<ISI_MID>': session_summary.isi_mid,
                            '<ISI_HALF_RANGE>': session_summary.isi_half_range,
                            '<PARAMETER1>': session_summary.parameter1,
                            '<PARAMETER2>': session_summary.parameter2,
                            '<FVALUE1>': '%.2f' % session_summary.anova_fvalues[0],
                            '<FVALUE2>': '%.2f' % session_summary.anova_fvalues[1],
                            '<FVALUE12>': '%.2f' % session_summary.anova_fvalues[2],
                            '<PVALUE1>': pvalue_formatting(session_summary.anova_pvalues[0]),
                            '<PVALUE2>': pvalue_formatting(session_summary.anova_pvalues[1]),
                            '<PVALUE12>': pvalue_formatting(session_summary.anova_pvalues[2])
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
            '<CUMULATIVE_ISI_MID>': self.get_passed_object('CUMULATIVE_ISI_MID'),
            '<CUMULATIVE_ISI_HALF_RANGE>': self.get_passed_object('CUMULATIVE_ISI_HALF_RANGE'),
            '<CUMULATIVE_PLOT_FILE>': self.pipeline.experiment + '-' + self.pipeline.subject + '-report_plot_Cumulative.pdf',
            '<CUMULATIVE_PARAMETER1>': self.get_passed_object('CUMULATIVE_PARAMETER1'),
            '<CUMULATIVE_PARAMETER2>': self.get_passed_object('CUMULATIVE_PARAMETER2'),
            '<AUC>': '%.2f' % (100*xval_output[-1].auc),
            '<PERM-P-VALUE>': ('= %.3f' % perm_test_pvalue) if perm_test_pvalue>=0.001 else '\leq 0.001',
            '<J-THRESH>': '%.3f' % xval_output[-1].jstat_thresh,
            '<ROC_AND_TERC_PLOT_FILE>': self.pipeline.task + '-' + self.pipeline.subject + '-roc_and_terc_plot_combined.pdf'
        }


        TextTemplateUtils.replace_template(template_file_name=tex_template, out_file_name=report_tex_file_name, replace_dict=replace_dict)




class GeneratePlots(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        experiment = self.pipeline.experiment

        self.create_dir_in_workspace('reports')

        xval_output = self.get_passed_object('xval_output')
        fr1_summary = xval_output[-1]

        panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, title='', wspace=0.3, hspace=0.3)

        pd1 = PlotData(x=fr1_summary.fpr, y=fr1_summary.tpr, xlim=[0.0,1.0], ylim=[0.0,1.0], xlabel='False Alarm Rate\n(a)', ylabel='Hit Rate', levelline=((0.0,1.0),(0.0,1.0)))

        pc_diff_from_mean = (fr1_summary.low_pc_diff_from_mean, fr1_summary.mid_pc_diff_from_mean, fr1_summary.high_pc_diff_from_mean)

        ylim = np.max(np.abs(pc_diff_from_mean)) + 5.0
        if ylim > 100.0:
            ylim = 100.0
        pd2 = BarPlotData(x=(0,1,2), y=pc_diff_from_mean, ylim=[-ylim,ylim], xlabel='Tercile of Classifier Estimate\n(b)', ylabel='Recall Change From Mean (%)', x_tick_labels=['Low', 'Middle', 'High'], xhline_pos=0.0, barcolors=['grey','grey', 'grey'], barwidth=0.5)

        panel_plot.add_plot_data(0, 0, plot_data=pd1)
        panel_plot.add_plot_data(0, 1, plot_data=pd2)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + self.pipeline.task + '-' + self.pipeline.subject + '-roc_and_terc_plot_combined.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        session_summary_array = self.get_passed_object('session_summary_array')
        param1_name = self.get_passed_object('CUMULATIVE_PARAMETER1')
        param1_unit = self.get_passed_object('CUMULATIVE_UNIT1')
        param1_title = '%s (%s)' % (param1_name,param1_unit)

        for session_summary in session_summary_array:
            panel_plot = PanelPlot(i_max=1, j_max=1, title='', xtitle=param1_title, ytitle='$\Delta$ Post-Pre Classifier Output', wspace=0.3, hspace=0.3)

            pdc = PlotDataCollection(legend_on=True)
            
            for v,p in session_summary.plots.iteritems():
                p.xhline_pos=0.0
                pdc.add_plot_data(p)

            panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

            plot = panel_plot.generate_plot()

            plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + self.pipeline.experiment + '-' + self.pipeline.subject + '-report_plot_' + session_summary.name + '.pdf')

            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        cumulative_plots = self.get_passed_object('cumulative_plots')

        panel_plot = PanelPlot(i_max=1, j_max=1, title='', xtitle=param1_title, y_axis_title='$\Delta$ Post-Pre Classifier Output')

        pdc = PlotDataCollection(legend_on=True)

        for v,p in cumulative_plots.iteritems():
            p.xhline_pos=0.0
            pdc.add_plot_data(p)

        panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

        plot = panel_plot.generate_plot()

        plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + self.pipeline.experiment + '-' + self.pipeline.subject + '-report_plot_Cumulative.pdf')

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
