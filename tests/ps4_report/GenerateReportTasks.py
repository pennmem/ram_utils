from ReportUtils import ReportRamTask
import PlotUtils
import pandas as pd
from os.path import join,splitext
from TextTemplateUtils import replace_template,replace_template_to_string
import numpy as np
from datetime import date
from  TexUtils.latex_table import latex_table
from subprocess import call


class GeneratePlots(ReportRamTask):
    def __init__(self,mark_as_completed):
        super(GeneratePlots, self).__init__(mark_as_completed=mark_as_completed)

    def run(self):
        session_summaries = self.get_passed_object('session_summaries')
        events = self.get_passed_object('ps_events')
        xmax = events.amplitude.max()*(1.1/1000.)
        ymin = events[events.delta_classifier != -999].delta_classifier.min()*1.05
        ymax = events.delta_classifier.max()*1.05
        self.create_dir_in_workspace('reports')
        for session in session_summaries:
            session_summary = session_summaries[session]

            panel_plot = PlotUtils.PanelPlot(i_max = 1, j_max=len(session_summary.info_by_location),labelsize=16,)

            for i,location in enumerate(sorted(session_summary.info_by_location)):

                loc_info = session_summary.info_by_location[location]

                pdc = PlotUtils.PlotDataCollection(xlabel = location, ylabel='Change in classifier (post-pre)' if i==0 else '',
                                                   xlabel_fontsize=18,ylabel_fontsize=18,xlim = (-0.01,xmax),ylim=(ymin,ymax)
)

                pd_enc  = PlotUtils.PlotData(x=loc_info.amplitudes['ENCODING'],y=loc_info.delta_classifiers['ENCODING'],
                                            linestyle='', marker = 'x',color = 'red',label='encoding'
                                             )
                pdc.add_plot_data(pd_enc)

                pd_distr = PlotUtils.PlotData(x = loc_info.amplitudes['DISTRACT'],y=loc_info.delta_classifiers['DISTRACT'],
                                              linestyle='', marker='x',color='green',label='distract',
                                              )
                pdc.add_plot_data(pd_distr)

                pd_retr  = PlotUtils.PlotData(x = loc_info.amplitudes['RETRIEVAL'],y=loc_info.delta_classifiers['RETRIEVAL'],
                                              linestyle='',marker='x',color='blue',label='retrieval',
                                              )
                pdc.add_plot_data(pd_retr)

                pd_sham = PlotUtils.PlotData(x=np.array([0]),y=session_summary.sham_dc,yerr=session_summary.sham_sem,color='black',
                                             label='sham',
                                             )
                pdc.add_plot_data(pd_sham)
                panel_plot.add_plot_data_collection(0,i,plot_data_collection=pdc)

            session_summary.plot_filename = self.get_path_to_resource_in_workspace(join('reports','session_%s_classifier_response_plot.pdf'%session))
            plt = panel_plot.generate_plot()
            plt.legend()
            plt.savefig(session_summary.plot_filename)


class GenerateTex(ReportRamTask):
    def __init__(self,mark_as_completed):
        super(GenerateTex, self).__init__(mark_as_completed)

    def run(self):
        subject=self.pipeline.subject
        task = self.pipeline.task

        session_summaries = self.get_passed_object('session_summaries')
        session_tex  = ''
        for session in session_summaries:
            session_summary = session_summaries[session]
            result_table = pd.DataFrame(columns=['Best Amplitude','Predicted change in classifier (post-pre)','SE','SNR'])
            for location in sorted(session_summary.info_by_location):
                loc_info = session_summary.info_by_location[location]
                result_table.loc[location] = ['{:2.4}'.format(x) for x in [
                    loc_info.best_amplitude,loc_info.best_delta_classifier,loc_info.sem,loc_info.snr
                ]]
            result_table.loc['SHAM'] = [np.nan,'{:2.4}'.format(session_summary.sham_dc),'{:2.4}'.format(session_summary.sham_sem),np.nan]

            session_tex += replace_template_to_string('ps4_session.tex.tpl',{
                '<SESSION>':session,
                '<PS_PLOT_FILE>':session_summary.plot_filename,
                '<CONTEST_TABLE>':result_table.to_latex(),
                '<BEST_LOCATION>':str(session_summary.best_location).replace('_','-'),
                '<AMPLITUDE>':session_summary.best_amplitude,
                '<PVAL>':'{:.3}'.format(session_summary.pval),
                '<TIE>': 'True' if session_summary.tie else 'False',
                '<SHAM_PVAL>':'{:.3}'.format(session_summary.pval_vs_sham),
            })

        report_filename = '%s_%s_ps4_report.tex'%(subject,task)
        replace_template('ps4_report.tex.tpl',self.get_path_to_resource_in_workspace('reports',report_filename),{
            '<SUBJECT>':subject,
            '<TASK>':task,
            '<DATE>':date.today(),
            '<SESSION_DATA>':latex_table(self.get_passed_object('session_data')),
            '<NUMBER_OF_PS4_SESSIONS>':len(session_summaries),
            '<PS4_SESSION_PAGES>':session_tex
            })

        self.pass_object('report_tex_file_name',report_filename)

class GeneratePDF(ReportRamTask):
    def __init__(self,mark_as_completed):
        super(GeneratePDF, self).__init__(mark_as_completed)


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





