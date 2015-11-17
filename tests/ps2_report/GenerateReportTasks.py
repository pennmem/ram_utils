__author__ = 'm'

from RamPipeline import *


class GenerateTex(RamTask):
    def __init__(self, mark_as_completed=True): RamTask.__init__(self, mark_as_completed)

    def run(self):
        import TextTemplateUtils
        import datetime
        tex_template = 'ps2_report.tex.tpl'
        tex_session_template = 'ps2_session.tex.tpl'

        # self.set_file_resources_to_copy('ps2_report.tex')
        self.set_file_resources_to_move('ps2_report.tex', dst='reports')
        # self.set_file_resources_to_copy('deluxetable.sty', dst='reports')

        import numpy as np
        a = np.fromfunction(lambda x, y: (x + 1) * y, shape=(4, 4))

        from TexUtils.matrix2latex import matrix2latex

        tex_session_pages_str = ''
        session_summary_array = self.pipeline.get_passed_object('session_summary_array')

        for session_summary in session_summary_array:
            replace_dict = {'<PLOT_FILE>': 'report_plot_' + session_summary.name + '.pdf',
                            '<STIMTAG>': session_summary.stimtag,
                            '<CONSTANT_NAME>': session_summary.constant_name,
                            '<CONSTANT_VALUE>': session_summary.constant_value,
                            '<CONSTANT_UNIT>': session_summary.constant_unit,
                            '<ISI_MID>': session_summary.isi_mid,
                            '<ISI_HALF_RANGE>': session_summary.isi_half_range,
                            '<PARAMETER1>': session_summary.parameter1,
                            '<PARAMETER2>': session_summary.parameter2
                            }

            tex_session_pages_str += TextTemplateUtils.replace_template_to_string(tex_session_template, replace_dict)
            tex_session_pages_str += '\n'

        session_summary = session_summary_array[0]

        # replace_template_to_string


        session_data_tex_table = matrix2latex(self.pipeline.get_passed_object('SESSION_DATA'), None, "tabular",
                                              alignment='|c|c|c|', headerRow=["Session \\#", "Date", "Length (min)"])


        # print 'session_data_tex_table=\n',session_data_tex_table

        replace_dict = {
            '<SUBJECT_ID>': self.pipeline.subject_id,
            '<EXPERIMENT>': self.pipeline.experiment,
            '<DATE>': datetime.date.today(),
            '<SESSION_DATA>': session_data_tex_table,
            '<NUMBER_OF_SESSIONS>': self.pipeline.get_passed_object('NUMBER_OF_SESSIONS'),
            '<NUMBER_OF_ELECTRODES>': self.pipeline.get_passed_object('NUMBER_OF_ELECTRODES'),
            '<REPORT_PAGES>': tex_session_pages_str,
            '<CUMULATIVE_ISI_MID>': self.pipeline.get_passed_object('CUMULATIVE_ISI_MID'),
            '<CUMULATIVE_ISI_HALF_RANGE>': self.pipeline.get_passed_object('CUMULATIVE_ISI_HALF_RANGE'),
            '<CUMULATIVE_PLOT_FILE>': 'report_plot_Cumulative.pdf',
            '<CUMULATIVE_PARAMETER1>': self.pipeline.get_passed_object('CUMULATIVE_PARAMETER1'),
            '<CUMULATIVE_PARAMETER2>': self.pipeline.get_passed_object('CUMULATIVE_PARAMETER2')
        }

        TextTemplateUtils.replace_template(template_file_name=tex_template, replace_dict=replace_dict)




class GeneratePlots(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        self.create_dir_in_workspace('reports')

        session_summary_array = self.pipeline.get_passed_object('session_summary_array')

        from PlotUtils import PanelPlot
        for session_summary in session_summary_array:
            panel_plot = PanelPlot(i_max=3, j_max=2, title='', y_axis_title='$\Delta$ Post-Pre Stim Biomarker')

            for plot_panel_index, pd in session_summary.plot_data_dict.iteritems():
                print 'plot_panel_index=', plot_panel_index
                print 'pd.x=', pd.x
                print 'pd.y=', pd.y
                plot_letter = chr(ord('a') + 2 * plot_panel_index[0] + plot_panel_index[1])
                panel_plot.add_plot_data(plot_panel_index[0], plot_panel_index[1], x=pd.x, y=pd.y, yerr=pd.yerr,
                                         x_tick_labels=pd.x_tick_labels, title='(' + plot_letter + ')', ylim=pd.ylim)

            plot = panel_plot.generate_plot()
            plot.subplots_adjust(wspace=0.3, hspace=0.3)
            # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')

            plot_out_fname = self.get_path_to_file_in_workspace('reports/report_plot_' + session_summary.name + '.pdf')

            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        cumulative_plot_data_dict = self.pipeline.get_passed_object('cumulative_plot_data_dict')

        panel_plot = PanelPlot(i_max=3, j_max=2, title='', y_axis_title='$\Delta$ Post-Pre Stim Biomarker')

        for plot_panel_index, pd in cumulative_plot_data_dict.iteritems():
            print 'plot_panel_index=', plot_panel_index
            print 'pd.x=', pd.x
            print 'pd.y=', pd.y
            plot_letter = chr(ord('a') + 2 * plot_panel_index[0] + plot_panel_index[1])
            panel_plot.add_plot_data(plot_panel_index[0], plot_panel_index[1], x=pd.x, y=pd.y, yerr=pd.yerr,
                                     x_tick_labels=pd.x_tick_labels, title='(' + plot_letter + ')', ylim=pd.ylim)

        plot = panel_plot.generate_plot()
        plot.subplots_adjust(wspace=0.3, hspace=0.3)
        # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')

        plot_out_fname = self.get_path_to_file_in_workspace('reports/report_plot_Cumulative.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')


class GenerateReportPDF(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        from subprocess import call
        call(['ls', '-l'])
        # call(['module load Tex'], shell=True)
        # call(["module", "load", "Tex"])
        # call(["module load Tex;pdflatex -shell-escape", self.get_path_to_file_in_workspace('reports/R1086M_PS2_report.tex')], shell=True)
        # call(["module load Tex;pdflatex -shell-escape ~/scratch/py_run_7/R1086M/reports/R1086M_PS2_report.tex"], shell=True)
        # call(["module load Tex;pdflatex -shell-escape "+self.get_path_to_file_in_workspace('reports/R1086M_PS2_report.tex')], shell=True)

        texinputs_set_str = r'export TEXINPUTS="' + self.get_path_to_file_in_workspace('reports') + '":$TEXINPUTS;'
        call([texinputs_set_str + "module load Tex;pdflatex -shell-escape " + self.get_path_to_file_in_workspace(
            'reports/ps2_report.tex')], shell=True)



class GenerateTexTable(RamTask):
    def __init__(self):
        RamTask.__init__(self)

    def run(self):
        import numpy as np
        a = np.fromfunction(lambda x, y: (x + 1) * y, shape=(4, 4))
        print a
        print a[:, 1]
        print a[:, 2]
        print a[:, 3]

        import TexUtils
        self.set_file_resources_to_move('mytable.tex', dst='reports')
        TexUtils.generate_tex_table(caption='Numpy_table', header=['col1', 'col2', 'col3'],
                                    columns=[a[:, 1], a[:, 2], a[:, 3]], label='tab:numpy_table')