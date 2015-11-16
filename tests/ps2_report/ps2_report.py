import sys
from os.path import *

sys.path.append(expanduser('~/python/mswat'))

from MatlabUtils import *
from RamPipeline import *


class PS2ReportPipeline(RamPipeline):

    def __init__(self, subject_id, experiment, workspace_dir, matlab_paths=[]):
        RamPipeline.__init__(self)
        self.subject_id = subject_id
        self.experiment = experiment
        self.set_workspace_dir(workspace_dir+self.subject_id)
        self.matlab_paths = matlab_paths
        add_matlab_search_paths(*matlab_paths)


        # self.set_workspace_dir('~/scratch/py_run_4/'+self.subject_id)
        # self.classifier_dir = self.create_dir_in_workspace('biomarker/L2LR/Feat_Freq')


class PrepareBPSTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True): MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # Saves bm_params
        self.eng.PrepareBPS(self.pipeline.subject_id, self.pipeline.workspace_dir)

class CreateParamsTask(MatlabRamTask):
    def __init__(self): MatlabRamTask.__init__(self)

    def run(self):
        # Saves bm_params
        self.eng.CreateParams(self.pipeline.subject_id, self.pipeline.workspace_dir)

class ComputePowersAndClassifierTask(MatlabRamTask):
    def __init__(self): MatlabRamTask.__init__(self)

    def run(self):
        # saves powers and classifier matlab files
        params_path = join(self.get_workspace_dir(),'bm_params.mat')
        # print 'params_path=',params_path
        self.eng.ComputePowersAndClassifier(self.pipeline.subject_id, self.get_workspace_dir(), params_path)

class ComputePowersPSTask(MatlabRamTask):
    def __init__(self): MatlabRamTask.__init__(self)

    def run(self):
        # saves powers and classifier matlab files
        params_path = join(self.get_workspace_dir(),'bm_params.mat')
        # print 'params_path=',params_path
        self.eng.ComputePowersPS(self.pipeline.subject_id, self.get_workspace_dir())


class SaveEventsTask(MatlabRamTask):
    def __init__(self,mark_as_completed=True): MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        #saves GroupPSL.mat and PS2Events.mat
        self.eng.SaveEvents(self.pipeline.subject_id, self.pipeline.experiment, self.get_workspace_dir())


# class GenerateTex(RamTask):
#     def __init__(self, mark_as_completed=True): RamTask.__init__(self, mark_as_completed)
#
#     def run(self):
#         import TextTemplateUtils
#         import datetime
#         tex_template = 'report.tex.tpl'
#
#         # self.set_file_resources_to_copy('ps2_report.tex')
#         self.set_file_resources_to_move('report.tex', dst='reports')
#         self.set_file_resources_to_copy('deluxetable.sty', dst='reports')
#
#         import numpy as np
#         a = np.fromfunction(lambda x,y: (x+1)*y, shape=(4,4))
#
#         import TexUtils
#         patient_table = TexUtils.generate_tex_table(caption='Numpy_table', header=['col1', 'col2', 'col3'], columns=[ a[:, 1] , a[:, 2], a[:, 3] ], label='tab:numpy_table')
#         print 'patient_table=\n',patient_table
#
#         replace_dict={
#             '<HEADER_LEFT>':'RAM FR1 report v 2.0',
#             '<DATE>': str(datetime.date.today()),
#             '<SECTION_TITLE>': 'R1074M RAM FR1 Free Recall Report',
#             '<PATIENT_TABLE>': patient_table,
#             # '<PT>': r'\begin'
#         }
#
#         TextTemplateUtils.replace_template(template_file_name=tex_template, replace_dict=replace_dict)

# class GenerateTex(RamTask):
#     def __init__(self, mark_as_completed=True): RamTask.__init__(self, mark_as_completed)
#
#     def run(self):
#         import TextTemplateUtils
#         import datetime
#         tex_template = 'ps2_report.tex.tpl'
#
#         # self.set_file_resources_to_copy('ps2_report.tex')
#         self.set_file_resources_to_move('ps2_report.tex', dst='reports')
#         # self.set_file_resources_to_copy('deluxetable.sty', dst='reports')
#
#         import numpy as np
#         a = np.fromfunction(lambda x,y: (x+1)*y, shape=(4,4))
#
#         from TexUtils.matrix2latex import matrix2latex
#
#         # patient_table_data=[[1, "21 Oct 2015", "30"],[5, "27 Oct 2015", "28"]]
#         # patient_table_tex = matrix2latex(patient_table_data, None, "tabular", alignment='|c|c|c|', headerRow=["Session \\#", "Date", "Length (min)"])
#
#         session_data_tex_table = matrix2latex(self.pipeline.get_passed_object('SESSION_DATA'), None, "tabular", alignment='|c|c|c|', headerRow=["Session \\#", "Date", "Length (min)"])
#
#             # TexUtils.generate_tex_table(caption='Numpy_table', header=['col1', 'col2', 'col3'], columns=[ a[:, 1] , a[:, 2], a[:, 3] ], label='tab:numpy_table')
#         print 'session_data_tex_table=\n',session_data_tex_table
#
#         replace_dict={
#             '<SUBJECT_ID>':self.pipeline.subject_id,
#             '<DATE>': str(datetime.date.today()),
#             # '<SECTION_TITLE>': 'R1074M RAM FR1 Free Recall Report',
#             '<SESSION_DATA>': session_data_tex_table,
#             '<DURATION>': self.pipeline.get_passed_object('DURATION'),
#             '<ISI_MID>': self.pipeline.get_passed_object('ISI_MID'),
#             '<ISI_HALF_RANGE>': self.pipeline.get_passed_object('ISI_HALF_RANGE'),
#             '<NUMBER_OF_SESSIONS>':self.pipeline.get_passed_object('NUMBER_OF_SESSIONS'),
#             '<NUMBER_OF_ELECTRODES>':self.pipeline.get_passed_object('NUMBER_OF_ELECTRODES'),
#             '<STIMTAG>': self.pipeline.get_passed_object('STIMTAG'),
#             # '<PT>': r'\begin'
#         }
#
#         TextTemplateUtils.replace_template(template_file_name=tex_template, replace_dict=replace_dict)



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
        a = np.fromfunction(lambda x,y: (x+1)*y, shape=(4,4))

        from TexUtils.matrix2latex import matrix2latex


        tex_session_pages_str = ''
        session_summary_array = self.pipeline.get_passed_object('session_summary_array')

        for session_summary in session_summary_array:
            replace_dict = {'<PLOT_FILE>':'report_plot_'+session_summary.name+'.pdf',
                            '<STIMTAG>': session_summary.stimtag,
                            '<CONSTANT_NAME>': session_summary.constant_name,
                            '<CONSTANT_VALUE>': session_summary.constant_value,
                            '<CONSTANT_UNIT>': session_summary.constant_unit,
                            '<ISI_MID>': session_summary.isi_mid,
                            '<ISI_HALF_RANGE>': session_summary.isi_half_range,
                            '<PARAMETER1>': session_summary.parameter1,
                            '<PARAMETER2>': session_summary.parameter2
                            }

            tex_session_pages_str += TextTemplateUtils.replace_template_to_string(tex_session_template,replace_dict)
            tex_session_pages_str += '\n'

        session_summary = session_summary_array[0]

        # replace_template_to_string


        session_data_tex_table = matrix2latex(self.pipeline.get_passed_object('SESSION_DATA'), None, "tabular", alignment='|c|c|c|', headerRow=["Session \\#", "Date", "Length (min)"])


        # print 'session_data_tex_table=\n',session_data_tex_table

        replace_dict={
            '<SUBJECT_ID>':self.pipeline.subject_id,
            '<EXPERIMENT>':self.pipeline.experiment,
            '<DATE>': datetime.date.today(),
            '<SESSION_DATA>': session_data_tex_table,
            '<NUMBER_OF_SESSIONS>':self.pipeline.get_passed_object('NUMBER_OF_SESSIONS'),
            '<NUMBER_OF_ELECTRODES>':self.pipeline.get_passed_object('NUMBER_OF_ELECTRODES'),
            '<REPORT_PAGES>':tex_session_pages_str,
            '<CUMULATIVE_ISI_MID>':self.pipeline.get_passed_object('CUMULATIVE_ISI_MID'),
            '<CUMULATIVE_ISI_HALF_RANGE>':self.pipeline.get_passed_object('CUMULATIVE_ISI_HALF_RANGE'),
            '<CUMULATIVE_PLOT_FILE>':'report_plot_Cumulative.pdf',
            '<CUMULATIVE_PARAMETER1>':self.pipeline.get_passed_object('CUMULATIVE_PARAMETER1'),
            '<CUMULATIVE_PARAMETER2>':self.pipeline.get_passed_object('CUMULATIVE_PARAMETER2')
        }

        TextTemplateUtils.replace_template(template_file_name=tex_template, replace_dict=replace_dict)



class ExtractWeightsTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True):
        MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # saves Weights.mat
        from MatlabIO import deserialize_single_object_from_matlab_format, serialize_objects_in_matlab_format
        # classifier_output_file_name = 'R1086M_RAM_FR1_L2LR_Freq_Time-Enc_CV-list_Pen-154.45.mat'

        classifier_output_location = 'biomarker/L2LR/Feat_Freq'
        from glob import glob
        self.get_path_to_file_in_workspace(classifier_output_location)

        classifier_files = glob(self.get_path_to_file_in_workspace(classifier_output_location)+'/*.mat')
        try:
            classifier_output_file_name_full = classifier_files[0] # picking first file, there shuld be just one file there!
        except IndexError:
            print 'Could not locate *.mat in '+self.get_path_to_file_in_workspace(classifier_output_location)
            sys.exit()

        # classifier_output_file_name_full = join(self.get_workspace_dir(),classifier_output_location, classifier_output_file_name )

        # ----classifier_output_file_name_full = self.get_path_to_file_in_workspace(classifier_output_location, classifier_output_file_name)

        res = deserialize_single_object_from_matlab_format(classifier_output_file_name_full,'res')

        serialize_objects_in_matlab_format(self.get_path_to_file_in_workspace('Weights.mat'), (res.Weights,'Weights'))
        # save weights in matlab format
        print 'res.Weights=',res.Weights
        # print 'res.W0=',res.W0



class GenerateTexTable(RamTask):
    def __init__(self):
        RamTask.__init__(self)


    def run(self):
        import numpy as np
        a = np.fromfunction(lambda x,y: (x+1)*y, shape=(4,4))
        print a
        print a[:,1]
        print a[:,2]
        print a[:,3]

        import TexUtils
        self.set_file_resources_to_move('mytable.tex', dst='reports')
        TexUtils.generate_tex_table(caption='Numpy_table', header=['col1', 'col2', 'col3'], columns=[ a[:, 1] , a[:, 2], a[:, 3] ], label='tab:numpy_table')


class GeneratePlots(RamTask):
    def __init__(self,mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        self.create_dir_in_workspace('reports')

        session_summary_array = self.pipeline.get_passed_object('session_summary_array')

        from PlotUtils import PanelPlot
        for session_summary in session_summary_array:
            panel_plot = PanelPlot(i_max=3, j_max=2, title='', y_axis_title='$\Delta$ Post-Pre Stim Biomarker')

            for plot_panel_index, pd in session_summary.plot_data_dict.iteritems():
                print 'plot_panel_index=',plot_panel_index
                print 'pd.x=',pd.x
                print 'pd.y=',pd.y
                plot_letter = chr(ord('a')+2*plot_panel_index[0]+plot_panel_index[1])
                panel_plot.add_plot_data(plot_panel_index[0], plot_panel_index[1], x=pd.x, y=pd.y, yerr=pd.yerr, x_tick_labels=pd.x_tick_labels, title='('+plot_letter+')', ylim=pd.ylim)

            plot = panel_plot.generate_plot()
            plot.subplots_adjust(wspace=0.3, hspace=0.3)
            # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')

            plot_out_fname = self.get_path_to_file_in_workspace('reports/report_plot_'+session_summary.name+'.pdf')

            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

        cumulative_plot_data_dict = self.pipeline.get_passed_object('cumulative_plot_data_dict')

        panel_plot = PanelPlot(i_max=3, j_max=2, title='', y_axis_title='$\Delta$ Post-Pre Stim Biomarker')

        for plot_panel_index, pd in cumulative_plot_data_dict.iteritems():
            print 'plot_panel_index=',plot_panel_index
            print 'pd.x=',pd.x
            print 'pd.y=',pd.y
            plot_letter = chr(ord('a')+2*plot_panel_index[0]+plot_panel_index[1])
            panel_plot.add_plot_data(plot_panel_index[0], plot_panel_index[1], x=pd.x, y=pd.y, yerr=pd.yerr, x_tick_labels=pd.x_tick_labels, title='('+plot_letter+')', ylim=pd.ylim)

        plot = panel_plot.generate_plot()
        plot.subplots_adjust(wspace=0.3, hspace=0.3)
        # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')

        plot_out_fname = self.get_path_to_file_in_workspace('reports/report_plot_Cumulative.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')




        #     plot_specs = self.pipeline.get_passed_object('amp_all')
        #     print 'plot_specs=',plot_specs
        #     # panel_plot.add_plot_data(0, 0, plot_specs.x, plot_specs.y, yerr=plot_specs.yerr, title='(a)')
        #     panel_plot.add_plot_data(0, 0, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(a)', ylim=plot_specs.ylim)
        #
        #
        #
        # from PlotUtils import PanelPlot
        # import numpy as np
        # panel_plot = PanelPlot(i_max=3, j_max=2, title='', x_axis_title='Stimulation Amplitude (mA)', y_axis_title='$\Delta$ Post-Pre Stim Biomarker')
        #
        # plot_specs = self.pipeline.get_passed_object('amp_all')
        # print 'plot_specs=',plot_specs
        # # panel_plot.add_plot_data(0, 0, plot_specs.x, plot_specs.y, yerr=plot_specs.yerr, title='(a)')
        # panel_plot.add_plot_data(0, 0, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(a)', ylim=plot_specs.ylim)
        #
        # plot_specs = self.pipeline.get_passed_object('amp_low')
        # panel_plot.add_plot_data(1, 0, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(c)', ylim=plot_specs.ylim)
        #
        # plot_specs = self.pipeline.get_passed_object('amp_high')
        # panel_plot.add_plot_data(2, 0, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(e)', ylim=plot_specs.ylim)
        #
        # plot_specs = self.pipeline.get_passed_object('freq_all')
        # panel_plot.add_plot_data(0, 1, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(b)', ylim=plot_specs.ylim)
        #
        # plot_specs = self.pipeline.get_passed_object('freq_low')
        # panel_plot.add_plot_data(1, 1, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(d)', ylim=plot_specs.ylim)
        #
        # plot_specs = self.pipeline.get_passed_object('freq_high')
        # panel_plot.add_plot_data(2, 1, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(f)', ylim=plot_specs.ylim)
        #
        # plot = panel_plot.generate_plot()
        # plot.subplots_adjust(wspace=0.3, hspace=0.3)
        # # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')
        #
        # plot_out_fname = self.get_path_to_file_in_workspace('reports/report_plot_session.pdf')
        #
        # plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')
        #
        # # ------------------------------------------- Combined accross sessions
        #
        #
        # panel_plot = PanelPlot(i_max=3, j_max=2, title='All Sessions combined', x_axis_title='Stimulation Amplitude (mA)', y_axis_title='$\Delta$ Post-Pre Stim Biomarker')
        #
        # plot_specs = self.pipeline.get_passed_object('tot_amp_all')
        # print 'plot_specs=',plot_specs
        # # panel_plot.add_plot_data(0, 0, plot_specs.x, plot_specs.y, yerr=plot_specs.yerr, title='(a)')
        # panel_plot.add_plot_data(0, 0, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(a)', ylim=plot_specs.ylim)
        #
        # plot_specs = self.pipeline.get_passed_object('tot_amp_low')
        # panel_plot.add_plot_data(1, 0, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(c)', ylim=plot_specs.ylim)
        #
        # plot_specs = self.pipeline.get_passed_object('tot_amp_high')
        # panel_plot.add_plot_data(2, 0, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(e)', ylim=plot_specs.ylim)
        #
        # plot_specs = self.pipeline.get_passed_object('tot_freq_all')
        # panel_plot.add_plot_data(0, 1, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(b)', ylim=plot_specs.ylim)
        #
        # plot_specs = self.pipeline.get_passed_object('tot_freq_low')
        # panel_plot.add_plot_data(1, 1, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(d)', ylim=plot_specs.ylim)
        #
        # plot_specs = self.pipeline.get_passed_object('tot_freq_high')
        # panel_plot.add_plot_data(2, 1, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(f)', ylim=plot_specs.ylim)
        #
        # plot = panel_plot.generate_plot()
        # plot.subplots_adjust(wspace=0.3, hspace=0.3)
        # # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')
        #
        # plot_out_fname = self.get_path_to_file_in_workspace('reports/report_plot_tot.pdf')
        #
        # plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')



class GenerateReportPDF(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        from subprocess import call
        call(['ls','-l'])
        # call(['module load Tex'], shell=True)
        # call(["module", "load", "Tex"])
        # call(["module load Tex;pdflatex -shell-escape", self.get_path_to_file_in_workspace('reports/R1086M_PS2_report.tex')], shell=True)
        # call(["module load Tex;pdflatex -shell-escape ~/scratch/py_run_7/R1086M/reports/R1086M_PS2_report.tex"], shell=True)
        # call(["module load Tex;pdflatex -shell-escape "+self.get_path_to_file_in_workspace('reports/R1086M_PS2_report.tex')], shell=True)

        texinputs_set_str = r'export TEXINPUTS="'+self.get_path_to_file_in_workspace('reports')+'":$TEXINPUTS;'
        call([texinputs_set_str+"module load Tex;pdflatex -shell-escape "+self.get_path_to_file_in_workspace('reports/ps2_report.tex')], shell=True)


ps_report_pipeline = PS2ReportPipeline(subject_id='R1056M', experiment='PS1', workspace_dir='/scratch/busygin/py_run_8/', matlab_paths=['~/eeg','~/matlab/beh_toolbox','~/RAM/RAM_reporting','~/RAM/RAM_sys2Biomarkers','.'])

#ps_report_pipeline.add_task(CreateParamsTask())

#ps_report_pipeline.add_task(ComputePowersAndClassifierTask())

ps_report_pipeline.add_task(ComputePowersPSTask())

ps_report_pipeline.add_task(SaveEventsTask())

ps_report_pipeline.add_task(PrepareBPSTask())
ps_report_pipeline.add_task(ExtractWeightsTask(mark_as_completed=False))

########################## UNCOMMENT


from PSReportingTask import PSReportingTask
ps_report_pipeline.add_task(PSReportingTask(mark_as_completed=False))


ps_report_pipeline.add_task(GeneratePlots(mark_as_completed=False))


ps_report_pipeline.add_task(GenerateTex(mark_as_completed=False))

ps_report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))


# ps_report_pipeline.add_task(GenerateTexTable())
########################## UNCOMMENT

ps_report_pipeline.execute_pipeline()
