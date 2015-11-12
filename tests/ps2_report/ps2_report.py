import sys
from os.path import *

sys.path.append(expanduser('~/RAM_UTILS_GIT'))

from  MatlabUtils import *
from RamPipeline import *


class PS2ReportPipeline(RamPipeline):

    def __init__(self,subject_id, workspace_dir,matlab_paths=[]):
        RamPipeline.__init__(self)
        self.subject_id = subject_id
        self.set_workspace_dir(workspace_dir+self.subject_id)
        self.matlab_paths = matlab_paths
        add_matlab_search_paths(*matlab_paths)


        # self.subject_id = 'R1086M'

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

class SaveEventsTask(MatlabRamTask):
    def __init__(self): MatlabRamTask.__init__(self)

    def run(self):
        #saves GroupPSL.mat and PS2Events.mat
        self.eng.SaveEvents(self.pipeline.subject_id, self.get_workspace_dir())


class GenerateTex(RamTask):
    def __init__(self, mark_as_completed=True): RamTask.__init__(self, mark_as_completed)

    def run(self):
        import TextTemplateUtils
        import datetime
        tex_template = 'report.tex.tpl'

        # self.set_file_resources_to_copy('ps2_report.tex')
        self.set_file_resources_to_move('report.tex', dst='reports')
        self.set_file_resources_to_copy('deluxetable.sty', dst='reports')

        import numpy as np
        a = np.fromfunction(lambda x,y: (x+1)*y, shape=(4,4))

        import TexUtils
        patient_table = TexUtils.generate_tex_table(caption='Numpy_table', header=['col1', 'col2', 'col3'], columns=[ a[:, 1] , a[:, 2], a[:, 3] ], label='tab:numpy_table')
        print 'patient_table=\n',patient_table

        replace_dict={
            '<HEADER_LEFT>':'RAM FR1 report v 2.0',
            '<DATE>': str(datetime.date.today()),
            '<SECTION_TITLE>': 'R1074M RAM FR1 Free Recall Report',
            '<PATIENT_TABLE>': patient_table,
            # '<PT>': r'\begin'
        }

        TextTemplateUtils.replace_template(template_file_name=tex_template, replace_dict=replace_dict)


class ExtractWeightsTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True):
        MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # saves Weights.mat
        from MatlabIO import deserialize_single_object_from_matlab_format, serialize_objects_in_matlab_format
        classifier_output_file_name = 'R1086M_RAM_FR1_L2LR_Freq_Time-Enc_CV-list_Pen-154.45.mat'
        classifier_output_location = 'biomarker/L2LR/Feat_Freq'
        # classifier_output_file_name_full = join(self.get_workspace_dir(),classifier_output_location, classifier_output_file_name )
        classifier_output_file_name_full = self.get_path_to_file_in_workspace(classifier_output_location, classifier_output_file_name)

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
        from PlotUtils import PanelPlot
        import numpy as np
        panel_plot = PanelPlot(i_max=3, j_max=2, title='', x_axis_title='Stimulation Amplitude (mA)', y_axis_title='$\Delta$ Post-Pre Stim Biomarker')

        plot_specs = self.pipeline.get_passed_object('amp_all')
        print 'plot_specs=',plot_specs
        # panel_plot.add_plot_data(0, 0, plot_specs.x, plot_specs.y, yerr=plot_specs.yerr, title='(a)')
        panel_plot.add_plot_data(0, 0, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(a)', ylim=plot_specs.ylim)

        plot_specs = self.pipeline.get_passed_object('amp_low')
        panel_plot.add_plot_data(1, 0, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(c)', ylim=plot_specs.ylim)

        plot_specs = self.pipeline.get_passed_object('amp_high')
        panel_plot.add_plot_data(2, 0, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(e)', ylim=plot_specs.ylim)

        plot_specs = self.pipeline.get_passed_object('freq_all')
        panel_plot.add_plot_data(0, 1, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(b)', ylim=plot_specs.ylim)

        plot_specs = self.pipeline.get_passed_object('freq_low')
        panel_plot.add_plot_data(1, 1, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(d)', ylim=plot_specs.ylim)

        plot_specs = self.pipeline.get_passed_object('freq_high')
        panel_plot.add_plot_data(2, 1, x=plot_specs.x, y=plot_specs.y, yerr=plot_specs.yerr, x_tick_labels=plot_specs.x_tick_labels, title='(f)', ylim=plot_specs.ylim)




        # pd = self.pipeline.get_passed_object('amp_low')
        # panel_plot.add_plot_data(1, 0, x=pd[0], y=pd[1], yerr=pd[2], title='(c)')
        #
        # pd = self.pipeline.get_passed_object('amp_high')
        # panel_plot.add_plot_data(2, 0, x=pd[0], y=pd[1], yerr=pd[2], title='(e)')


        # pd = self.pipeline.get_passed_object('freq_all')
        # panel_plot.add_plot_data(0, 1, x=pd[0], y=pd[1], yerr=pd[2], title='(b)')
        #
        # pd = self.pipeline.get_passed_object('freq_low')
        # panel_plot.add_plot_data(1, 1, x=pd[0], y=pd[1], yerr=pd[2], title='(d)')
        #
        # pd = self.pipeline.get_passed_object('freq_high')
        # panel_plot.add_plot_data(2, 1, x=pd[0], y=pd[1], yerr=pd[2], title='(f)')



        # panel_plot.add_plot_data(0, 1, x=np.arange(10), y=np.random.rand(10), title='data01')
        # panel_plot.add_plot_data(1, 0, x=np.arange(10), y=np.random.rand(10), title='data10')
        # panel_plot.add_plot_data(1, 1, x=np.arange(10), y=np.random.rand(10), yerr=np.random.rand(10), title='data11')
        plot = panel_plot.generate_plot()
        plot.subplots_adjust(wspace=0.3, hspace=0.3)
        # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')

        plot_out_fname = self.get_path_to_file_in_workspace('reports/report_plot.pdf')

        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')
        # plot.savefig('demo.png')
        # plot.show()



        # panel_plot = PanelPlot(i_max=2, j_max=2, title='Random Data', x_axis_title='x_axis_label', y_axis_title='y_axis_random')
        #
        # panel_plot.add_plot_data(0, 0, x=np.arange(10), y=np.random.rand(10), title='data00')
        # panel_plot.add_plot_data(0, 1, x=np.arange(10), y=np.random.rand(10), title='data01')
        # panel_plot.add_plot_data(1, 0, x=np.arange(10), y=np.random.rand(10), title='data10')
        # panel_plot.add_plot_data(1, 1, x=np.arange(10), y=np.random.rand(10), yerr=np.random.rand(10), title='data11')
        # plot = panel_plot.generate_plot()
        # plot.subplots_adjust(wspace=0.3, hspace=0.3)
        # # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')
        #
        # plot_out_fname = self.get_path_to_file_in_workspace('reports/demo_2.pdf')
        #
        # plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')
        # # plot.savefig('demo.png')
        # # plot.show()

class GenerateReportPDF(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        from subprocess import call
        call(['ls','-l'])
        call(['module load Tex'], shell=True)
        # call(["module", "load", "Tex"])
        call(["pdflatex", self.get_path_to_file_in_workspace('reports/report.tex')])





a = 'my \n string'

print a.encode('string-escape')

ps_report_pipeline = PS2ReportPipeline(subject_id='R1086M', workspace_dir='~/scratch/py_run_7/', matlab_paths=['~/RAM_MATLAB','.'])

# ps_report_pipeline.add_task(CreateParamsTask())
#
# ps_report_pipeline.add_task(ComputePowersAndClassifierTask())
#
# ps_report_pipeline.add_task(SaveEventsTask())
#
# ps_report_pipeline.add_task(PrepareBPSTask())
# ps_report_pipeline.add_task(ExtractWeightsTask(mark_as_completed=False))

########################## UNCOMMENT
# ps_report_pipeline.add_task(GenerateTex(mark_as_completed=False))

from PSReportingTask import PSReportingTask
ps_report_pipeline.add_task(PSReportingTask(mark_as_completed=False))


ps_report_pipeline.add_task(GeneratePlots(mark_as_completed=False))





# ps_report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))


# ps_report_pipeline.add_task(GenerateTexTable())
########################## UNCOMMENT

ps_report_pipeline.execute_pipeline()