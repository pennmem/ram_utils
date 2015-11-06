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

class CreateParamsTask(RamTask):
    def __init__(self):
        RamTask.__init__(self)

    def run(self):

        from MatlabUtils import matlab_engine as eng

        eng.CreateParams(self.pipeline.subject_id, self.pipeline.workspace_dir)




ps_report_pipeline = PS2ReportPipeline(subject_id='R1086M', workspace_dir='~/scratch/py_run_5/', matlab_paths=['~/RAM_MATLAB'])

ps_report_pipeline.add_task(CreateParamsTask())

ps_report_pipeline.execute_pipeline()