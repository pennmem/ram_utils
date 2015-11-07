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

class CreateParamsTask(MatlabRamTask):
    def __init__(self): MatlabRamTask.__init__(self)

    def run(self):

        self.eng.CreateParams(self.pipeline.subject_id, self.pipeline.workspace_dir)

class ComputePowersAndClassifierTask(MatlabRamTask):
    def __init__(self): MatlabRamTask.__init__(self)

    def run(self):

        params_path = join(self.get_workspace_dir(),'bm_params.mat')
        # print 'params_path=',params_path
        self.eng.ComputePowersAndClassifier(self.pipeline.subject_id, self.get_workspace_dir(), params_path)


ps_report_pipeline = PS2ReportPipeline(subject_id='R1086M', workspace_dir='~/scratch/py_run_6/', matlab_paths=['~/RAM_MATLAB','.'])

ps_report_pipeline.add_task(CreateParamsTask())

ps_report_pipeline.add_task(ComputePowersAndClassifierTask())

ps_report_pipeline.execute_pipeline()