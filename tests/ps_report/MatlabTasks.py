__author__ = 'm'
import sys
from os.path import *
from RamPipeline.MatlabRamTask import *


class CreateParamsTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True): MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # Saves bm_params
        self.eng.CreateParamsPy(self.pipeline.subject_id, self.pipeline.workspace_dir)


class ComputePowersAndClassifierTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True): MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # saves powers and classifier matlab files
        params_path = join(self.get_workspace_dir(), 'bm_params.mat')
        # print 'params_path=',params_path
        self.eng.ComputePowersAndClassifierPy(self.pipeline.subject_id, self.get_workspace_dir(), params_path)

class ComputePowersPSTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True): MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # saves powers and classifier matlab files
        params_path = join(self.get_workspace_dir(), 'bm_params.mat')
        # print 'params_path=',params_path
        self.eng.ComputePowersPSPy(self.pipeline.subject_id, self.get_workspace_dir())

class SaveEventsTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True): MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # saves GroupPSL.mat and PS2Events.mat
        self.eng.SaveEventsPy(self.pipeline.subject_id, self.pipeline.experiment, self.get_workspace_dir())



class PrepareBPSTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True): MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # Saves bm_params
        self.eng.PrepareBPSPy(self.pipeline.subject_id, self.pipeline.workspace_dir)


# class ExtractWeightsTask(MatlabRamTask):
#     def __init__(self, mark_as_completed=True):
#         MatlabRamTask.__init__(self, mark_as_completed)

