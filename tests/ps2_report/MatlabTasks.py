__author__ = 'm'
import sys
from os.path import *
from RamPipeline.MatlabRamTask import *


class PrepareBPSTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True): MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # Saves bm_params
        self.eng.PrepareBPS(self.pipeline.subject_id, self.pipeline.workspace_dir)


class CreateParamsTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True): MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # Saves bm_params
        self.eng.CreateParams(self.pipeline.subject_id, self.pipeline.workspace_dir)


class ComputePowersAndClassifierTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True): MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # saves powers and classifier matlab files
        params_path = join(self.get_workspace_dir(), 'bm_params.mat')
        # print 'params_path=',params_path
        self.eng.ComputePowersAndClassifier(self.pipeline.subject_id, self.get_workspace_dir(), params_path)


class ComputePowersPSTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True): MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # saves powers and classifier matlab files
        params_path = join(self.get_workspace_dir(), 'bm_params.mat')
        # print 'params_path=',params_path
        self.eng.ComputePowersPS(self.pipeline.subject_id, self.get_workspace_dir())


class SaveEventsTask(MatlabRamTask):
    def __init__(self, mark_as_completed=True): MatlabRamTask.__init__(self, mark_as_completed)

    def run(self):
        # saves GroupPSL.mat and PS2Events.mat
        self.eng.SaveEvents(self.pipeline.subject_id, self.pipeline.experiment, self.get_workspace_dir())


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

        classifier_files = glob(self.get_path_to_file_in_workspace(classifier_output_location) + '/*.mat')
        try:
            classifier_output_file_name_full = classifier_files[
                0]  # picking first file, there shuld be just one file there!
        except IndexError:
            print 'Could not locate *.mat in ' + self.get_path_to_file_in_workspace(classifier_output_location)
            sys.exit()

        res = deserialize_single_object_from_matlab_format(classifier_output_file_name_full, 'res')

        serialize_objects_in_matlab_format(self.get_path_to_file_in_workspace('Weights.mat'), (res.Weights, 'Weights'))
        # save weights in matlab format
        print 'res.Weights=', res.Weights
        # print 'res.W0=',res.W0
