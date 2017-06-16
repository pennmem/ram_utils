import luigi
from RamTaskL import RamTaskL
from os.path import *

from Params import Params
from Pipeline import Pipeline

from FR1EventPreparation_1 import FR1EventPreparation_1
from RepetitionRatio_1 import RepetitionRatio_1
from MontagePreparation_1 import MontagePreparation_1
from ComputeFR1Powers_1 import ComputeFR1Powers_1
from ComputeClassifier_1 import ComputeClassifier_1
from ComputeClassifier_1 import ComputeJointClassifier_1

pipeline = Pipeline(Params())

class Sink(RamTaskL):
    subject = luigi.Parameter(default='')
    workspace_dir = luigi.Parameter(default=join(expanduser('~'), 'scratch'))
    pipeline = luigi.Parameter(default=None)

    def requires(self):
        # yield Setup(pipeline=self.pipeline, mark_as_completed=False, subject='R1065J', workspace_dir='d:\sc_lui')
        # yield FR1EventPreparation_1(pipeline=self.pipeline, mark_as_completed=True, subject=self.subject, workspace_dir=self.workspace_dir)
        self.pipeline = pipeline
        self.pipeline.subject = self.subject
        self.pipeline.workspace_dir = join(self.workspace_dir, self.pipeline.subject)

        # yield FR1EventPreparation_1(pipeline=self.pipeline, mark_as_completed=True)
        yield MontagePreparation_1(pipeline=self.pipeline, mark_as_completed=True)
        yield RepetitionRatio_1(pipeline=self.pipeline)
        yield ComputeFR1Powers_1(pipeline=self.pipeline,mark_as_completed=True)
        yield ComputeClassifier_1(pipeline=self.pipeline,mark_as_completed=True)
        yield ComputeJointClassifier_1(pipeline=self.pipeline,mark_as_completed=True)


    def define_outputs(self):
        self.add_file_resource('empty_file_sink')

    def run_impl(self):
        print 'Hello'


        self.clear_output_file('empty_file_sink')
