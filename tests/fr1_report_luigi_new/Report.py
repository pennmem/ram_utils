import luigi
from RamTaskL import RamTaskL
from os.path import *
from Params import Params
from Pipeline import Pipeline
from RepetitionRatio import RepetitionRatio
from MontagePreparation import MontagePreparation
from ComputeFR1Powers import ComputeFR1Powers
from ComputeFR1HFPowers import ComputeFR1HFPowers
from ComputeClassifier import ComputeClassifier
from ComputeClassifier import ComputeJointClassifier
from ComputeTTest import ComputeTTest
from ComposeSessionSummary import ComposeSessionSummary
from GenerateReportTasks import GeneratePlots
from GenerateReportTasks import GenerateTex

pipeline = Pipeline(Params())

"""
To run from command line (on Windows):

    cd <YOUR_RAM_UTILS_GIT_REPOSITORY>

    python -m luigi --module tests.fr1_report_luigi_new.Report Report --local-scheduler --subject=R1065J --workspace-dir=d:\sc_lui

"""


class Report(RamTaskL):
    subject = luigi.Parameter(default='')
    workspace_dir = luigi.Parameter(default=join(expanduser('~'), 'scratch'))
    pipeline = luigi.Parameter(default=None)

    def requires(self):
        self.pipeline = pipeline
        self.pipeline.subject = self.subject
        self.pipeline.workspace_dir = join(self.workspace_dir, self.pipeline.subject)

        yield MontagePreparation(pipeline=self.pipeline, mark_as_completed=True)
        yield RepetitionRatio(pipeline=self.pipeline)
        yield ComputeFR1Powers(pipeline=self.pipeline, mark_as_completed=True)
        yield ComputeClassifier(pipeline=self.pipeline, mark_as_completed=True)
        yield ComputeJointClassifier(pipeline=self.pipeline, mark_as_completed=True)
        yield ComputeFR1HFPowers(pipeline=self.pipeline, mark_as_completed=True)
        yield ComputeTTest(pipeline=self.pipeline, mark_as_completed=False)
        yield ComposeSessionSummary(pipeline=self.pipeline, mark_as_completed=False)
        yield GeneratePlots(pipeline=pipeline, mark_as_completed=False)
        yield GenerateTex(pipeline=pipeline, mark_as_completed=False)
