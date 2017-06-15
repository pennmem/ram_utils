DEBUG = False

from os.path import *

from ReportUtils import CMLParser,ReportPipeline
import sys
import time


parser = CMLParser()
parser.parser.add_argument('--classsifier')

args_obj=parser.parse()

from PAL1EventPreparation import PAL1EventPreparation
from PAL5EventPreparation import PAL5EventPreparation



from FREventPreparation import FREventPreparation

from CombinedEventPreparation import CombinedEventPreparation

from ComputePAL5Powers import ComputePAL5Powers

from ComputePowers import ComputePowers

from MontagePreparation import MontagePreparation

from ComputeClassifier import ComputeClassifier
from ComputeClassifier import ComputePAL1Classifier

from ComputePALStimTable import ComputePALStimTable
from ComposeSessionSummary import ComposeSessionSummary
from GenerateReportTasks import GenerateReportPDF,GeneratePlots,GenerateTex
import numpy as np


class StimParams(object):
    def __init__(self, **kwds):
        pass


class Params(object):
    def __init__(self):
        self.version = '5.00'

        self.include_fr1 = True
        self.include_catfr1 = True
        self.include_fr3 = True
        self.include_catfr3 = True

        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.fr1_retrieval_start_time = -0.525
        self.fr1_retrieval_end_time = 0.0
        self.fr1_retrieval_buf = 0.524


        self.pal1_start_time = 0.3
        self.pal1_end_time = 2.00
        self.pal1_buf = 1.2

        # original code
        self.pal1_retrieval_start_time = -0.625
        self.pal1_retrieval_end_time = -0.1
        self.pal1_retrieval_buf = 0.524

        # self.retrieval_samples_weight = 2.5
        # self.encoding_samples_weight = 2.5
        self.encoding_samples_weight = 7.2
        self.pal_samples_weight = 1.93

        self.recall_period = 5.0

        self.sliding_window_interval = 0.1
        self.sliding_window_start_offset = 0.3

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(6), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4


        self.n_perm = 200


        self.stim_params = StimParams(
        )


params = Params()




if __name__ == '__main__':

        # setting workspace

        report_pipeline = ReportPipeline(subject=args_obj.subject, task = 'PAL5',
                                         workspace_dir=join(args_obj.workspace_dir, args_obj.subject),
                                         mount_point=args_obj.mount_point,
                                         args=args_obj)

        report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

        report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))

        report_pipeline.add_task(FREventPreparation(mark_as_completed=False))

        report_pipeline.add_task(PAL5EventPreparation(mark_as_completed=False))

        report_pipeline.add_task(CombinedEventPreparation(mark_as_completed=False))

        report_pipeline.add_task(ComputePowers(params=params, mark_as_completed=(True & args_obj.allow_fast_rerun)))

        report_pipeline.add_task(ComputePAL5Powers(params=params,mark_as_completed=True))

        if args_obj.classifier == 'combined':
            report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))
        else:
            report_pipeline.add_task(ComputePAL1Classifier(params=params, mark_as_completed=True))

        report_pipeline.add_task(ComputePALStimTable(mark_as_completed=False,params=params))

        report_pipeline.add_task(ComposeSessionSummary(mark_as_completed=False,params=params))

        report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

        report_pipeline.add_task(GenerateTex(mark_as_completed=False))

        report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

        # starts processing pipeline
        report_pipeline.execute_pipeline()



