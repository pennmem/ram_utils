import luigi
import numpy as np
import os
import os.path
import numpy as np
from sklearn.externals import joblib

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import hashlib
from ReportTasks.RamTaskMethods import create_baseline_events

# # command line example:
# # python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT
#
# # python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT
#
# # python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
# import sys
# from os.path import *
#
#
# from ReportUtils import CMLParser,ReportPipeline
#
# cml_parser = CMLParser(arg_count_threshold=1)
# cml_parser.arg('--subject','R1304N')
# cml_parser.arg('--task','FR1')
# cml_parser.arg('--workspace-dir','scratch/leond/FR1_reports')
# cml_parser.arg('--mount-point','/Volumes/rhino_root')
# #cml_parser.arg('--recompute-on-no-status')
# # cml_parser.arg('--exit-on-no-change')
#
# args = cml_parser.parse()
#
#
# from FR1EventPreparation import FR1EventPreparation
#
# from RepetitionRatio import RepetitionRatio
#
# from ComputeFR1Powers import ComputeFR1Powers
#
# from MontagePreparation import MontagePreparation
#
# from ComputeFR1HFPowers import ComputeFR1HFPowers
#
# from ComputeTTest import ComputeTTest
#
# from ComputeClassifier import ComputeClassifier,ComputeJointClassifier
#
# from ComposeSessionSummary import ComposeSessionSummary
#
# from GenerateReportTasks import *
#
#
# # turn it into command line options
#

from Params import Params
from Pipeline import Pipeline
from RamTaskL import RamTaskL
from FR1EventPreparation import FR1EventPreparation
from RepetitionRatio import RepetitionRatio
from MontagePreparation import MontagePreparation
from ComputeFR1Powers import ComputeFR1Powers
from ComputeFR1HFPowers import ComputeFR1HFPowers
from ComputeTTest import ComputeTTest
from ComputeClassifier import ComputeClassifier
from ComputeClassifier import ComputeJointClassifier
from ComposeSessionSummary import ComposeSessionSummary
from GenerateReportTasks import GeneratePlots
from GenerateReportTasks import GenerateTex

params = Params()
pipeline = Pipeline(params)




class EventCheck(RamTaskL):


    def requires(self):
        # yield FR1EventPreparation(pipeline=self.pipeline, mark_as_completed=True)
        # return FR1EventPreparation(pipeline=self.pipeline, mark_as_completed=True)
        yield FR1EventPreparation(pipeline=self.pipeline )





    def run_impl(self):
        print 'GOT HERE'

        # with self.input()[0]['event_files'].open('r') as f:
        #     print 'THOSE ARE READ FILES ', f.read()

        events = self.get_passed_object(self.pipeline.task + '_all_events')
        print events

        #
        # pass


if __name__ == '__main__':
    # try:
        luigi.build([FR1EventPreparation(pipeline=pipeline, mark_as_completed=True),
                 EventCheck(pipeline=pipeline),
                 MontagePreparation(pipeline=pipeline, mark_as_completed=True),
                 RepetitionRatio(pipeline=pipeline),
                 ComputeFR1Powers(pipeline=pipeline, mark_as_completed=True),
                 ComputeFR1HFPowers(pipeline=pipeline, mark_as_completed=True),
                 ComputeTTest(pipeline=pipeline,mark_as_completed=False),
                 ComputeClassifier(pipeline=pipeline,mark_as_completed=True),
                 ComputeJointClassifier(pipeline=pipeline,mark_as_completed=True),
                 ComposeSessionSummary(pipeline=pipeline,mark_as_completed=False),
                 GeneratePlots(pipeline=pipeline,mark_as_completed=False),
                 GenerateTex(pipeline=pipeline,mark_as_completed=False)
                 ],
                local_scheduler=True)
    # except RuntimeError as e:
    #     raise e


    # luigi.build([pipeline(params=params,subject='R1065J',task='FR1'), FR1EventPreparation()], local_scheduler=True)

    # luigi.build([FR1EventPreparation(pipeline=pipeline, mark_as_completed=True), EventCheck(pipeline=pipeline)],
    #             local_scheduler=True)
    # luigi.build([EventCheck(pipeline=pipeline)], local_scheduler=True)

#
#
# params = Params()
#
#
#
# # sets up processing pipeline
# report_pipeline = ReportPipeline(subject=args.subject, task=args.task,experiment=args.task,sessions =args.sessions,
#                                  workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
#                                  recompute_on_no_status=args.recompute_on_no_status)
#
#
# report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))
#
# report_pipeline.add_task(MontagePreparation(params, mark_as_completed=False))
#
# if 'cat' in args.task:
#     report_pipeline.add_task(RepetitionRatio(mark_as_completed=True))
#
# report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))
#
# report_pipeline.add_task(ComputeFR1HFPowers(params=params, mark_as_completed=True))
#
# report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=True))
#
# report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=False))
#
# report_pipeline.add_task(ComputeJointClassifier(params=params,mark_as_completed=False))
#
# report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))
#
# report_pipeline.add_task(GeneratePlots(mark_as_completed=False))
#
# report_pipeline.add_task(GenerateTex(mark_as_completed=False))
#
# report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))
#
#
# # starts processing pipeline
# report_pipeline.execute_pipeline()
