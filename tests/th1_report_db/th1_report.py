# command line example:
# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT

import sys
from os.path import *

from ReportUtils import CMLParser,ReportPipeline


cml_parser = CMLParser(arg_count_threshold=1)
#cml_parser.arg('--subject','R1168T')
cml_parser.arg('--subject','R1180C')
cml_parser.arg('--task','RAM_TH1')
cml_parser.arg('--workspace-dir','/scratch/RAM_maint/automated_reports_json/TH1_reports_db')
cml_parser.arg('--mount-point','')
#cml_parser.arg('--recompute-on-no-status')
#cml_parser.arg('--exit-on-no-change')
args = cml_parser.parse()


# cml_parser = CMLParser(arg_count_threshold=1)
# cml_parser.arg('--subject','R1168T')
# cml_parser.arg('--task','RAM_TH1')
# # cml_parser.arg('--workspace-dir','/scratch/jfm2/python/TH1')
# cml_parser.arg('--mount-point','')
# cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')
# cml_parser.arg('--python-path','/home1/jfm2/python/ram_utils')
# cml_parser.arg('--python-path','/home1/jfm2/python/ptsa_new')
# cml_parser.arg('--python-path','/home1/jfm2/python/extra_libs')
# args = cml_parser.parse()

# ------------------------------- end of processing command line

# import numpy as np
# from RamPipeline import RamPipeline
# from RamPipeline import RamTask

from TH1EventPreparation import TH1EventPreparation

from ComputeTH1Powers import ComputeTH1Powers

from ComputeTH1ClassPowers import ComputeTH1ClassPowers

from MontagePreparation import MontagePreparation

from ComputeTTest import ComputeTTest

from ComputeClassifier import ComputeClassifier

from ComputeClassifier_conf import ComputeClassifier_conf

from ComputeClassifier_distThresh import ComputeClassifier_distThresh

from ComputeClassifier_withTranspose import ComputeClassifier_withTranspose

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.th1_start_time = -1.2
        self.th1_end_time = 0.5
        self.th1_buf = 1.7

        # self.th1_start_time = 0.0
        # self.th1_end_time = 1.5
        # self.th1_buf = 0.0

        self.filt_order = 4

        self.log_powers = True
        self.classifier_freqs = np.logspace(np.log10(1), np.log10(200), 8)
        self.freqs = np.logspace(np.log10(1), np.log10(200), 50)
        self.ttest_frange = np.array([[1.0,3.0],[3.0,9.0],[40.0,70.0],[70.0,200.0]])


        self.penalty_type = 'l2'
        self.C = 7.2e-4
        # self.C = 0.00215

        self.n_perm = 200
        self.doStratKFold = False
        self.n_folds = 8
        
        self.doConf_classification = True
        self.doDist_classification = False
        self.doClass_wTranspose    = False        

params = Params()


# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,
                                       workspace_dir=join(args.workspace_dir,args.task+'_'+args.subject), mount_point=args.mount_point,exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)

report_pipeline.add_task(TH1EventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeTH1Powers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeTH1ClassPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

if params.doConf_classification:
    report_pipeline.add_task(ComputeClassifier_conf(params=params, mark_as_completed=True))

if params.doDist_classification:
    report_pipeline.add_task(ComputeClassifier_distThresh(params=params, mark_as_completed=True))

if params.doClass_wTranspose:
    report_pipeline.add_task(ComputeClassifier_withTranspose(params=params, mark_as_completed=True))

report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))
# #
report_pipeline.add_task(GeneratePlots(params=params,mark_as_completed=False))
# #
report_pipeline.add_task(GenerateTex(params=params,mark_as_completed=False))
#
report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
