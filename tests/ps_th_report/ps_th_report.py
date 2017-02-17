# command line example:

import sys
from os.path import *
from setup_utils import parse_command_line, configure_python_paths


from ReportUtils import CMLParser,ReportPipeline

cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--subject','R1157C')
cml_parser.arg('--workspace-dir','/scratch/leond/PS2_TH')
cml_parser.arg('--mount-point','')
#cml_parser.arg('--recompute-on-no-status')
cml_parser.arg('--task','PS2')

# cml_parser.arg('--exit-on-no-change')

args = cml_parser.parse()


from THEventPreparation import THEventPreparation
from ControlEventPreparation import ControlEventPreparation
from PSEventPreparation import PSEventPreparation

from ComputeTH1ClassPowers import ComputeTH1ClassPowers
from ComputeControlPowers import ComputeControlPowers
from ComputePSPowers import ComputePSPowers

from MontagePreparation import MontagePreparation

from ComputeClassifier import ComputeClassifier

from ComputeControlTable import ComputeControlTable
from ComputePSTable import ComputePSTable

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.th1_start_time = -1.2
        self.th1_end_time = 0.5
        self.th1_buf = 1.7

        self.sham1_start_time = 1.0
        self.sham1_end_time = 2.0
        self.sham_buf = 1.0

        self.sham2_start_time = 10.0 - 3.7
        self.sham2_end_time = 10.0 - 2.7

        self.ps_start_time = -1.0
        self.ps_end_time = 0.0
        self.ps_buf = 1.0
        self.ps_offset = 0.1

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(1.0), np.log10(200.0), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200
        self.doStratKFold = False
        self.n_folds = 8

        self.doConf_classification = True
        self.doDist_classification = False
        self.doClass_wTranspose    = False


params = Params()


# sets up processing pipeline

report_pipeline = ReportPipeline(subject=args.subject,
                                 experiment=args.task,
                                 sessions=args.session,
                                 workspace_dir=join(args.workspace_dir, args.subject),
                                 mount_point=args.mount_point,
                                 exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)

report_pipeline.add_task(THEventPreparation(mark_as_completed=False))

# report_pipeline.add_task(ControlEventPreparation(params=params, mark_as_completed=False))

report_pipeline.add_task(PSEventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(mark_as_completed=True))

report_pipeline.add_task(ComputeTH1ClassPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeControlPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputePSPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeControlTable(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputePSTable(params=params, mark_as_completed=True))

report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

report_pipeline.add_task(GenerateTex(mark_as_completed=False))

report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
