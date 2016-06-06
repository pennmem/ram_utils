import sys
from os.path import *

from ReportUtils import CMLParser,ReportPipeline


cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--subject','R1175N')
cml_parser.arg('--task','RAM_PAL3')
cml_parser.arg('--workspace-dir','/scratch/busygin/PAL3_reports_db')
cml_parser.arg('--mount-point','')
#cml_parser.arg('--recompute-on-no-status')

args = cml_parser.parse()


import numpy as np
from RamPipeline import RamPipeline
from RamPipeline import RamTask

from PAL1EventPreparation import PAL1EventPreparation

from EventPreparation import EventPreparation

from MathEventPreparation import MathEventPreparation

from ComputePAL1Powers import ComputePAL1Powers

from ComputeClassifier import ComputeClassifier

from ComputePALStimPowers import ComputePALStimPowers

from MontagePreparation import MontagePreparation

from ComputePALStimTable import ComputePALStimTable

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.pal1_start_time = 0.3
        self.pal1_end_time = 2.0
        self.pal1_buf = 1.0

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.ttest_frange = (70.0, 200.0)

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


params = Params()

#
# class ReportPipeline(RamPipeline):
#     def __init__(self, subject, task, workspace_dir, mount_point=None):
#         RamPipeline.__init__(self)
#         self.subject = subject
#         self.task = self.experiment = task
#         self.mount_point = mount_point
#         self.set_workspace_dir(workspace_dir)

# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,experiment=args.task,
                                 workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)




# sets up processing pipeline
# report_pipeline = ReportPipeline(subject=args.subject, task=args.task,
#                                        workspace_dir=join(args.workspace_dir,args.task+'_'+args.subject), mount_point=args.mount_point)
#

report_pipeline.add_task(PAL1EventPreparation(params=params, mark_as_completed=False))

report_pipeline.add_task(EventPreparation(mark_as_completed=False))

report_pipeline.add_task(MathEventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputePALStimPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputePALStimTable(params=params, mark_as_completed=True))

report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))
#
report_pipeline.add_task(GeneratePlots(mark_as_completed=False))
#
report_pipeline.add_task(GenerateTex(mark_as_completed=False))
#
report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
