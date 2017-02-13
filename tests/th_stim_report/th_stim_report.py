import sys
from os.path import *
sys.path.append(join(dirname(__file__),'..','..'))

from ReportUtils import CMLParser,ReportPipeline


cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--subject','R1201P_1')
cml_parser.arg('--task','TH3')
cml_parser.arg('--workspace-dir','/scratch/leond/TH3_reports')
# cml_parser.arg('--workspace-dir','/scratch/RAM_maint/automated_reports_json/TH3_reports')
cml_parser.arg('--mount-point','')
#cml_parser.arg('--recompute-on-no-status')

args = cml_parser.parse()


import numpy as np
from RamPipeline import RamPipeline
from RamPipeline import RamTask

from THEventPreparation import THEventPreparation

from EventPreparation import EventPreparation

from ComputeTHPowers import ComputeTHPowers

from ComputeClassifier import ComputeClassifier

from ComputeTHStimPowers import ComputeTHStimPowers

from MontagePreparation import MontagePreparation

from ComputeTHStimTable import ComputeTHStimTable

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.th1_start_time = -1.2
        self.th1_end_time = 0.5
        self.th1_buf = 1.7

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(1), np.log10(200), 8)

        self.log_powers = True

        self.ttest_frange = (70.0, 200.0)

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200

        self.include_th1 = True


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

report_pipeline.add_task(THEventPreparation(params=params, mark_as_completed=False))

report_pipeline.add_task(EventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeTHPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeTHStimPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeTHStimTable(params=params, mark_as_completed=True))

report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

report_pipeline.add_task(GenerateTex(mark_as_completed=False))

report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
