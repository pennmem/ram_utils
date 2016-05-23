import sys
from glob import glob
import re

from ReportUtils import CMLParser,ReportPipeline


cml_parser = CMLParser(arg_count_threshold=1)
# cml_parser.arg('--task','RAM_FR1')
# cml_parser.arg('--workspace-dir','/scratch/mswat/automated_reports/FR1_reports')
# cml_parser.arg('--mount-point','')
# cml_parser.arg('--recompute-on-no-status')
# # cml_parser.arg('--exit-on-no-change')

cml_parser.arg('--subject','R1065J')
cml_parser.arg('--task','RAM_FR1')
cml_parser.arg('--workspace-dir','/Users/m/scratch/automated_reports/FR1_reports')
cml_parser.arg('--mount-point','/Users/m')
cml_parser.arg('--recompute-on-no-status')
cml_parser.arg('--python-path','/Users/m/PTSA_NEW_GIT')
cml_parser.arg('--python-path','/Users/m/RAM_UTILS_GIT')
cml_parser.arg('--python-path','/Users/m/src/morlet_git_clion_install')
# cml_parser.arg('--exit-on-no-change')


args = cml_parser.parse()



from ReportUtils import ReportSummaryInventory, ReportSummary
from ReportUtils import ReportPipelineBase


from FR1EventPreparation import FR1EventPreparation

from MathEventPreparation import MathEventPreparation

from ComputeFR1Powers import ComputeFR1Powers

from TalPreparation import TalPreparation

from GetLocalization import GetLocalization

from ComputeFR1HFPowers import ComputeFR1HFPowers

from ComputeTTest import ComputeTTest

from ComputeClassifier import ComputeClassifier

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.hfs_start_time = 0.0
        self.hfs_end_time = 1.6
        self.hfs_buf = 1.0

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)
        self.hfs = np.logspace(np.log10(2), np.log10(200), 50)
        self.hfs = self.hfs[self.hfs>=70.0]

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


params = Params()

task = args.task
subject = args.subject


print '--Generating', task, 'report for', subject


report_pipeline = ReportPipeline(
                                 args=args,
                                 subject=subject,
                                 workspace_dir=join(args.workspace_dir, task + '_' + subject)
                                 )

report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))

report_pipeline.add_task(MathEventPreparation(mark_as_completed=False))

report_pipeline.add_task(TalPreparation(mark_as_completed=False))

report_pipeline.add_task(GetLocalization(mark_as_completed=False))

report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeFR1HFPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

report_pipeline.add_task(GenerateTex(mark_as_completed=False))

report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

report_pipeline.add_task(DeployReportPDF(mark_as_completed=False))

report_pipeline.execute_pipeline()




