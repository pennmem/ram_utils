import sys
from glob import glob
import re

from setup_utils import parse_command_line, configure_python_paths

# -------------------------------processing command line
if len(sys.argv)>2:

    args = parse_command_line()


else: # emulate command line
    # command_line_emulation_argument_list = ['--subject','R1086M',
    #                                         '--workspace-dir','/scratch/busygin/FR1_joint_reports',
    #                                         '--mount-point','',
    #                                         '--python-path','/home1/busygin/ram_utils_new_ptsa',
    #                                         '--python-path','/home1/busygin/python/ptsa_latest',
    #                                         #'--exit-on-no-change'
    #                                         ]

    command_line_emulation_argument_list = ['--subject','R1086M',
                                            '--workspace-dir','/scratch/mswat/FR1_check_1',
                                            '--mount-point','',
                                            '--python-path','/home1/mswat/RAM_UTILS_GIT',
                                            '--python-path','/home1/mswat/PTSA_NEW_GIT'
                                            #'--exit-on-no-change'
                                            ]

    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line

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


class ReportPipeline(ReportPipelineBase):
    def __init__(self, subject, workspace_dir, mount_point=None, exit_on_no_change=False):
        super(ReportPipeline,self).__init__(subject=subject, workspace_dir=workspace_dir, mount_point=mount_point, exit_on_no_change=exit_on_no_change)
        self.task = self.experiment = 'RAM_FR1_CatFR1_joint'





def find_subjects_by_task(task):
    ev_files = glob(args.mount_point + ('/data/events/%s/R*_events.mat' % task))
    return [re.search(r'R\d\d\d\d[A-Z](_\d+)?', f).group() for f in ev_files]


subjects = list(set(find_subjects_by_task('RAM_FR1')).intersection(find_subjects_by_task('RAM_CatFR1')))
subjects.sort()


rsi = ReportSummaryInventory(label='RAM_FR1_CatFR1_joint')

for subject in subjects:
    print '--Generating FR1&CatFR1 joint report for', subject

    # sets up processing pipeline
    report_pipeline = ReportPipeline(subject=subject,
                                           workspace_dir=join(args.workspace_dir,subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change)

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

    # starts processing pipeline
    report_pipeline.execute_pipeline()



print 'this is summary for all reports report ', rsi.compose_summary(detail_level=1)

rsi.send_email_digest(detail_level_list=[0,1])
