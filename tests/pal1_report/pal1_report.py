# command line example:
# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
import sys

from setup_utils import parse_command_line, configure_python_paths

# -------------------------------processing command line
if len(sys.argv)>2:

    args = parse_command_line()


else: # emulate command line
    # command_line_emulation_argument_list = ['--subject','R1149N',
    #                                         '--task','RAM_PAL1',
    #                                         '--workspace-dir','/scratch/busygin/PAL1_reports',
    #                                         '--mount-point','',
    #                                         '--python-path','/home1/busygin/ram_utils_new_ptsa',
    #                                         '--python-path','/home1/busygin/python/ptsa_latest'
    #                                         # '--exit-on-no-change'
    #                                         ]

    # command_line_emulation_argument_list = ['--subject','R1028M',
    #                                         '--task','RAM_PAL1',
    #                                         '--workspace-dir','/Users/busygin/scratch/PAL1_reports',
    #                                         '--mount-point','/Volumes/RHINO',
    #                                         '--python-path','/Users/busygin/ram_utils_new_ptsa',
    #                                         '--python-path','/Users/busygin/ptsa_latest',
    #                                         '--python-path','/Users/busygin/cpp/morlet_flex_install'
    #                                         ]

    command_line_emulation_argument_list = ['--subject','R1028M',
                                            '--task','RAM_PAL1',
                                            '--workspace-dir','/scratch/mswat/PAL1_reports',
                                            '--mount-point','',
                                            '--python-path','/home1/mswat/RAM_UTILS_GIT',
                                            '--python-path','/home1/mswat/PTSA_NEW_GIT'
                                            # '--exit-on-no-change'
                                            ]


    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line


from ReportUtils import ReportPipelineBase


from PAL1EventPreparation import PAL1EventPreparation

from MathEventPreparation import MathEventPreparation

from ComputePAL1Powers import ComputePAL1Powers

from TalPreparation import TalPreparation

from GetLocalization import GetLocalization

from ComputePAL1HFPowers import ComputePAL1HFPowers

from ComputeTTest import ComputeTTest

from ComputeClassifier import ComputeClassifier

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.pal1_start_time = 1.0
        self.pal1_end_time = 3.0
        self.pal1_buf = 1.0

        self.hfs_start_time = 1.0
        self.hfs_end_time = 3.0
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
    def __init__(self, subject, task, workspace_dir, mount_point=None, exit_on_no_change=False):
        super(ReportPipeline,self).__init__(subject=subject, workspace_dir=workspace_dir, mount_point=mount_point, exit_on_no_change=exit_on_no_change)
        self.task = task
        self.experiment = task




# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,
                                       workspace_dir=join(args.workspace_dir,args.task+'_'+args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change)

report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))

report_pipeline.add_task(MathEventPreparation(mark_as_completed=False))

report_pipeline.add_task(TalPreparation(mark_as_completed=False))

report_pipeline.add_task(GetLocalization(mark_as_completed=False))

report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputePAL1HFPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

report_pipeline.add_task(GenerateTex(mark_as_completed=False))

report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

report_pipeline.add_task(DeployReportPDF(mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
