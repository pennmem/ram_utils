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
    # command_line_emulation_argument_list = ['--subject','R1127P_2',
    #                                         '--task','RAM_CatFR1',
    #                                         '--workspace-dir','/scratch/busygin/CatFR1_reports_new_new',
    #                                         '--mount-point','',
    #                                         '--python-path','/home1/busygin/ram_utils_new_ptsa',
    #                                         '--python-path','/home1/busygin/python/ptsa_latest',
    #                                         # '--exit-on-no-change'
    #                                         ]

    # command_line_emulation_argument_list = ['--subject','R1127P_2',
    #                                         '--task','RAM_CatFR1',
    #                                         '--workspace-dir','/Users/m/scratch/CatFR1_reports_ms',
    #                                         '--mount-point','/Volumes/rhino_root',
    #                                         '--python-path','/Users/m/RAM_UTILS_GIT',
    #                                         '--python-path','/Users/m/PTSA_NEW_GIT',
    #                                         # '--exit-on-no-change'
    #                                         ]

    command_line_emulation_argument_list = ['--subject','R1060M',
                                            '--task','RAM_FR1',
                                            '--workspace-dir','/Users/m/scratch/FR1_reports_ms',
                                            '--mount-point','/Volumes/rhino_root',
                                            '--python-path','/Users/m/RAM_UTILS_GIT',
                                            '--python-path','/Users/m/PTSA_NEW_GIT',
                                            # '--exit-on-no-change'
                                            ]


    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line

from ReportUtils.DependencyChangeTrackerLegacy import DependencyChangeTrackerLegacy

from FR1EventPreparation import FR1EventPreparation

from MathEventPreparation import MathEventPreparation

from ComputeFR1Powers import ComputeFR1Powers

from TalPreparation import TalPreparation

from GetLocalization import GetLocalization

from ComputeTTest import ComputeTTest

#from CheckTTest import CheckTTest

#from XValTTest import XValTTest

#from XValPlots import XValPlots

from ComputeClassifier import ComputeClassifier

#from CheckClassifier import CheckClassifier

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.ttest_frange = (70.0, 200.0)

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


params = Params()

# from ReportUtils import ReportPipeline
class ReportPipeline(RamPipeline):
    def __init__(self, subject, task, workspace_dir, mount_point=None, exit_on_no_change=False):
        RamPipeline.__init__(self)
        self.exit_on_no_change = exit_on_no_change
        self.subject = subject
        self.task = self.experiment = task
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)
        dependency_tracker = DependencyChangeTrackerLegacy(subject=subject, workspace_dir=workspace_dir, mount_point=mount_point)

        self.set_dependency_tracker(dependency_tracker=dependency_tracker)



# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,
                                       workspace_dir=join(args.workspace_dir,args.task+'_'+args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change)

report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))

report_pipeline.add_task(MathEventPreparation(mark_as_completed=False))

report_pipeline.add_task(TalPreparation(mark_as_completed=False))

report_pipeline.add_task(GetLocalization(mark_as_completed=False))

report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False))

#report_pipeline.add_task(CheckTTest(params=params, mark_as_completed=False))

#report_pipeline.add_task(XValTTest(params=params, mark_as_completed=False))

#report_pipeline.add_task(XValPlots(params=params, mark_as_completed=False))

#
report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))
#
# #report_pipeline.add_task(CheckClassifier(params=params, mark_as_completed=False))
#
report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))
#
report_pipeline.add_task(GeneratePlots(mark_as_completed=False))
#
report_pipeline.add_task(GenerateTex(mark_as_completed=False))
#
report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
