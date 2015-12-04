# command line example:
# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
import sys
from os.path import *
from setup_utils import parse_command_line, configure_python_paths

# -------------------------------processing command line
if len(sys.argv)>2:

    args = parse_command_line()

# R1051J had 3 sessions
# R1060M had 4 sessions
# R1061T had 4 sessions
# R1065J had 6 sessions
# R1092J_2 had 3 sessions

else: # emulate command line
    command_line_emulation_argument_list = ['--subject','R1061T',
                                            '--task','RAM_FR1',
                                            '--workspace-dir','~/scratch/FR1_reports',

                                            # '--mount-point','/Users/m/',
                                            '--mount-point','/Volumes/rhino_root',
                                            '--python-path','~/PTSA_GIT',
                                            '--python-path','~/RAM_UTILS_GIT'
                                            ]
    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line

import numpy as np
from RamPipeline import RamPipeline
from RamPipeline import RamTask

from EventPreparation import EventPreparation

from ComputeFR1Powers import ComputeFR1Powers

from TalPreparation import TalPreparation


from ComputeTTest import ComputeTTest

from CheckTTest import CheckTTest

from XValTTest import XValTTest

from XValPlots import XValPlots


from ComputeClassifier import ComputeClassifier

from CheckClassifier import CheckClassifier

#from ComposeSessionSummary import ComposeSessionSummary

#from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.norm_method = 'zscore'

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.6
        self.fr1_buf = 1.0

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 12)

        self.log_powers = True

        self.timewin_start = 0
        self.timewin_step = 5
        self.timewin_end = 85
        self.timewin_width = 25

        self.ttest_frange = (70.0, 200.0)

        self.penalty_type = 'l1'
        self.Cs = np.logspace(np.log10(1e-2), np.log10(1e4), 22)


params = Params()


class ReportPipeline(RamPipeline):
    def __init__(self, subject, output_dir, task, workspace_dir, mount_point=None):
        RamPipeline.__init__(self)
        self.subject = subject
        self.task = self.experiment = task
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)
        self.output_dir = output_dir



# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,output_dir=expanduser(args.workspace_dir),
                                       workspace_dir=join(args.workspace_dir,args.task+'_'+args.subject), mount_point=args.mount_point)

report_pipeline.add_task(EventPreparation(mark_as_completed=False))

report_pipeline.add_task(TalPreparation(mark_as_completed=False))

report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=True))

report_pipeline.add_task(CheckTTest(params=params, mark_as_completed=False))

report_pipeline.add_task(XValTTest(params=params, mark_as_completed=False))

report_pipeline.add_task(XValPlots(params=params, mark_as_completed=False))

#
# #report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))
#
# #report_pipeline.add_task(CheckClassifier(params=params, mark_as_completed=False))
#
# #report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))
#
# #report_pipeline.add_task(GeneratePlots(mark_as_completed=False))
#
# #report_pipeline.add_task(GenerateTex(mark_as_completed=False))
#
# #report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
