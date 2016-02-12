import sys
from os.path import *
from setup_utils import parse_command_line, configure_python_paths

# -------------------------------processing command line
if len(sys.argv)>2:

    args = parse_command_line()


else: # emulate command line
    # command_line_emulation_argument_list = ['--subject','R1124J_1',
    #                                         '--task','RAM_FR3',
    #                                         '--workspace-dir','/scratch/busygin/FR3_reports',
    #                                         '--mount-point','',
    #                                         '--python-path','/home1/busygin/ram_utils',
    #                                         '--python-path','/home1/busygin/python/ptsa/build/lib.linux-x86_64-2.7'
    #                                         ]

    command_line_emulation_argument_list = ['--subject','R1124J_1',
                                            '--task','RAM_FR3',
                                            '--workspace-dir','/Users/busygin/scratch/FR3_reports',
                                            '--mount-point','/Volumes/RHINO',
                                            '--python-path','/Users/busygin/ram_utils_new_ptsa',
                                            '--python-path','/Users/busygin/ptsa_latest'
                                            ]

    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line

import numpy as np
from RamPipeline import RamPipeline
from RamPipeline import RamTask

from EventPreparation import EventPreparation

from MathEventPreparation import MathEventPreparation

#from ComputeFR1Powers import ComputeFR1Powers

from TalPreparation import TalPreparation

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.6
        self.fr1_buf = 1.0

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.ttest_frange = (70.0, 200.0)

        self.penalty_type = 'l2'
        self.C = 7.2e-4


params = Params()


class ReportPipeline(RamPipeline):
    def __init__(self, subject, task, workspace_dir, mount_point=None):
        RamPipeline.__init__(self)
        self.subject = subject
        self.task = self.experiment = task
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)



# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,
                                       workspace_dir=join(args.workspace_dir,args.task+'_'+args.subject), mount_point=args.mount_point)

report_pipeline.add_task(EventPreparation(mark_as_completed=False))

report_pipeline.add_task(MathEventPreparation(mark_as_completed=False))

report_pipeline.add_task(TalPreparation(mark_as_completed=False))

#report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))
#
report_pipeline.add_task(GeneratePlots(mark_as_completed=False))
#
report_pipeline.add_task(GenerateTex(mark_as_completed=False))
#
report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
