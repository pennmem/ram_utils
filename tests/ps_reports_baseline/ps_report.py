# command line example:

import sys
from setup_utils import parse_command_line, configure_python_paths
from os.path import join

# -------------------------------processing command line
if len(sys.argv)>2:

    args = parse_command_line()


else: # emulate command line
    command_line_emulation_argument_list = ['--subject','R1062J',
                                            '--experiment','PS1',
                                            '--workspace-dir','/scratch/busygin/PS1',
                                            '--mount-point','',
                                            '--python-path','/home1/busygin/ram_utils_new_ptsa',
                                            '--python-path','/home1/busygin/python/ptsa_new'
                                            ]
    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line


import numpy as np
from RamPipeline import RamPipeline

from FR1EventPreparation import FR1EventPreparation
from ControlEventPreparation import ControlEventPreparation
from PSEventPreparation import PSEventPreparation

from ComputeFR1Powers import ComputeFR1Powers
from ComputeControlPowers import ComputeControlPowers
from ComputePSPowers import ComputePSPowers

from TalPreparation import TalPreparation

from ComputeClassifier import ComputeClassifier

from ComputeControlTable import ComputeControlTable
from ComputePSTable import ComputePSTable

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.control_start_time = -1.0
        self.control_end_time = 0.0
        self.control_buf = 1.0

        self.ps_start_time = -1.0
        self.ps_end_time = 0.0
        self.ps_buf = 1.0
        self.ps_offset = 0.1

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


params = Params()


class ReportPipeline(RamPipeline):
    def __init__(self, subject, experiment, workspace_dir, mount_point=None):
        RamPipeline.__init__(self)
        self.subject = subject
        self.task = 'RAM_FR1'
        self.experiment = experiment
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)



# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, experiment=args.experiment,
                                       workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point)

report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))

report_pipeline.add_task(ControlEventPreparation(mark_as_completed=False))

report_pipeline.add_task(PSEventPreparation(mark_as_completed=False))

report_pipeline.add_task(TalPreparation(mark_as_completed=False))

report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))

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
