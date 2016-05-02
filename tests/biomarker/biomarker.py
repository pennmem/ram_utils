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


else: # emulate command line
    command_line_emulation_argument_list = ['--subject','R1166D',
                                            '--task','RAM_FR1',
                                            '--workspace-dir','/scratch/busygin/biomarkers',
                                            '--mount-point','',
                                            '--python-path','/home1/busygin/ram_utils',
                                            '--python-path','/home1/busygin/python/ptsa_latest'
                                            ]
    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line

from RamPipeline import RamPipeline

from FREventPreparation import FREventPreparation

from ComputeFRPowers import ComputeFRPowers

from TalPreparation import TalPreparation

from ComputeClassifier import ComputeClassifier

from SaveMatlabFile import SaveMatlabFile

import numpy as np


class StimParams(object):
    def __init__(self):
        self.n_channels = 128
        self.elec1 = 3
        self.elec2 = 4
        self.amplitude = 1000
        self.duration = 300
        self.trainFrequency = 1
        self.trainCount = 1
        self.pulseFrequency = 200
        self.pulseCount = 100

# turn it into command line options

class Params(object):
    def __init__(self):
        self.version = '2.00'

        self.include_fr1 = True
        self.include_catfr1 = True
        self.include_fr3 = True
        self.include_catfr3 = True

        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200

        self.stim_params = StimParams()


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

report_pipeline.add_task(FREventPreparation(params=params, mark_as_completed=False))

report_pipeline.add_task(TalPreparation(mark_as_completed=False))

report_pipeline.add_task(ComputeFRPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(SaveMatlabFile(params=params, mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
