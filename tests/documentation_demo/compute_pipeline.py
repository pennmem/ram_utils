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
    command_line_emulation_argument_list = ['--subject','R1060M',
                                            '--task','RAM_FR1',
                                            '--workspace-dir','~/scratch/FR1_DEBUG',

                                            '--mount-point','/Users/m/',
                                            # '--mount-point','/Volumes/rhino_root',
                                            '--python-path','~/PTSA_GIT',
                                            '--python-path','~/RAM_UTILS_GIT'
                                            ]
    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line

import numpy as np
from RamPipeline import RamPipeline
from EventPreparation import EventPreparation
from EEGRawPreparation import EEGRawPreparation
from PlotTask import PlotTask


class ComputePipeline(RamPipeline):
    def __init__(self, workspace_dir):
        RamPipeline.__init__(self)
        self.set_workspace_dir(workspace_dir)

# sets up processing pipeline
compute_pipeline = ComputePipeline(workspace_dir='~/scratch/documentation_demo')

# compute_pipeline.add_task(EventPreparation(mark_as_completed=False))
#
# compute_pipeline.add_task(EEGRawPreparation(mark_as_completed=False))

compute_pipeline.add_task(PlotTask(mark_as_completed=False))

# starts processing pipeline
compute_pipeline.execute_pipeline()
