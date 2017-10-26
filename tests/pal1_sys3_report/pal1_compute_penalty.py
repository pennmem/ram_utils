import re
import sys
from glob import glob
from os.path import *

from setup_utils import parse_command_line, configure_python_paths

# -------------------------------processing command line
if len(sys.argv)>2:

    args = parse_command_line()


else: # emulate command line
    command_line_emulation_argument_list = ['--subject','R1086M',
                                            '--task','RAM_PAL1',
                                            '--workspace-dir','/scratch/busygin/PAL1_penalty',
                                            '--mount-point','',
                                            '--python-path','/home1/busygin/ram_utils',
                                            '--python-path','/home1/busygin/python/ptsa_latest',
                                            '--python-path','/home1/mswat/extra_libs'
                                            ]
    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line

import numpy as np
from ReportUtils import ReportPipelineBase

from PAL1EventPreparation import PAL1EventPreparation

from ComputePAL1Powers import ComputePAL1Powers

from TalPreparation import TalPreparation

from ComputeAUCs import ComputeAUCs


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.pal1_start_time = 0.5
        self.pal1_end_time = 2.5
        self.pal1_buf = 1.0

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.ttest_frange = (70.0, 200.0)

        self.penalty_type = 'l2'
        self.Cs = np.logspace(np.log10(1e-6), np.log10(1e4), 22)


params = Params()


class ReportPipeline(ReportPipelineBase):
    def __init__(self, subject, task, workspace_dir, mount_point=None, exit_on_no_change=False):
        super(ReportPipeline,self).__init__(subject=subject, workspace_dir=workspace_dir, mount_point=mount_point, exit_on_no_change=exit_on_no_change)
        self.task = task
        self.experiment = task


task = 'RAM_PAL1'


def find_subjects_by_task(task):
    ev_files = glob(args.mount_point + ('/data/events/%s/R*_events.mat' % task))
    return [re.search(r'R1\d\d\d[A-Z](_\d+)?', f).group() for f in ev_files]


subjects = find_subjects_by_task(task)
subjects.sort()

for subject in subjects:
    print '--Generating', task, 'report for', subject

    # sets up processing pipeline
    report_pipeline = ReportPipeline(subject=subject, task=task,
                                           workspace_dir=join(args.workspace_dir,task+'_'+subject), mount_point=args.mount_point)

    report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))

    report_pipeline.add_task(TalPreparation(mark_as_completed=False))

    report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeAUCs(params=params, mark_as_completed=False))

    # starts processing pipeline

    report_pipeline.execute_pipeline()
