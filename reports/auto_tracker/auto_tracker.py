# command line example:

import sys


from setup_utils import parse_command_line, configure_python_paths
from os.path import join

# -------------------------------processing command line
if len(sys.argv)>2:

    args = parse_command_line()


else: # emulate command line
    command_line_emulation_argument_list = ['--subject','R1075J',
                                         '--experiment','PS2',
                                         '--workspace-dir','/Users/m/scratch/auto_tracker',
                                         '--mount-point','/Users/m',
                                         '--python-path','/Users/m/RAM_UTILS_GIT',
                                         '--python-path','/Users/m/PTSA_NEW_GIT']

    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line


import numpy as np

from ReportUtils import ReportPipeline
# from RamPipeline import RamPipeline
#
# from RamPipeline.DependencyChangeTrackerLegacy import DependencyChangeTrackerLegacy

from FREventPreparation import FREventPreparation
from JSONStubPreparation import JSONStubPreparation


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.control_start_time = -1.1
        self.control_end_time = -0.1
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

        self.include_fr1 = True
        self.include_catfr1 = True


params = Params()


# class ReportPipeline(RamPipeline):
#     def __init__(self, subject, experiment, workspace_dir, mount_point=None):
#         RamPipeline.__init__(self)
#         self.subject = subject
#         #self.task = 'RAM_FR1'
#         self.experiment = experiment
#         self.mount_point = mount_point
#         self.set_workspace_dir(workspace_dir)
#         dependency_tracker = DependencyChangeTrackerLegacy(subject=subject, workspace_dir=workspace_dir, mount_point=mount_point)
#
#         self.set_dependency_tracker(dependency_tracker=dependency_tracker)



# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, experiment=args.experiment,
                                       workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point)




report_pipeline.add_task(JSONStubPreparation(params=params, mark_as_completed=True))
report_pipeline.add_task(FREventPreparation(params=params, mark_as_completed=True))

# starts processing pipeline
report_pipeline.execute_pipeline()
