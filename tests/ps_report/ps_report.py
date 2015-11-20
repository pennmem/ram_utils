# command line example:
# python ps_report.py --subject=R1056M --experiment=PS2 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1056M --experiment=PS2 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1086M --experiment=PS2 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
import sys
from setup_utils import parse_command_line, configure_python_paths

# -------------------------------processing command line
if len(sys.argv)>2:

    args = parse_command_line()


else: # emulate command line
    command_line_emulation_argument_list = ['--subject','R1086M',
                                            '--experiment','PS2',
                                            '--workspace-dir','/Users/m/scratch/mswat/Python_pipeline',
                                            '--mount-point','/Volumes/rhino_root',
                                            '--python-path','~/RAM_UTILS_GIT',
                                            '--python-path','~/PTSA_INSTALL'
                                            ]
    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line

import numpy as np
from RamPipeline import RamPipeline
from RamPipeline import MatlabRamTask, RamTask
# from RamPipeline import *


# from MatlabTasks import *
# from GenerateReportTasks import *
# from PSReportingTask import PSReportingTask


from EventPreparation import EventPreparation
from PSEventPreparation import PSEventPreparation

from ComputeFR1Powers import ComputeFR1Powers

from ComputePowersPS import ComputePowersPS

from TalPreparation import TalPreparation

from ComputeClassifier import ComputeClassifier

from ComputeProbabilityDeltas import ComputeProbabilityDeltas

from ComposeSessionSummary import ComposeSessionSummary


# turn it into command line options

class Params(object):
    def __init__(self):
        self.fr1_start_time = -0.5
        self.fr1_end_time = 2.1
        self.fr1_buf = 1.0

        self.ps_pre_start_time = -1.0
        self.ps_pre_end_time = 0.0
        self.ps_pre_buf = 1.0

        self.ps_post_offset = 0.1

        # eeghz = 500
        # powhz = 50
        self.freqs = np.logspace(np.log10(3), np.log10(180), 12)


params = Params()


class ReportPipeline(RamPipeline):
    def __init__(self, subject_id, experiment, workspace_dir, mount_point=None):
        RamPipeline.__init__(self)
        self.subject_id = subject_id
        self.experiment = experiment
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)



# sets up processing pipeline
report_pipeline = ReportPipeline(subject_id=args.subject, experiment=args.experiment,
                                       workspace_dir=args.workspace_dir, mount_point=args.mount_point)

# ps_report_pipeline = PS2ReportPipeline(subject_id='R1056M', experiment='PS1', workspace_dir='/scratch/busygin/py_run_8/', matlab_paths=['~/eeg','~/matlab/beh_toolbox','~/RAM/RAM_reporting','~/RAM/RAM_sys2Biomarkers','.'])

#  ----------------------------------- Matlab Tasks
# creates parameter .mat file - to match Youssef's code
report_pipeline.add_task(EventPreparation(task='RAM_FR1',mark_as_completed=False))

report_pipeline.add_task(TalPreparation(task='RAM_FR1',mark_as_completed=False))

report_pipeline.add_task(ComputeFR1Powers(params = params, task='RAM_FR1',mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params = params, task='RAM_FR1',mark_as_completed=False))

report_pipeline.add_task(PSEventPreparation(task='RAM_FR1',mark_as_completed=False))

report_pipeline.add_task(ComputePowersPS(params = params, task='RAM_FR1',mark_as_completed=True))

report_pipeline.add_task(ComputeProbabilityDeltas(params = params, task='RAM_FR1',mark_as_completed=True))

report_pipeline.add_task(ComposeSessionSummary(params = params, task='RAM_FR1',mark_as_completed=True))



# # Computes FR1 Spectral Powers and classifier - calls Youssef's code
# ps_report_pipeline.add_task(ComputePowersAndClassifierTask())
#
# # Computes Spectral Powers for Parameter Search Stimulation sessions
# ps_report_pipeline.add_task(ComputePowersPSTask())
#
# # Saves Events Stimulation events for a given PS session
# ps_report_pipeline.add_task(SaveEventsTask())
#
# ps_report_pipeline.add_task(PrepareBPSTask())


# # ------------------------------------ Report Generating tasks
# ps_report_pipeline.add_task(ExtractWeightsTask(mark_as_completed=True))
#
# # #  does actual analysis of the PS2 data - passes cumulative_plot_data_dict
# ps_report_pipeline.add_task(PSReportingTask(mark_as_completed=False))
# #
# #  generates plots for the report
# ps_report_pipeline.add_task(GeneratePlots(mark_as_completed=False))
# #
# #  generates tex for the reports
# ps_report_pipeline.add_task(GenerateTex(mark_as_completed=False))
# #
# # # compiles generted tex to PDF
# ps_report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))
#
# # starts processing pipeline
report_pipeline.execute_pipeline()
