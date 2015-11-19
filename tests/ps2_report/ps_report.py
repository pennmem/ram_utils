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
                                            '--workspace-dir','/data10/scratch/mswat/R1086M_10',
                                            '--matlab-path','~/eeg',
                                            '--matlab-path','~/matlab/beh_toolbox',
                                            '--matlab-path','~/RAM/RAM_reporting',
                                            '--matlab-path','~/RAM/RAM_sys2Biomarkers',
                                            '--matlab-path','~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab',
                                            '--python-path','~/RAM_UTILS_GIT']
    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line

import RamPipeline
from RamPipeline import MatlabRamTask, RamTask
# from RamPipeline import *


from MatlabTasks import *
from GenerateReportTasks import *
from PSReportingTask import PSReportingTask

class PSReportPipeline(RamPipeline):
    def __init__(self, subject_id, experiment, workspace_dir, matlab_paths=[]):
        RamPipeline.__init__(self)
        self.subject_id = subject_id
        self.experiment = experiment
        self.set_workspace_dir(workspace_dir)
        self.matlab_paths = matlab_paths
        self.add_matlab_search_paths(matlab_paths)



# sets up processing pipeline
ps_report_pipeline = PSReportPipeline(subject_id=args.subject, experiment=args.experiment,
                                       workspace_dir=args.workspace_dir, matlab_paths=args.matlab_path)

# ps_report_pipeline = PS2ReportPipeline(subject_id='R1056M', experiment='PS1', workspace_dir='/scratch/busygin/py_run_8/', matlab_paths=['~/eeg','~/matlab/beh_toolbox','~/RAM/RAM_reporting','~/RAM/RAM_sys2Biomarkers','.'])

#  ----------------------------------- Matlab Tasks
# creates parameter .mat file - to match Youssef's code
ps_report_pipeline.add_task(CreateParamsTask())

# Computes FR1 Spectral Powers and classifier - calls Youssef's code
ps_report_pipeline.add_task(ComputePowersAndClassifierTask())

# Computes Spectral Powers for Parameter Search Stimulation sessions
ps_report_pipeline.add_task(ComputePowersPSTask())

# Saves Events Stimulation events for a given PS session
ps_report_pipeline.add_task(SaveEventsTask())

ps_report_pipeline.add_task(PrepareBPSTask())


# ------------------------------------ Report Generating tasks
ps_report_pipeline.add_task(ExtractWeightsTask(mark_as_completed=True))

# #  does actual analysis of the PS2 data - passes cumulative_plot_data_dict
ps_report_pipeline.add_task(PSReportingTask(mark_as_completed=False))
#
#  generates plots for the report
ps_report_pipeline.add_task(GeneratePlots(mark_as_completed=False))
#
#  generates tex for the reports
ps_report_pipeline.add_task(GenerateTex(mark_as_completed=False))
#
# # compiles generted tex to PDF
ps_report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))
#
# # starts processing pipeline
ps_report_pipeline.execute_pipeline()
