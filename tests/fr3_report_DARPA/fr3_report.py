import sys
from os.path import *
from setup_utils import parse_command_line, configure_python_paths

# -------------------------------processing command line
if len(sys.argv)>2:

    args = parse_command_line()


else: # emulate command line

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

from TalPreparation import TalPreparation

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


class ReportPipeline(RamPipeline):
    def __init__(self, subject, task, workspace_dir, mount_point=None):
        RamPipeline.__init__(self)
        self.subject = subject
        self.task = self.experiment = task
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)


# sets up processing pipeline - the entire computation is divided into separate tasks that are manages
# by the Pipeline object
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,
                                       workspace_dir=join(args.workspace_dir,args.task+'_'+args.subject), mount_point=args.mount_point)

# EventPreparation task reads experiment summary file (so called events file) containing information about
# events that took place during the course of experiment. This file focuses on the word presentation events
report_pipeline.add_task(EventPreparation(mark_as_completed=False))

# MathEventPreparation task reads experiment summary file (so called events file) containing information
# about events that took place during the course of experiment. This file focuses on the math distractor events
report_pipeline.add_task(MathEventPreparation(mark_as_completed=False))

# TalPreparation task reads information about electrodes (monopolar and bipolar) localizations
report_pipeline.add_task(TalPreparation(mark_as_completed=False))

# ComposeSessionSummary is the core of this computational pipeline. It begins by reading FR3 experiment EEG data for a given patient.
# The data being read is split into short segments corresponding to single experimental events. So for a word presentation events
# the data will include 1.366 second worth of EEG data starting from the word onset and will be padded with 1.365 buffer on both sides of the data
# to allow computing wavelet decomposition without introducing any edge effects. Once wavelet decomposition is done for each event we compute
# features of the classifiers as follows: For each bipolar pair of electrodes we compute mean value of the wavelet power for each spectral frequency present in out wavelet decomposition
# Once we computed features we fit L2 Logistic Regression classifier and generate ROC plot data. We also perform rudimenary behavioral analysis
# keeping track of %'tages of recalled, non-recalled data, change in memory performance between low and high tercile of hte classifier etc...
# all data needed to produce FR3 summary report is being generated in ComposeSessionSummary task
report_pipeline.add_task(ComposeSessionSummary(mark_as_completed=False))


# GeneratePlots task uses data from ComposeSessionSummary task to produce plots that will be included in the report
report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

# GenerateTex task fills in .tex report template ann generates .tex document for the report
report_pipeline.add_task(GenerateTex(mark_as_completed=False))

# GenerateReportPDF compiles generated .tex document and produces pdf with the report
report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
