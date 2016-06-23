import sys
from os.path import *

from ReportUtils import CMLParser,ReportPipeline


cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--subject','R1175N')
cml_parser.arg('--task','RAM_PAL3')
cml_parser.arg('--workspace-dir','/scratch/busygin/PAL3_reports')
cml_parser.arg('--mount-point','')
#cml_parser.arg('--recompute-on-no-status')

args = cml_parser.parse()


import numpy as np
from RamPipeline import RamPipeline
from RamPipeline import RamTask

from PAL1EventPreparation import PAL1EventPreparation

from EventPreparation import EventPreparation

from MathEventPreparation import MathEventPreparation

from ComputePAL1Powers import ComputePAL1Powers

from ComputeClassifier import ComputeClassifier

from ComputePALStimPowers import ComputePALStimPowers

from TalPreparation import TalPreparation

from ComputePALStimTable import ComputePALStimTable

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.pal1_start_time = 0.3
        self.pal1_end_time = 2.0
        self.pal1_buf = 1.0

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.ttest_frange = (70.0, 200.0)

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


params = Params()


# sets up processing pipeline - the entire computation is divided into separate tasks that are manages
# by the Pipeline object
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,experiment=args.task,
                                 workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)


# PAL1EventPreparation task reads experiment summary file (so-called events file) containing information about
# events that took place during the course record-only sessions (PAL1) that were used to train the classifier.
# This file contains the associative pair presentation events.
report_pipeline.add_task(PAL1EventPreparation(params=params, mark_as_completed=False))

# EventPreparation task reads experiment summary file (so-called events file) containing information about
# events that took place during the course the analyzed closed-loop session (PAL3).
# This file contains the associative pair presentation events.
report_pipeline.add_task(EventPreparation(mark_as_completed=False))

# MathEventPreparation task reads experiment summary file (so-called events file) containing information
# about events that took place during the course of experiment. This file focuses on the math distractor events
report_pipeline.add_task(MathEventPreparation(mark_as_completed=False))

# TalPreparation task reads information about electrodes (monopolar and bipolar) localizations
report_pipeline.add_task(TalPreparation(mark_as_completed=False))

# ComputePAL1Powers reads PAL1 session(s) EEG data for the subject, then splits it
# into short segments corresponding to single associative pair presentation events.
# A pair presentation event will include 1.7 second worth of EEG data starting from
# 0.3 seconds into the pair presentation onset. It is padded with 1.0 sec buffer on
# both sides of the data to allow computing wavelet decomposition without introducing any edge effects.
report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))

# Once wavelet decomposition is done for each event we compute features of the classifiers as follows:
# For each bipolar pair of electrodes we compute mean value of the wavelet power for each spectral
# frequency present in our wavelet decomposition. Once we computed the features, we fit L2 Logistic
# Regression classifier and generate ROC plot data.
report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

# ComputePALStimPowers reads PAL3 session(s) EEG data for the subject, then splits it
# into short segments corresponding to single associative pair presentation events.
# A pair presentation event will include 1.7 second worth of EEG data starting from
# 0.3 seconds into the pair presentation onset. It is padded with 1.0 sec buffer on
# both sides of the data to allow computing wavelet decomposition without introducing any edge effects.
report_pipeline.add_task(ComputePALStimPowers(params=params, mark_as_completed=True))

# ComputePALStimTable computes the classifier output and the probability of recall based on it
# for each associative pair event in PAL3 session(s). It constructs a pandas DataFrame table where
# each row corresponds to a presented pair and contains its session, list number, serial position,
# whether it was a stim list or not, whether stim was applied, or the classifier output.
report_pipeline.add_task(ComputePALStimTable(params=params, mark_as_completed=True))

# ComposeSessionSummary collects behavioral and classifier-based statistics (%'tages of recalled,
# non-recalled data, change in memory performance between low and high tercile of the classifier etc...)
# All data needed to produce PAL3 summary report are being generated in ComposeSessionSummary task
report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

# GeneratePlots task uses data from ComposeSessionSummary task to produce plots that will be included in the report
report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

# GenerateTex task fills in .tex report template ann generates .tex document for the report
report_pipeline.add_task(GenerateTex(mark_as_completed=False))

# GenerateReportPDF compiles generated .tex document and produces pdf with the report
report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
