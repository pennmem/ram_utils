from glob import glob
import re

import sys
from os.path import *
from setup_utils import parse_command_line, configure_python_paths
from ptsa.data.readers.IndexReader import JsonIndexReader
from ReportUtils import CMLParser,ReportPipeline

cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--workspace-dir','/scratch/RAM_maint/automated_reports_json/PS2.1_TH')
cml_parser.arg('--mount-point','')
cml_parser.arg('--recompute-on-no-status')
cml_parser.arg('--task','PS2.1')
cml_parser.arg('--exit-on-no-change')

args = cml_parser.parse()


import numpy as np
from ReportUtils import ReportSummaryInventory

from THEventPreparation import THEventPreparation
from ControlEventPreparation import ControlEventPreparation
from PSEventPreparation import PSEventPreparation

from ComputeTH1ClassPowers import ComputeTH1ClassPowers
from ComputeControlPowers import ComputeControlPowers
from ComputePSPowers import ComputePSPowers

from MontagePreparation import MontagePreparation

from ComputeClassifier import ComputeClassifier

from ComputeControlTable import ComputeControlTable
from ComputePSTable import ComputePSTable

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.th1_start_time = -1.2
        self.th1_end_time = 0.5
        self.th1_buf = 1.7

        self.sham1_start_time = 1.0
        self.sham1_end_time = 2.0
        self.sham_buf = 1.0

        self.sham2_start_time = 10.0 - 3.7
        self.sham2_end_time = 10.0 - 2.7

        self.ps_start_time = -1.0
        self.ps_end_time = 0.0
        self.ps_buf = 1.0
        self.ps_offset = 0.1

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(1.0), np.log10(200.0), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200
        self.doStratKFold = False
        self.n_folds = 8

        self.doConf_classification = True
        self.doDist_classification = False
        self.doClass_wTranspose    = False


params = Params()
task = args.task

json_reader = JsonIndexReader(os.path.join(args.mount_point,'protocols/r1.json'))
subject_set = json_reader.aggregate_values('subjects', experiment=task) & json_reader.aggregate_values('subjects', experiment='TH1')
print subject_set
subjects = []
for s in subject_set:
    montages = json_reader.aggregate_values('montage', subject=s, experiment=task)
    subject = str(s)
    for m_ in montages:
        m = str(m_)
        has_th1 = bool(json_reader.aggregate_values('sessions', subject=subject, montage=m, experiment='TH1'))
        if has_th1:
            if m!='0':
                subject += '_' + m
            subjects.append(subject)
subjects.sort()


# print "subjects=",subjects
# sys.exit()

subject_fail_list = []
subject_missing_experiment_list = []
subject_missing_data_list = []

rsi = ReportSummaryInventory(label=args.task)

for subject in subjects:
    print subject
    if args.skip_subjects is not None and subject in args.skip_subjects:
        continue

    # sets up processing pipeline

    report_pipeline = ReportPipeline(subject=subject,
                                     experiment=args.task,
                                     experiment_label=args.task+'_TH', # NOTE: I am adding experiment_label tso that json status outpu has proper experiment annotation that does not overlap with other repor experiments names
                                     workspace_dir=join(args.workspace_dir, subject),
                                     mount_point=args.mount_point,
                                     exit_on_no_change=args.exit_on_no_change,
                                     recompute_on_no_status=args.recompute_on_no_status)

    report_pipeline.add_task(THEventPreparation(mark_as_completed=False))

    # report_pipeline.add_task(ControlEventPreparation(params=params, mark_as_completed=False))

    report_pipeline.add_task(PSEventPreparation(mark_as_completed=True))

    report_pipeline.add_task(MontagePreparation(mark_as_completed=True))

    report_pipeline.add_task(ComputeTH1ClassPowers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeControlPowers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputePSPowers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeControlTable(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputePSTable(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

    report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

    report_pipeline.add_task(GenerateTex(mark_as_completed=False))

    report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

    report_pipeline.add_task(DeployReportPDF(mark_as_completed=False))

    report_pipeline.execute_pipeline()

    rsi.add_report_summary(report_summary=report_pipeline.get_report_summary())

print 'all subjects = ', subjects
print 'subject_fail_list=', subject_fail_list
print 'subject_missing_experiment_list=', subject_missing_experiment_list
print 'subject_missing_data_list=', subject_missing_data_list

print 'this is summary for all reports report ', rsi.compose_summary(detail_level=1)

rsi.output_json_files(dir=args.status_output_dir)
# rsi.send_email_digest()
# print report_pipeline.report_summary.compose_summary()
