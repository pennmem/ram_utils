from glob import glob
import re

import sys
from setup_utils import parse_command_line, configure_python_paths
from os.path import join

from ReportUtils import CMLParser,ReportPipeline

cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--workspace-dir','/scratch/mswat/automated_reports/FR3_reports')
cml_parser.arg('--task','RAM_FR3')
cml_parser.arg('--mount-point','')
cml_parser.arg('--recompute-on-no-status')

# cml_parser.arg('--exit-on-no-change')

args = cml_parser.parse()

from ReportUtils import ReportSummaryInventory

from FREventPreparation import FREventPreparation

from EventPreparation import EventPreparation

from MathEventPreparation import MathEventPreparation

from ComputeFRPowers import ComputeFRPowers

from ComputeClassifier import ComputeClassifier

from ComputeFRStimPowers import ComputeFRStimPowers

from TalPreparation import TalPreparation

from ComputeFRStimTable import ComputeFRStimTable

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.ttest_frange = (70.0, 200.0)

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200

        self.include_fr1 = True
        self.include_catfr1 = True


params = Params()

task = args.task

def find_subjects_by_task(task):
    ev_files = glob(args.mount_point + '/data/events/%s/R*_events.mat' % task)
    return [re.search(r'R1\d\d\d[A-Z](_\d+)?', f).group() for f in ev_files]


subjects = find_subjects_by_task(task)
subjects.sort()


subject_fail_list = []
subject_missing_experiment_list = []
subject_missing_data_list = []

rsi = ReportSummaryInventory(label=args.experiment)

for subject in subjects:
    print subject
    if args.skip_subjects is not None and subject in args.skip_subjects:
        continue

    # sets up processing pipeline

    report_pipeline = ReportPipeline(subject=subject,
                                     experiment=args.experiment,
                                     task=task,
                                     workspace_dir=join(args.workspace_dir, subject),
                                     mount_point=args.mount_point,
                                     exit_on_no_change=args.exit_on_no_change,
                                     recompute_on_no_status=args.recompute_on_no_status)

    report_pipeline.add_task(FREventPreparation(params=params, mark_as_completed=False))

    report_pipeline.add_task(EventPreparation(mark_as_completed=False))

    report_pipeline.add_task(MathEventPreparation(mark_as_completed=False))

    report_pipeline.add_task(TalPreparation(mark_as_completed=False))

    report_pipeline.add_task(ComputeFRPowers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeFRStimPowers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeFRStimTable(params=params, mark_as_completed=True))

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
