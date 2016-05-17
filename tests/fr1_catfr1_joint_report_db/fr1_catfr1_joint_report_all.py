import sys
from glob import glob
import re

from ReportUtils import CMLParser,ReportPipeline

cml_parser = CMLParser(arg_count_threshold=1)
#cml_parser.arg('--workspace-dir','/scratch/mswat/automated_reports/FR1_CatFr1_check_1')
cml_parser.arg('--workspace-dir','/scratch/busygin/FR1_joint_reports_db')
cml_parser.arg('--mount-point','')
cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')

args = cml_parser.parse()


from ReportUtils import ReportSummaryInventory

from FR1EventPreparation import FR1EventPreparation

from MathEventPreparation import MathEventPreparation

from ComputeFR1Powers import ComputeFR1Powers

from MontagePreparation import MontagePreparation

from ComputeFR1HFPowers import ComputeFR1HFPowers

from ComputeTTest import ComputeTTest

from ComputeClassifier import ComputeClassifier

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.api_monopolar_url = 'https://stimstaging.psych.upenn.edu/api/rhino/1.0/configured-contacts-by-montage-codes.json'
        self.api_bipolar_url = 'https://stimstaging.psych.upenn.edu/api/rhino/1.0/configured-pairs-by-montage-codes.json'

        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.hfs_start_time = 0.0
        self.hfs_end_time = 1.6
        self.hfs_buf = 1.0

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)
        self.hfs = np.logspace(np.log10(2), np.log10(200), 50)
        self.hfs = self.hfs[self.hfs>=70.0]

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


params = Params()


def find_subjects_by_task(task):
    ev_files = glob(args.mount_point + ('/data/events/%s/R*_events.mat' % task))
    return [re.search(r'R\d\d\d\d[A-Z](_\d+)?', f).group() for f in ev_files]


subjects = list(set(find_subjects_by_task('RAM_FR1')).intersection(find_subjects_by_task('RAM_CatFR1')))
subjects.sort()


rsi = ReportSummaryInventory(label='RAM_FR1_CatFR1_joint')

for subject in subjects:
    print '--Generating FR1&CatFR1 joint report for', subject
    if args.skip_subjects is not None and subject in args.skip_subjects:
        continue


    # sets up processing pipeline
    # report_pipeline = ReportPipeline(subject=subject,
    #                                  task='RAM_FR1_CatFR1_joint',
    #                                  experiment='RAM_FR1_CatFR1_joint',
    #                                  workspace_dir=join(args.workspace_dir, subject),
    #                                  mount_point=args.mount_point,
    #                                  exit_on_no_change=args.exit_on_no_change,
    #                                  recompute_on_no_status=args.recompute_on_no_status)

    report_pipeline = ReportPipeline(
        args=args,
        subject=subject,
        workspace_dir=join(args.workspace_dir,  subject)
    )

    report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))

    report_pipeline.add_task(MathEventPreparation(mark_as_completed=False))

    report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeFR1HFPowers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

    report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

    report_pipeline.add_task(GenerateTex(mark_as_completed=False))

    report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

    # report_pipeline.add_task(DeployReportPDF(mark_as_completed=False))

    # starts processing pipeline
    report_pipeline.execute_pipeline()

    rsi.add_report_summary(report_summary=report_pipeline.get_report_summary())



print 'this is summary for all reports report ', rsi.compose_summary(detail_level=1)

rsi.output_json_files(dir=args.status_output_dir)
# rsi.send_email_digest(detail_level_list=[0,1,2])
