import sys
import os

from ptsa.data.readers.IndexReader import JsonIndexReader

from ReportUtils import CMLParser,ReportPipeline


cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--task','FR1')
cml_parser.arg('--workspace-dir','/scratch/RAM_maint/automated_reports_json/FR1_reports')
cml_parser.arg('--mount-point','')
cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')

#cml_parser.arg('--task','FR1')
#cml_parser.arg('--workspace-dir','/scratch/RAM_maint/automated_reports/FR1_reports')
# cml_parser.arg('--mount-point','/Users/m')
#cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--python-path','/Users/m/PTSA_NEW_GIT')
#cml_parser.arg('--exit-on-no-change')


args = cml_parser.parse()



from ReportUtils import ReportSummaryInventory, ReportSummary
from ReportUtils import ReportPipelineBase


from FR1EventPreparation import FR1EventPreparation

from ComputeFR1Powers import ComputeFR1Powers

from MontagePreparation import MontagePreparation

from ComputeFR1HFPowers import ComputeFR1HFPowers

from ComputeTTest import ComputeTTest

from ComputeClassifier import ComputeClassifier

from ComposeSessionSummary import ComposeSessionSummary

from RepetitionRatio import RepetitionRatio

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
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

task = args.task

json_reader = JsonIndexReader(os.path.join(args.mount_point,'protocols/r1.json'))
subject_set = json_reader.aggregate_values('subjects', experiment=task)
subjects = []
for s in subject_set:
    montages = json_reader.aggregate_values('montage', subject=s, experiment=task)
    for m_ in montages:
        m = str(m_)
        subject = str(s)
        if m!='0':
            subject += '_' + m
        subjects.append(subject)
subjects.sort()


rsi = ReportSummaryInventory(label=task)


for subject in subjects:
    if args.skip_subjects is not None and subject in args.skip_subjects:
        continue
    print '--Generating', task, 'report for', subject

    # sets up processing pipeline
    # report_pipeline = ReportPipeline(subject=subject, task=task, experiment=task,
    #                                        workspace_dir=join(args.workspace_dir,task+'_'+subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,recompute_on_no_status=args.recompute_on_no_status)


    # report_pipeline = ReportPipeline(subject=subject, task=task, experiment=task,
    #                                  workspace_dir=join(args.workspace_dir, task + '_' + subject),
    #                                  mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
    #                                  recompute_on_no_status=args.recompute_on_no_status)

    report_pipeline = ReportPipeline(
                                     args=args,
                                     subject=subject,
                                     workspace_dir=join(args.workspace_dir,subject)
                                     )

    report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))

    report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

    if 'cat' in args.task:
        report_pipeline.add_task(RepetitionRatio(mark_as_completed=False))

    report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeFR1HFPowers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

    report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

    report_pipeline.add_task(GenerateTex(mark_as_completed=False))

    report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

    report_pipeline.add_task(DeployReportPDF(mark_as_completed=False))

    report_pipeline.execute_pipeline()

    rsi.add_report_summary(report_summary=report_pipeline.get_report_summary())


print 'this is summary for all reports report ', rsi.compose_summary(detail_level=1)

rsi.output_json_files(dir=args.status_output_dir)
#rsi.send_email_digest()

