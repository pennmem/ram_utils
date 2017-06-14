from ...ReportUtils import ReportPipeline,ReportSummaryInventory
from ptsa.data.readers import IndexReader
from os.path import join
import numpy as np

from .FR1EventPreparation import FR1EventPreparation

from .RepetitionRatio import RepetitionRatio

from .ComputeFR1Powers import ComputeFR1Powers

from .MontagePreparation import MontagePreparation

from .ComputeFR1HFPowers import ComputeFR1HFPowers

from .ComputeTTest import ComputeTTest

from .ComputeClassifier import ComputeClassifier,ComputeJointClassifier

from .ComposeSessionSummary import ComposeSessionSummary

from .GenerateReportTasks import GenerateReportPDF,GeneratePlots,GenerateTex,DeployReportPDF


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.fr1_retrieval_start_time= -0.525
        self.fr1_retrieval_end_time=0.0
        self.fr1_retrieval_buf = 0.524


        self.hfs_start_time = 0.0
        self.hfs_end_time = 1.6
        self.hfs_buf = 1.0

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(6), np.log10(180), 8)
        self.hfs = np.logspace(np.log10(2), np.log10(200), 50)
        self.hfs = self.hfs[self.hfs>=70.0]

        self.encoding_samples_weight=2.5

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200
        self.parallelize = True

def run_report(args):
    report_pipeline = build_pipeline(args)


    # starts processing pipeline
    report_pipeline.execute_pipeline()

def run_all_reports(args):
    rsi = ReportSummaryInventory()
    jr = IndexReader.JsonIndexReader(join(args.mount_point,'protocols','r1.json'))
    subjects = set(jr.subjects(experiment=args.task))
    for subject in subjects:
        montages = jr.montages(subject=subject,experiment='FR3')
        for montage in montages:
            subject += '_%s'%str(montage) if montage>0 else ''
            args.subject=subject
            report_pipeline= build_pipeline(args)
            report_pipeline.add_task(DeployReportPDF(False))
            report_pipeline.execute_pipeline()
            rsi.add_report_summary(report_pipeline.get_report_summary())

    rsi.output_json_files(args.report_status_dir)

def build_pipeline(args):
    params = Params()
    # sets up processing pipeline
    report_pipeline = ReportPipeline(subject=args.subject, task=args.task, experiment=args.task, sessions=args.sessions,
                                     workspace_dir=join(args.workspace_dir, args.subject), mount_point=args.mount_point,
                                     exit_on_no_change=args.exit_on_no_change,
                                     recompute_on_no_status=args.recompute_on_no_status)
    report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))
    report_pipeline.add_task(MontagePreparation(params, mark_as_completed=False))
    if 'cat' in args.task:
        report_pipeline.add_task(RepetitionRatio(mark_as_completed=True))
    report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))
    report_pipeline.add_task(ComputeFR1HFPowers(params=params, mark_as_completed=True))
    report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=True))
    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))
    report_pipeline.add_task(ComputeJointClassifier(params=params, mark_as_completed=True))
    report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))
    report_pipeline.add_task(GeneratePlots(mark_as_completed=False))
    report_pipeline.add_task(GenerateTex(mark_as_completed=False))
    report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))
    return report_pipeline
