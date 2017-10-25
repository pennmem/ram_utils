from ptsa.data.readers.IndexReader import JsonIndexReader
from ReportUtils import CMLParser,ReportPipeline

cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--workspace-dir','scratch/FR1_joint_reports')
#cml_parser.arg('--workspace-dir','/scratch/RAM_maint/automated_reports/FR1_joint_reports_db')
cml_parser.arg('--mount-point','/Volumes/rhino_root')
# cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')

args = cml_parser.parse()


from ReportUtils import ReportSummaryInventory
from FR1EventPreparation import FR1EventPreparation
from RepetitionRatio import RepetitionRatio
from ComputeFR1Powers import ComputeFR1Powers
from MontagePreparation import MontagePreparation
from ComputeFR1HFPowers import ComputeFR1HFPowers
from ComputeTTest import ComputeTTest
from ComputeClassifier import ComputeClassifier,ComputeJointClassifier
from ComposeSessionSummary import ComposeSessionSummary
from GenerateReportTasks import *


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
        self.parallelize = False



params = Params()


json_reader = JsonIndexReader(os.path.join(args.mount_point,'protocols/r1.json'))
subject_set = json_reader.aggregate_values('subjects', experiment='FR1') & json_reader.aggregate_values('subjects', experiment='catFR1')
subjects = []
for s in subject_set:
    montages = json_reader.aggregate_values('montage', subject=s, experiment='FR1') & json_reader.aggregate_values('montage', subject=s, experiment='catFR1')
    for m_ in montages:
        m = str(m_)
        subject = str(s)
        if m!='0':
            subject += '_' + m
        subjects.append(subject)
subjects.sort()

rsi = ReportSummaryInventory(label='FR1_catFR1_joint')

for subject in subjects:
    print '--Generating FR1&CatFR1 joint report for', subject
    if args.skip_subjects is not None and subject in args.skip_subjects:
        continue

    report_pipeline = ReportPipeline(
        args=args,
        subject=subject,
        task='FR1_catFR1_joint',
        experiment='FR1_catFR1_joint',
        workspace_dir=join(args.workspace_dir, subject)
    )

    report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))

    report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeFR1HFPowers(params=params, mark_as_completed=True))

    report_pipeline.add_task(RepetitionRatio(mark_as_completed=True))

    report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeJointClassifier(params=params,mark_as_completed=True))

    report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

    report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

    report_pipeline.add_task(GenerateTex(mark_as_completed=False))

    report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

    report_pipeline.add_task(DeployReportPDF(mark_as_completed=False))

    # starts processing pipeline
    report_pipeline.execute_pipeline()

    rsi.add_report_summary(report_summary=report_pipeline.get_report_summary())

print 'this is summary for all reports report ', rsi.compose_summary(detail_level=1)

rsi.output_json_files(dir=args.status_output_dir)
# rsi.send_email_digest(detail_level_list=[0,1,2])
