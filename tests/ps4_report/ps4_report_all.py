
from ReportUtils import ReportPipeline,CMLParser
from os.path import join
from ptsa.data.readers.IndexReader import JsonIndexReader

parser=CMLParser()
parser.arg('--task','catFR5')
parser.arg('--workspace-dir','/scratch/leond/PS4_catFR5_reports')
parser.arg('--mount-point','/')
parser.arg('--recompute-on-no-status')
parser.arg('--status-output-dir','statuses')




args=parser.parse()

from ComposeSessionSummary import ComposeSessionSummary

from EventPreparation import EventPreparation

from GenerateReportTasks import GeneratePlots,GenerateTex,GeneratePDF

from MontagePreparation import MontagePreparation

from LoadEEG import LoadEEG

from ReportUtils import ReportSummaryInventory

rsi = ReportSummaryInventory(label='PS4_%s'%args.task)

class Params(object):
    def __init__(self):
        self.start_time=-0.03
        self.end_time=0.75

task = args.task
params = Params()

jr = JsonIndexReader(join(args.mount_point,'protocols','r1.json'))
subjects = [s for s in jr.subjects() if any(jr.aggregate_values('ps4_events',subject=s,experiment=args.task))]
for subject in subjects:

        if 'PS4_' in task:
                task = task.split('_')[1]

        report_pipeline = ReportPipeline(subject=subject, task=task,
                                         workspace_dir=join(args.workspace_dir, subject),
                                         mount_point=args.mount_point, sessions=args.sessions,
                                         exit_on_no_change=args.exit_on_no_change,
                                         recompute_on_no_status=args.recompute_on_no_status)

        report_pipeline.add_task(EventPreparation(mark_as_completed=False))

        report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

        report_pipeline.add_task(LoadEEG(params=params, mark_as_completed=True))

        report_pipeline.add_task(ComposeSessionSummary(mark_as_completed=False))

        report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

        report_pipeline.add_task(GenerateTex(mark_as_completed=False))

        report_pipeline.add_task(GeneratePDF(mark_as_completed=False))

        report_pipeline.execute_pipeline()

        # rsi.add_report_summary(report_summary=report_pipeline.get_report_summary())

# rsi.output_json_files(dir=args.status_output_dir)

