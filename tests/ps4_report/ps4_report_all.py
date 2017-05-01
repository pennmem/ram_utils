
from ReportUtils import ReportPipeline,CMLParser
from os.path import join
from ptsa.data.readers.IndexReader import JsonIndexReader

parser=CMLParser()
parser.arg('--subject','R1293P')
parser.arg('--task','FR5')
parser.arg('--workspace-dir','/Users/leond/ps4_reports')
parser.arg('--mount-point','/Volumes/rhino_root/')
parser.arg('--recompute-on-no-status')




args=parser.parse()

from ComposeSessionSummary import ComposeSessionSummary

from EventPreparation import EventPreparation

from GenerateReportTasks import GeneratePlots,GenerateTex,GeneratePDF

from ReportTasks.DeployReportPDF import DeployReportPDF

from ReportUtils import ReportSummaryInventory

rsi = ReportSummaryInventory(label='PS4')

jr = JsonIndexReader(join(args.mount_point,'protocols','r1.json'))
subjects = jr.subjects()

for subject in subjects:
    if jr.aggregate_values('ps4_events',subject=subject) and args.task in jr.experiments(subject=subject):


        report_pipeline = ReportPipeline(subject=args.subject,task=args.task,workspace_dir= join(args.workspace_dir,args.subject),
                                         mount_point=args.mount_point,sessions=args.sessions,
                                         exit_on_no_change=args.exit_on_no_change,recompute_on_no_status=args.recompute_on_no_status)

        report_pipeline.add_task(EventPreparation(mark_as_completed=False))

        report_pipeline.add_task(ComposeSessionSummary(mark_as_completed=False))

        report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

        report_pipeline.add_task(GenerateTex(mark_as_completed=False))

        report_pipeline.add_task(GeneratePDF(mark_as_completed=False))

        report_pipeline.add_task(DeployReportPDF())

        report_pipeline.execute_pipeline()

        rsi.add_report_summary(report_summary=report_pipeline.get_report_summary())

rsi.output_json_files(dir=args.status_output_dir)

