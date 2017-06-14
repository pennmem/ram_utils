
from ReportUtils import ReportPipeline,CMLParser
from os.path import join
from ptsa.data.readers.IndexReader import JsonIndexReader

parser=CMLParser()
parser.arg('--task','catFR5')
parser.arg('--workspace-dir','ps4_reports')
parser.arg('--mount-point','/Volumes/rhino_root/')
parser.arg('--recompute-on-no-status')
parser.arg('--status-output-dir','statuses')




args=parser.parse()

from ComposeSessionSummary import ComposeSessionSummary

from EventPreparation import EventPreparation

from GenerateReportTasks import GeneratePlots,GenerateTex,GeneratePDF

from ReportTasks.DeployReportPDF import DeployReportPDF

from ReportUtils import ReportSummaryInventory

rsi = ReportSummaryInventory(label='PS4_%s'%args.task)

jr = JsonIndexReader(join(args.mount_point,'protocols','r1.json'))
subjects = [s for s in jr.subjects() if jr.aggregate_values('ps4_events',subject=s,experiment=args.task)]
for subject in subjects:

        report_pipeline = ReportPipeline(subject=subject,task=args.task,workspace_dir= join(args.workspace_dir,subject),
                                         mount_point=args.mount_point,
                                         exit_on_no_change=args.exit_on_no_change,recompute_on_no_status=args.recompute_on_no_status)

        report_pipeline.add_task(EventPreparation(mark_as_completed=False))

        report_pipeline.add_task(ComposeSessionSummary(mark_as_completed=False))

        report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

        report_pipeline.add_task(GenerateTex(mark_as_completed=False))

        report_pipeline.add_task(GeneratePDF(mark_as_completed=False))

        # report_pipeline.add_task(DeployReportPDF())

        report_pipeline.execute_pipeline()

        rsi.add_report_summary(report_summary=report_pipeline.get_report_summary())

rsi.output_json_files(dir=args.status_output_dir)

