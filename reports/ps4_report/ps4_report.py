
from ReportUtils import ReportPipeline,ReportSummaryInventory
from ptsa.data.readers import IndexReader
from os.path import join


from .ComposeSessionSummary import ComposeSessionSummary
from .EventPreparation import EventPreparation
from .GenerateReportTasks import GeneratePlots,GenerateTex,GeneratePDF
from ReportTasks.DeployReportPDF import DeployReportPDF


def run_report(args):
    report_pipeline = build_pipeline(args)

    report_pipeline.execute_pipeline()

def run_all_reports(args):
    rsi = ReportSummaryInventory()
    jr = IndexReader.JsonIndexReader(join(args.mount_point, 'protocols', 'r1.json'))
    subjects = set(jr.subjects(experiment=args.task))
    for subject in subjects:
        montages = jr.montages(subject=subject, experiment=args.task)
        for montage in montages:
            subject += '_%s' % str(montage) if montage > 0 else ''
            args.subject = subject
            report_pipeline = build_pipeline(args)
            report_pipeline.add_task(DeployReportPDF(False))
            report_pipeline.execute_pipeline()
            rsi.add_report_summary(report_pipeline.get_report_summary())

    rsi.output_json_files(args.report_status_dir)


def build_pipeline(args):
    report_pipeline = ReportPipeline(subject=args.subject, task=args.task,
                                     workspace_dir=join(args.workspace_dir, args.subject),
                                     mount_point=args.mount_point, sessions=args.sessions,
                                     exit_on_no_change=args.exit_on_no_change,
                                     recompute_on_no_status=args.recompute_on_no_status)
    report_pipeline.add_task(EventPreparation(mark_as_completed=False))
    report_pipeline.add_task(ComposeSessionSummary(mark_as_completed=False))
    report_pipeline.add_task(GeneratePlots(mark_as_completed=False))
    report_pipeline.add_task(GenerateTex(mark_as_completed=False))
    report_pipeline.add_task(GeneratePDF(mark_as_completed=False))
    return report_pipeline

