from RamPipeline import RamTask
from ReportUtils import MissingExperimentError, MissingDataError, ReportError, ReportStatus


class ReportRamTask(RamTask):
    def __init__(self, mark_as_completed):
        super(ReportRamTask, self).__init__(mark_as_completed=mark_as_completed)

    def add_report_status(self, message=''):
        rs = ReportStatus(task=self.__class__.__name__, message=message)
        self.pipeline.report_summary.add_report_status(status_obj=rs)
        self.pipeline.report_summary.set_subject(self.pipeline.subject)

    def raise_and_log_report_exception(self, exception_type='', exception_message=''):

        if exception_type == 'MissingExperimentError':
            # rs = ReportStatus(subject=self.pipeline.subject)
            excpt = MissingExperimentError(
                message=exception_message,
                # status=rs
            )
        elif exception_type == 'MissingDataError':
            # rs = ReportStatus(subject=self.pipeline.subject)
            excpt = MissingDataError(
                message=exception_message,
                # status=rs
            )

        else:
            # rs = ReportStatus(subject=self.pipeline.subject)
            excpt = ReportError(
                message=exception_message,
                # status=rs
            )

        self.pipeline.report_summary.add_report_error(error=excpt)
        self.pipeline.report_summary.set_subject(self.pipeline.subject)

        raise excpt
