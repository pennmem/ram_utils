from RamPipeline import RamTask
from ReportUtils import *
import inspect
from os.path import *
from hashlib import md5

class ReportRamTask(RamTask):
    def __init__(self, mark_as_completed,name=None):
        super(ReportRamTask, self).__init__(mark_as_completed=mark_as_completed)
        self.hash = md5()
        self.set_name(name)


    def get_code_data(self):
        """
        returns name of the file and line number of the calling function
        :return {tuple: str, int}: filename, line number
        """
        (frame, file, line,function_name, lines, index) = inspect.getouterframes(inspect.currentframe())[1]
        return file, line

    def add_report_status(self, message=''):
        (frame, file, line,function_name, lines, index) = inspect.getouterframes(inspect.currentframe())[1]

        rs = ReportStatus(task=self.__class__.__name__, message=message, file=file, line=line)

        self.pipeline.report_summary.add_report_status_obj(status_obj=rs)


    # def add_report_file(self,file):
    #     self.pipeline.report_summary.add_report_file(file=file)
    #
    #
    # def add_report_link(self,link):
    #     self.pipeline.report_summary.add_report_link(link=link)


    def pre(self):
        if self.pipeline.report_summary is None:
            return
        self.pipeline.report_summary.set_subject(self.pipeline.subject)

    def post(self):
        if self.pipeline.report_summary is None:
            return

        message = 'TASK COMPLETED OK'
        (frame, file, line,function_name, lines, index) = inspect.getouterframes(inspect.currentframe())[0]
        file, line = self.get_code_data()
        rs = ReportStatus(task=self.__class__.__name__, message=message, file=file, line=line)

        self.pipeline.report_summary.add_report_status_obj(status_obj=rs)

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

        elif exception_type == 'NumericalError':
            # rs = ReportStatus(subject=self.pipeline.subject)
            excpt = NumericalError(
                message=exception_message,
                # status=rs
            )


        else:
            # rs = ReportStatus(subject=self.pipeline.subject)
            excpt = ReportError(
                message=exception_message,
                # status=rs
            )

        # file, line = self.get_code_data()
        (frame, file, line,function_name, lines, index) = inspect.getouterframes(inspect.currentframe())[1]
        error_rs = ReportStatus(task=self.__class__.__name__, error=excpt, file=file, line=line)

        self.pipeline.report_summary.add_report_error_status(error_status=error_rs)


        raise excpt
