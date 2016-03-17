
from collections import OrderedDict


class ReportStatus(object):
    def __init__(self, exception=None, subject=None):
        self.exception = exception
        self.subject = subject

    def add_exception(self,exception):
        self.exception = exception


class ReportSummary(object):
    def __init__(self):
        pass
        self.report_status_dict = OrderedDict()

    def add_report_status(self,subject,status_obj):
        self.report_status_dict[subject] = status_obj

    def compose_summary(self):
        pass
