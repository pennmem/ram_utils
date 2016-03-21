
from collections import OrderedDict
from datetime import date
from ReportUtils import MissingDataError,MissingExperimentError, ReportError

class ReportStatus(object):
    def __init__(self,  task=None,message=''):

        self.task = task
        self.message = message


class ReportSummaryInventory(object):
    def __init__(self):
        self.summary_dict = OrderedDict()

    def add_report_summary(self,report_summary):
        self.summary_dict[report_summary.subject] = report_summary

    def compose_summary(self, detailed=True):
        d = date.today()
        s = 'Report status summary as of : '+d.isoformat()+'\n'
        reports_with_missing_data = OrderedDict()
        reports_with_missing_experiment = OrderedDict()
        reports_other_failure = OrderedDict()

        for subject, report_summary in self.summary_dict.items():
            s += report_summary.summary(detailed=detailed)

        return s

class ReportSummary(object):
    """
    This object holds report status for a single report
    """
    def __init__(self):
        self.subject = None
        self.report_error = None
        # self.report_error_status = None
        self.report_status_list = []


    def set_subject(self,subject):
        self.subject=subject


    def add_report_status_obj(self,status_obj):
        self.report_status_list.append(status_obj)


    def add_report_error(self,error):
        if isinstance(error,(MissingDataError,MissingExperimentError, ReportError)):
            self.report_error = error
            # self.report_error_status = error.status

    def detailed_status(self):
        s = ''

        for status in self.report_status_list:
            s += 'Task: '+ status.task + ' : ' + status.message + '\n'

        return s


    def summary(self, detailed=True):




        if not self.report_error:
            return 'No errors reported'

        s = '\nSubject '+self.subject+'\n'
        e = self.report_error
        if e:
            s += '------------ERROR REPORTED:\n'

            if isinstance(e,MissingDataError):
                s += 'Missing Data Error: '
            elif isinstance(e,MissingExperimentError):
                s += 'Missing Experiment Error: '
            elif isinstance(e,ReportError):
                s += 'General Report Error: '
            else:
                s += 'Error: '

            s += str(self.report_error)+'\n'
            s += '---------------------------\n'


        if detailed:
            s += '\nDetailed report\n'
            s+= self.detailed_status()
        else:
            if not self.report_error:
                s+= ' REPORT_SUCCEFULY GENERATED'
        return s

