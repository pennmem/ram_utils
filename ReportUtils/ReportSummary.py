from collections import OrderedDict
from datetime import date
from ReportUtils import MissingDataError, MissingExperimentError, ReportError


class ReportStatus(object):
    def __init__(self, task=None, error=None, message='', line=-1, file=''):
        self.task = task
        self.message = message
        self.line = line
        self.file = file
        self.error = error


class ReportSummaryInventory(object):
    def __init__(self):
        self.summary_dict = OrderedDict()

    def add_report_summary(self, report_summary):
        if report_summary.subject:
            self.summary_dict[report_summary.subject] = report_summary

    def compose_summary(self, detail_level=True):
        d = date.today()
        s = 'Report status summary as of : ' + d.isoformat() + '\n'
        reports_with_missing_data = OrderedDict()
        reports_with_missing_experiment = OrderedDict()
        reports_other_failure = OrderedDict()

        for subject, report_summary in self.summary_dict.items():
            s += '------------------------------------------------------------------------------------\n'
            # s += 'Subject: '+subject+'\n'
            # s+='------------------------------------------------------------------------------------\n'
            s += report_summary.summary(detail_level=detail_level)
            s += '------------------------------------------------------------------------------------\n\n'

        return s


class ReportSummary(object):
    """
    This object holds report status for a single report
    """

    def __init__(self):
        self.subject = None
        self.report_error_status = None
        # self.report_error_status = None
        self.report_status_list = []
        self.changed_resources = None

    def add_changed_resources(self, changed_resources):
        self.changed_resources = changed_resources

    def set_subject(self, subject):
        self.subject = subject

    def add_report_status_obj(self, status_obj):
        self.report_status_list.append(status_obj)

    def add_report_error_status(self, error_status):
        self.report_error_status = error_status

    def add_report_error(self, error):
        error_rs = ReportStatus(error=error)
        self.report_error_status = error_rs

        # if isinstance(error, (MissingDataError, MissingExperimentError, ReportError)):
        #
        #     self.report_error_status = error_rs
        #     # self.report_error_status = error.status
        # else:
        #     self.report_error = error

    def detailed_status(self,detail_level=2):
        s = ''
        if detail_level>0:
            s += '\nDetailed report level %s\n'%str(detail_level)

        if detail_level>1:
            for status in self.report_status_list:
                s += '\n'
                s += 'Task: ' + status.task + ' : ' + status.message + '\n'
                if status.file and status.line >= 0:
                    s += 'file: ' + status.file + '-------- line: ' + str(status.line) + ' \n'

        if detail_level==1:
            s += '\n'
            if len(self.changed_resources):

                s += '------ Changed  Resources -----\n'
                for resource, change_type in self.changed_resources.items():
                    s += 'Changed resource : ' + resource + '\n'
                    s += 'Change type: ' + change_type + '\n'
                    s += '\n'
                    # s += str(self.changed_resources) + '\n'

        return s

    def summary(self, detail_level=2):
        s = ''
        s += '\nSubject: ' + self.subject + '\n'
        s += '------------------------------------------------------------------------------------\n'

        if not self.report_error_status:
            s += 'No errors reported\n'


        if self.report_error_status:
            e = self.report_error_status.error
            if e:
                s += '------------ERROR REPORTED:\n'

                if isinstance(e, MissingDataError):
                    s += 'Missing Data Error: '
                elif isinstance(e, MissingExperimentError):
                    s += 'Missing Experiment Error: '
                elif isinstance(e, ReportError):
                    s += 'General Report Error: '
                else:
                    s += 'Error: '

                s += str(self.report_error_status.error) + '\n'
                if self.report_error_status.file and self.report_error_status.line >= 0:
                    s += 'File:' + self.report_error_status.file + '---------------------------' + ' line: ' + str(
                        self.report_error_status.line) + '\n'



        s += self.detailed_status(detail_level=detail_level)
        # else:
        #     if not self.report_error:
        #         s += ' REPORT_SUCCEFULY GENERATED'
        return s
