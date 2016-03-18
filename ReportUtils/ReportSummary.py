
from collections import OrderedDict
from datetime import date
from ReportUtils import MissingDataError,MissingExperimentError, ReportError

class ReportStatus(object):
    def __init__(self, exception=None, task=None,message=''):
        # self.exception = exception
        self.task = task
        self.message = ''

    # def add_exception(self,exception):
    #     self.exception = exception

class ReportSummaryInventory(object):
    def __init__(self):
        self.inventory_dict = OrderedDict()

    def add_report_summary(self,report_summary):
        self.inventory_dict[report_summary.subject] = report_summary

    def compose_summary(self):
        d = date.today()
        s = 'Report status summary as of : '+d.isoformat()+'\n'
        reports_with_missing_data = OrderedDict()
        reports_with_missing_experiment = OrderedDict()
        reports_other_failure = OrderedDict()

        for subject, report_status in self.inventory_dict.items():
            print 'subject=',subject
        #     report_status_obj = report_status()
        #     if not report_status_obj:
        #         reports_other_failure[subject] = 'Report Status could not be identified'
        #         continue
        #
        #     excpt = report_status_obj.exception
        #     if excpt is not None:
        #         if isinstance(excpt,MissingDataError):
        #             reports_with_missing_data[subject]=str(excpt)
        #         elif isinstance(excpt,MissingExperimentError):
        #             reports_with_missing_experiment[subject] = str(excpt)
        #         else:
        #             reports_other_failure[subject] = str(excpt)
        #     else:
        #         reports_other_failure[subject] = 'Unknown reason for report failure to complete'
        #
        # if len(reports_with_missing_data):
        #     s+= '\nReports that had missing data:\n'
        #     for subject, msg in reports_with_missing_data.items():
        #         s+='Subject: '+subject+' \nReason:'+ msg+'\n'
        #
        # if len(reports_with_missing_experiment):
        #     s+= '\nReports that had missing experiment:\n'
        #     for subject, msg in reports_with_missing_experiment.items():
        #         s+='Subject: '+subject+'\nreason:'+ msg+'\n'

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

    def add_report_status(self,status_obj):


        self.report_status_list.append(status_obj)

    def add_report_error(self,error):
        if isinstance(error,(MissingDataError,MissingExperimentError, ReportError)):
            self.report_error = error
            # self.report_error_status = error.status


    def summary(self):


        if not self.report_error:
            return 'No errors reported'

        report_error_obj = self.report_error()
        excpt = report_error_obj.exception
        if excpt is not None:
            if isinstance(excpt,MissingDataError):
                return 'Missing Data Error - message:'+str(excpt)
            elif isinstance(excpt,MissingExperimentError):
                return 'Missing Experiment Error - message:'+str(excpt)
            elif isinstance(excpt,ReportError):
                return 'General Report Error - message:'+str(excpt)
            else:
                return 'Error'+str(excpt)
        else:
            return 'Unknown report error'


    # def compose_summary(self):
    #     d = date.today()
    #     s = 'Report status summary as of : '+d.isoformat()+'\n'
    #     reports_with_missing_data = OrderedDict()
    #     reports_with_missing_experiment = OrderedDict()
    #     reports_other_failure = OrderedDict()
    #
    #     for subject, report_status in self.report_status_dict.items():
    #
    #         report_status_obj = report_status()
    #         if not report_status_obj:
    #             reports_other_failure[subject] = 'Report Status could not be identified'
    #             continue
    #
    #         excpt = report_status_obj.exception
    #         if excpt is not None:
    #             if isinstance(excpt,MissingDataError):
    #                 reports_with_missing_data[subject]=str(excpt)
    #             elif isinstance(excpt,MissingExperimentError):
    #                 reports_with_missing_experiment[subject] = str(excpt)
    #             else:
    #                 reports_other_failure[subject] = str(excpt)
    #         else:
    #             reports_other_failure[subject] = 'Unknown reason for report failure to complete'
    #
    #     if len(reports_with_missing_data):
    #         s+= '\nReports that had missing data:\n'
    #         for subject, msg in reports_with_missing_data.items():
    #             s+='Subject: '+subject+' \nReason:'+ msg+'\n'
    #
    #     if len(reports_with_missing_experiment):
    #         s+= '\nReports that had missing experiment:\n'
    #         for subject, msg in reports_with_missing_experiment.items():
    #             s+='Subject: '+subject+'\nreason:'+ msg+'\n'
    #
    #     return s