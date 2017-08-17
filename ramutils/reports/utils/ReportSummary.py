from collections import OrderedDict
from datetime import date
from ReportUtils import *
from ReportUtils.ReportExceptions import *

from JSONUtils import JSONNode
from os.path import *
import os
import shutil
import base64


class ReportStatus(object):
    def __init__(self, task=None, error=None, message='', line=-1, file=''):
        self.task = task
        self.message = message
        self.line = line
        self.file = file
        self.error = error

        # def to_json(self):


class ReportSummaryInventory(object):
    def __init__(self, label=''):
        self.summary_dict = OrderedDict()

        self.reports_generated_count = 0
        self.reports_error_count = 0
        self.label = label

    def add_report_summary(self, report_summary):
        if report_summary.subject:
            self.summary_dict[report_summary.subject] = report_summary

    def output_json_files(self, dir=''):

        if dir and not isdir(dir):
            try:
                os.makedirs(dir)
            except OSError:
                return

        for subject, report_summary in self.summary_dict.items():
            # outpath = join(dir, subject + '_' + report_summary.experiment_name + '_report.json')
            outpath = join(dir, subject + '_report.json')
            jn = report_summary.to_json()
            jn.write(outpath)

    def compose_summary(self, detail_level=2):
        d = date.today()
        s = 'Report status summary as of {} :\n '.format(d.isoformat())

        reports_with_missing_data = OrderedDict()
        reports_with_missing_experiment = OrderedDict()
        reports_other_failure = OrderedDict()

        s_details_ok = ''
        s_details_error = ''

        self.reports_generated_count = 0
        self.reports_error_count = 0

        for subject, report_summary in self.summary_dict.items():

            s_details = ''
            s_details += '------------------------------------------------------------------------------------\n'
            # s += 'Subject: '+subject+'\n'
            # s+='------------------------------------------------------------------------------------\n'
            s_details += report_summary.summary(detail_level=detail_level)
            s_details += '------------------------------------------------------------------------------------\n\n'

            if report_summary.get_report_generated_flag():
                self.reports_generated_count += 1
                s_details_ok += s_details

            if report_summary.error_flag():
                self.reports_error_count += 1
                s_details_error += s_details

        if self.reports_generated_count:
            s += '\n'
            reports_word = 'reports' if self.reports_generated_count > 1 else 'report'
            s += 'Generated ' + str(self.reports_generated_count) + ' ' + reports_word + '\n'

        if detail_level > 0:
            if self.reports_error_count:
                s += str(
                    self.reports_error_count) + ' reports were not generated due to errors. see detailes below' + '\n'

        s += '\n' + 'Detailed Report Generation Status' + '\n'

        if self.reports_generated_count:
            s += '\n' + '--------------------------NEWLY GENERATED REPORTS-----------------------------------' + '\n'
            s += s_details_ok

        if detail_level > 0:
            if self.reports_error_count:
                s += '\n' + '---------------------------REPORT ERRORS---------------------------------------' + '\n'
                s += s_details_error

        return s

    def get_email_list(self, email_list_file):
        mail_list_path = join(expanduser('~'), '.ram_report', email_list_file)
        mail_list_node = JSONNode.read(mail_list_path)
        print mail_list_node.output()

        email_list = []
        subscribers = mail_list_node['subscribers']
        for subscriber in subscribers:
            print 'Name: ', subscriber['FirstName'], ' ', subscriber['LastName'], ' email: ', subscriber['email']
            email_list.append(subscriber['email'])

        return email_list

    def send_to_single_list(self, subject, msg, email_list, ):
        import base64
        from datetime import date
        import smtplib

        mail_info_path = join(expanduser('~'), '.ram_report', 'mail_info.json')

        mail_info = JSONNode.read(mail_info_path)

        u = mail_info['u']
        p = mail_info['p']
        smtp_server = mail_info['server']
        smtp_port = int(mail_info['port'])

        DATE_FORMAT = "%d/%m/%Y"
        EMAIL_SPACE = ", "
        EMAIL_FROM = "ramdarpaproject@gmail.com"

        print 'u,p,server,port=', (u, p, smtp_server, smtp_port)

        msg['Subject'] = subject

        msg['To'] = EMAIL_SPACE.join(email_list)
        msg['From'] = EMAIL_FROM
        mail = smtplib.SMTP(smtp_server, smtp_port)
        mail.ehlo()

        mail.starttls()

        mail.login(u, base64.b64decode(p))
        mail.sendmail(EMAIL_FROM, email_list, msg.as_string())
        mail.quit()

    def send_email_digest(self, detail_level_list=[0, 1, 2]):

        from email.mime.text import MIMEText
        DATE_FORMAT = "%d/%m/%Y"

        # ------------ regular subscribers --------------

        if 0 in detail_level_list:

            report_summary = self.compose_summary(detail_level=0)

            if self.reports_generated_count:
                email_list = self.get_email_list(email_list_file='mail_list.json')

                msg = MIMEText(report_summary)
                subject = "Daily %s Report Digest for %s" % (self.label, date.today().strftime(DATE_FORMAT))
                self.send_to_single_list(subject=subject, msg=msg, email_list=email_list)

        # ------------ developer subscribers --------------

        if 1 in detail_level_list:
            report_summary_dev = self.compose_summary(detail_level=1)
            if self.reports_generated_count or self.reports_error_count:
                email_list_dev = self.get_email_list(email_list_file='developer_mail_list.json')

                msg_dev = MIMEText(report_summary_dev)

                subject_dev = "Developers' %s Report Digest for %s" % (self.label, date.today().strftime(DATE_FORMAT))
                self.send_to_single_list(subject=subject_dev, msg=msg_dev, email_list=email_list_dev)

        if 2 in detail_level_list:
            report_summary_dev = self.compose_summary(detail_level=2)
            if self.reports_generated_count or self.reports_error_count:
                email_list_dev = self.get_email_list(email_list_file='developer_mail_list.json')
                msg_dev = MIMEText(report_summary_dev)
                subject_dev = "Detailed  Developers' %s Report Digest for %s" % (
                self.label, date.today().strftime(DATE_FORMAT))
                self.send_to_single_list(subject=subject_dev, msg=msg_dev, email_list=email_list_dev)


class ReportSummary(object):
    """
    This object holds report status for a single report
    """

    def __init__(self):
        self.subject = None
        self.report_error_status = None
        self.stacktrace = None
        # self.report_error_status = None
        self.report_status_list = []
        self.changed_resources = None
        self.report_file = None
        self.report_link = None
        self._experiment_name = None

    @property
    def experiment_name(self):
        return self._experiment_name if self._experiment_name is not None else ''

    @experiment_name.setter
    def experiment_name(self,val):
        self._experiment_name  = val

    def task(self):
        try:
            return self.report_status_list[0].task
        except IndexError:

            return ''

    def set_experiment_name(self, exp_name):
        self.experiment_name = exp_name

    def add_report_file(self, file):
        self.report_file = file

    def add_report_link(self, link):
        self.report_link = link

    def add_changed_resources(self, changed_resources):
        self.changed_resources = changed_resources

    def set_subject(self, subject):
        self.subject = subject

    def add_report_status_obj(self, status_obj):
        self.report_status_list.append(status_obj)

    def add_report_error_status(self, error_status):
        self.report_error_status = error_status

    def add_report_error(self, error, stacktrace=None):
        error_rs = ReportStatus(error=error)
        self.report_error_status = error_rs
        if stacktrace is not None:
            self.stacktrace = stacktrace

    def detailed_status(self, detail_level=2):
        s = ''
        if detail_level > 0:
            s += '\nDetailed report level %s\n' % str(detail_level)

        if detail_level > 1:
            for status in self.report_status_list:
                s += '\n'
                s += 'Task: ' + status.task + ' : ' + status.message + '\n'
                if status.file and status.line >= 0:
                    s += 'file: ' + status.file + '-------- line: ' + str(status.line) + ' \n'

        if detail_level == 1:
            s += '\n'
            if len(self.changed_resources):

                s += '------ Changed  Resources -----\n'
                for resource_name, resource in self.changed_resources.items():
                    s += 'Changed resource : ' + resource_name + '\n'
                    s += 'Change type: ' + str(resource) + '\n'
                    s += '\n'
                    # s += str(self.changed_resources) + '\n'

        if detail_level == 2:
            s += '\n'
            if self.stacktrace is not None:
                s += '------ Stack trace -----\n'
                s += self.stacktrace
                s += '\n'

        return s

    def to_json(self):
        out_node = JSONNode()
        subject_node = out_node.add_child_node('subject')
        subject_node['id'] = self.subject

        exp_node = out_node.add_child_node('experiments')

        exp_node = exp_node.add_child_node(self.experiment_name)

        exp_node['report_file'] = self.report_file if self.report_file is not None else ''
        exp_node['report_link'] = self.report_link if self.report_link is not None else ''
        exp_node['error'] = '' if not self.report_error_status else base64.b64encode(str(self.report_error_status.error))
        exp_node['stacktrace'] = '' if not self.report_error_status else base64.b64encode(str(self.stacktrace))

        exp_node['changed_resources'] = []
        changed_res_list = exp_node['changed_resources']
        for resource_name, resource in self.changed_resources.items():
            # res_node = JSONNode()
            # res_node['task'] = resource
            # res_node['resource'] = resource
            # res_node['type'] = change_type
            changed_res_list.append(resource.to_json())

        return out_node


    def get_report_generated_flag(self):
        return bool(self.report_file)

    def error_flag(self):
        return bool(self.report_error_status)

    def summary(self, detail_level=2):
        s = ''
        s += '\nSubject: ' + self.subject + '\n'
        s += '------------------------------------------------------------------------------------\n'

        if self.get_report_generated_flag():
            if self.report_file:
                s += 'Report file (Rhino2): \n' + self.report_file + '\n'

            if self.report_link:
                s += 'Report URL: \n' + self.report_link + '\n'

        # if not self.report_error_status:
        #     s += 'No errors reported\n'


        if self.report_error_status:
            e = self.report_error_status.error
            if e:
                s += '------------ERROR REPORTED:\n'

                if isinstance(e, MissingDataError):
                    s += 'Missing Data Error: '
                elif isinstance(e, MissingExperimentError):
                    s += 'Missing Experiment Error: '
                elif isinstance(e, NumericalError):
                    s += 'Numerical Error: '
                elif isinstance(e, ReportError):
                    s += 'General Report Error: '
                else:
                    s += 'Error: '

                s += str(self.report_error_status.error) + '\n'
                if self.report_error_status.file and self.report_error_status.line >= 0:
                    s += 'File:' + self.report_error_status.file + '---------------------------' + ' line: ' + str(
                        self.report_error_status.line) + '\n'

        s += self.detailed_status(detail_level=detail_level)

        return s
