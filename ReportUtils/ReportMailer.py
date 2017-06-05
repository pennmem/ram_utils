from JSONUtils import JSONNode
from os.path import *
import glob
import base64
from collections import defaultdict, OrderedDict
from datetime import date
import pandas as pd
import shutil
import os

from email.mime.text import MIMEText


class ReportMailer(object):
    def __init__(self):

        self.dir_list = []
        # self.task_report_dict = defaultdict(OrderedDict)
        self.task_report_dict = OrderedDict()

    def add_directories(self, *args):
        map(lambda d: self.dir_list.append(d), args)

    def compose_summary(self, detail_level=0):
        # d = self.dir_list[0]
        for d in self.dir_list:
            self.extract_json_node_from_dir(d)

        summary = ''

        compose_fcn = None

        if detail_level == 0:
            compose_fcn = self.compose_experiment_summary_detail_0
        elif detail_level == 1:
            compose_fcn = self.compose_experiment_summary_detail_1

        if not compose_fcn:
            return ''

        for experiment_name, node_dict in self.task_report_dict.iteritems():
            summary += compose_fcn(node_dict=node_dict)

        return summary

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

    def mail_report_summary(self, detail_level_list=[0, 1, 2]):

        # summary_0, summary_1 = self.compose_summary()
        DATE_FORMAT = "%d/%m/%Y"

        # ------------ regular subscribers --------------

        if 0 in detail_level_list:
            report_summary = self.compose_summary(detail_level=0)

            # will only send level 0 mail if the reports were generated
            if report_summary.strip() != '':

                # if self.reports_generated_count:
                email_list = self.get_email_list(email_list_file='mail_list.json')

                msg = MIMEText(report_summary)
                subject = "Daily Report Digest for %s" % (date.today().strftime(DATE_FORMAT))
                self.send_to_single_list(subject=subject, msg=msg, email_list=email_list)

        if 1 in detail_level_list:
            report_summary = self.compose_summary(detail_level=1)

            email_list_dev = self.get_email_list(email_list_file='developer_mail_list.json')

            msg_dev = MIMEText(report_summary)

            subject_dev = "Developers' Report Digest for %s" % (date.today().strftime(DATE_FORMAT))
            self.send_to_single_list(subject=subject_dev, msg=msg_dev, email_list=email_list_dev)

    def output_error_log(self,error_file):

        try:
            os.makedirs(dirname(error_file))
        except OSError:
            pass


        report_summary = self.compose_summary(detail_level=1)

        print report_summary

        msg_dev = MIMEText(report_summary)

        # subject_dev = "Developers' Report Digest for %s" % (date.today().strftime(DATE_FORMAT))
        # self.send_to_single_list(subject=subject_dev, msg=msg_dev, email_list=email_list_dev)


        f = open(error_file,'w')
        f.write('%s'%report_summary)
        f.close()


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

    def extract_json_node_from_dir(self, d):

        report_files = glob.glob(join(d, '*.json'))
        report_files = sorted(report_files)

        for f in report_files:
            print f
            jn = JSONNode.read(f)
            experiment_name = str(jn['experiments'].keys()[0])
            subject = str(jn['subject']['id'])
            try:
                self.task_report_dict[experiment_name][subject]=jn
            except KeyError:
                self.task_report_dict[experiment_name]=OrderedDict()
                self.task_report_dict[experiment_name][subject]=jn

            # self.task_report_dict[experiment_name][subject] = jn
            print self.task_report_dict[experiment_name].keys()

    def compose_experiment_summary_detail_0(self, node_dict):

        for subject, jn in node_dict.iteritems():
            experiment_name = str(jn['experiments'].keys()[0])
            break

        s_generated_header = '\nGenerated the following ' + experiment_name + ' reports:\n'

        s_generated_body = ''
        generated_reports_num = 0
        error_report_num = 0

        for subject, jn in node_dict.iteritems():
            report_link = str(jn['experiments'][experiment_name]['report_link'])

            if report_link:
                s_generated_body += 'Subject: ' + subject + ' : ' + report_link + '\n'
                generated_reports_num += 1

        if not generated_reports_num:
            s_generated_header = ''
        else:
            s_generated_header += 'Total ' + str(generated_reports_num) + ' reports \n'
            s_generated_header += '--------------------------------------------\n'

        s_summary = s_generated_header + s_generated_body
        print s_summary

        return s_summary

    def compose_experiment_summary_detail_1(self, node_dict):
        for subject, jn in node_dict.iteritems():
            experiment_name = str(jn['experiments'].keys()[0])
            break

        s_error_header = 'Report Generation Errors For %s:\n' % experiment_name
        s_error_body = ''

        s_generated_header = '\nGenerated the following ' + experiment_name + ' reports:\n'

        s_generated_body = ''
        generated_reports_num = 0
        error_report_num = 0

        for subject, jn in node_dict.iteritems():
            report_link = str(jn['experiments'][experiment_name]['report_link'])
            report_error = str(base64.b64decode(jn['experiments'][experiment_name]['error']))
            report_stacktrace = str(base64.b64decode(jn['experiments'][experiment_name]['stacktrace']))

            if report_link:
                s_generated_body += 'Subject: ' + subject + ' : ' + report_link + '\n'
                generated_reports_num += 1

            if report_error:
                s_error_body += '\nSubject: ' + subject + '\n'
                s_error_body += 'Error: \n' + report_error + '\n'
                s_error_body += 'Stacktrace: \n' + report_stacktrace + '\n'
                error_report_num += 1

        if not generated_reports_num:
            s_generated_header = ''
        else:
            s_generated_header += 'Total ' + str(generated_reports_num) + ' reports\n'
            s_generated_header += '\n--------------------------------------------\n'

        if not error_report_num:
            s_error_header = ''
        else:
            s_error_header += 'Total ' + str(error_report_num) + ' errors\n'
            s_error_header += '\n--------------------------------------------\n'

        s_summary = s_error_header + s_error_body
        # s_summary += '\n-------------------------------------------\n'
        # s_summary += s_generated_header + s_generated_body
        print s_summary

        return s_summary


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Report Mailer')
    # parser.add_argument('--status-output-dir', required=False, action='append')
    parser.add_argument('--status-output-dirs', nargs='+')
    parser.add_argument('--error-log-file',dest='error_log_file', action='store',required=False)

    args = parser.parse_args()
    dir_list = args.status_output_dirs
    print dir_list
    print 'error log file=',args.error_log_file
    rm = ReportMailer()
    rm.add_directories(*dir_list)
    # rm.compose_summary(detail_level=0)
    rm.mail_report_summary(detail_level_list=[0])

    if args.error_log_file:
        rm.output_error_log(args.error_log_file)


# if __name__ == '__main__':
#     dir_list = ['/Volumes/rhino_root/scratch/mswat/automated_reports/FR1_reports/status_output/']
#
#     rm = ReportMailer()
#     rm.add_directories(*dir_list)
#     # rm.compose_summary(detail_level=0)
#     rm.mail_report_summary(detail_level_list=[0, 1])
