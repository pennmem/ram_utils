from JSONUtils import JSONNode
from os.path import *
import glob
import base64
from collections import defaultdict,OrderedDict
import pandas as pd



class ReportMailer(object):
    def __init__(self):

        self.dir_list = []
        self.task_report_dict = defaultdict(OrderedDict)

    def add_directories(self,*args):
        map(lambda d: self.dir_list.append(d),args)

    def compose_summary(self):
        d = self.dir_list[0]
        for d in self.dir_list:
            self.extract_json_node_from_dir(d)


    def extract_json_node_from_dir(self, d):

        report_files = glob.glob(join(d,'*.json'))
        report_files = sorted(report_files)

        for f in report_files:
            print f
            jn = JSONNode.read(f)
            experiment_name = str(jn['experiments'].keys()[0])
            subject = str(jn['subject']['id'])
            self.task_report_dict[experiment_name][subject] = jn
            print self.task_report_dict[experiment_name].keys()
            # print jn.output()
            # print jn['experiments'].keys()
            # break
            #
            # print jn.output()
            #
        # for experiment_name, o_dict  in self.task_report_dict.iteritems():
        #
        #     for subject, jn in o_dict.iteritems():
        #         print 'subject = ',subject
        #         print 'jn = ',jn

        for experiment_name, node_dict in self.task_report_dict.iteritems():
            self.compose_experiment_summary(node_dict=node_dict)

    def compose_experiment_summary(self, node_dict):

        for subject,jn in node_dict.iteritems():
            experiment_name = str(jn['experiments'].keys()[0])
            break

        s_generated_header = 'Generated the following '+experiment_name+' reports:\n'

        s_generated_body = ''
        generated_reports_num = 0
        for subject, jn in node_dict.iteritems():
            report_link = str(jn['experiments'][experiment_name]['report_link'])
            if report_link:
                s_generated_body += 'Subject: '+ subject + ' : ' + report_link + '\n'
                generated_reports_num +=1

        if not generated_reports_num:
            s_generated_header = ''
        else:
            s_generated_header += 'Total '+str(generated_reports_num)  + ' reports \n'
            s_generated_header += '--------------------------------------------\n'

        s_summary = s_generated_header+s_generated_body
        print s_summary
            # print 'subject = ', subject
            # print 'jn = ', jn


if __name__=='__main__':

    dir_list=['/Volumes/rhino_root/scratch/mswat/automated_reports/FR1_reports/status_output/']

    rm = ReportMailer()
    rm.add_directories(*dir_list)
    rm.compose_summary()

