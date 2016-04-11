from JSONUtils import JSONNode
from os.path import *
import glob
import base64
from collections import OrderedDict
import pandas as pd



class ReportMailer(object):
    def __init__(self):

        self.dir_list = []
        # self.task_report_dict = OrderedDict()

    def add_directories(self,*args):
        map(lambda d: self.dir_list.append(d),args)

    def compose_summary(self):
        d = self.dir_list[0]
        for d in self.dir_list:
            self.process_dir(d)


    def process_dir(self,d):

        report_files = glob.glob(join(d,'*.json'))
        report_files = sorted(report_files)

        for f in report_files:
            print f
            jn = JSONNode.read(f)
            # print jn.output()
            # print jn['experiments'].keys()
            # break
            #
            # print jn.output()
            #



if __name__=='__main__':

    dir_list=['/Volumes/rhino_root/scratch/mswat/automated_reports/FR1_reports/status_output/']

    rm = ReportMailer()
    rm.add_directories(*dir_list)
    rm.compose_summary()

