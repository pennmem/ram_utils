import re
from collections import namedtuple
import os
import shutil
from os.path import *

SplitSubjectCode = namedtuple(typename='SplitSubjectCode',field_names=['protocol','id','site','montage'])

class ReportDeployer(object):
    def __init__(self, pipeline=None):
        self.pipeline=pipeline

        self.protocol = 'R1'
        self.convert_subject_code_regex = re.compile('('+self.protocol+')'+'([0-9]*)([a-zA-Z]{1,1})([\S]*)')

    def split_subject_code(self,subject_code):
        match = re.match(self.convert_subject_code_regex,subject_code)
        if match:
            groups = match.groups()

            ssc = SplitSubjectCode(protocol=groups[0], id=groups[1],site=groups[2],montage=groups[3])
            return ssc
        return None

    def add_report_file(self,file):
        self.pipeline.report_summary.add_report_file(file=file)


    def add_report_link(self,link):
        self.pipeline.report_summary.add_report_link(link=link)


    def deploy_report(self, report_path, classifier_experiment=None,suffix=None):
        subject = self.pipeline.subject

        ssc = self.split_subject_code(subject)

        report_basename = basename(report_path)
        # report_base_dir = join('protocols',ssc.protocol.lower(),'subjects',str(ssc.id)+ssc.montage,'reports')
        if suffix is None:
            report_base_dir = join('protocols',ssc.protocol.lower(),'subjects',str(ssc.id),'reports')
        else:
            report_base_dir = join('scratch','RAM_maint',suffix,'subjects',str(ssc.id),'reports')

        report_dir = join(self.pipeline.mount_point,report_base_dir)


        if not isdir(report_dir):
            try:

                os.makedirs(report_dir)
            except OSError:

                return

        standard_report_basename = \
            (subject+'_'+self.pipeline.experiment+'_report.pdf')\
            if classifier_experiment is None else \
            (subject+'_'+self.pipeline.experiment+'_'+classifier_experiment+'_report.pdf')

        standard_report_path = join(report_dir,standard_report_basename)


        #  using copyfile is the right solution when copying files
        #  see http://stackoverflow.com/questions/11835833/why-would-shutil-copy-raise-a-permission-exception-when-cp-doesnt
        shutil.copyfile(report_path,standard_report_path)
        # shutil.copy(report_path,standard_report_path)




        self.add_report_file(file=standard_report_path)

        standard_report_link = join(self.pipeline.report_site_URL, report_base_dir, standard_report_basename)
        self.add_report_link(link=standard_report_link)

