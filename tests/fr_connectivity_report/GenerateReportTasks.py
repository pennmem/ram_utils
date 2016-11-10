from RamPipeline import *

import TextTemplateUtils
from PlotUtils import PlotData, BarPlotData, PanelPlot
from latex_table import latex_table
import re
import numpy as np
import datetime
from subprocess import call

from ReportUtils import ReportRamTask

# import re
# from collections import namedtuple
# SplitSubjectCode = namedtuple(typename='SplitSubjectCode',field_names=['protocol','id','site','montage'])
# import os
# import shutil


class GenerateTex(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GenerateTex,self).__init__(mark_as_completed)

    def run(self):
        subject = self.pipeline.subject

        # n_sess = self.get_passed_object('NUMBER_OF_SESSIONS')
        # n_bps = self.get_passed_object('NUMBER_OF_ELECTRODES')

        tex_template = 'fr_connectivity_report.tex.tpl'
        tex_file_name = '%s_fr_connectivity_report.tex' % subject

        self.set_file_resources_to_move(tex_file_name, dst='reports')

        connectivity_strength_table = latex_table(self.get_passed_object('connectivity_strength_table'))

        replace_dict = {'<DATE>': datetime.date.today(),
                        '<SUBJECT>': subject.replace('_','\\textunderscore'),
                        '<CONNECTIVITY_STRENGTH_TABLE>': connectivity_strength_table
                        }

        TextTemplateUtils.replace_template(template_file_name=tex_template, out_file_name=tex_file_name, replace_dict=replace_dict)

        self.pass_object('tex_file_name', tex_file_name)


class GenerateReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(GenerateReportPDF,self).__init__(mark_as_completed)

    def run(self):
        output_directory = self.get_path_to_resource_in_workspace('reports')

        texinputs_set_str = r'export TEXINPUTS="' + output_directory + '":$TEXINPUTS;'

        # report_tex_file_names = self.get_passed_object('report_tex_file_names')
        # for f in report_tex_file_names:
        #     # + '/Library/TeX/texbin/pdflatex '\
        #     pdflatex_command_str = texinputs_set_str \
        #                            + 'module load Tex; pdflatex '\
        #                            + ' -output-directory '+output_directory\
        #                            + ' -shell-escape ' \
        #                            + self.get_path_to_resource_in_workspace('reports/'+f)
        #
        #     call([pdflatex_command_str], shell=True)

        combined_report_tex_file_name = self.get_passed_object('tex_file_name')

        pdflatex_command_str = texinputs_set_str \
                               + 'module load Tex; pdflatex '\
                               + ' -output-directory '+output_directory\
                               + ' -shell-escape ' \
                               + self.get_path_to_resource_in_workspace('reports/'+combined_report_tex_file_name)

        call([pdflatex_command_str], shell=True)

        report_core_file_name, ext = splitext(combined_report_tex_file_name)
        report_file = join(output_directory,report_core_file_name+'.pdf')
        self.pass_object('report_file',report_file)




class DeployReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(DeployReportPDF,self).__init__(mark_as_completed)

    def run(self):
        report_file = self.get_passed_object('report_file')
        self.pipeline.deploy_report(report_path=report_file)
