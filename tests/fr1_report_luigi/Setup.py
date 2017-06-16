import luigi
from RamTaskL import RamTaskL
from os.path import *

from Params import Params
from Pipeline import Pipeline


# pipeline = Pipeline(Params())

class Setup(RamTaskL):
    # subject = luigi.Parameter(default='')

    pipeline = luigi.Parameter(default=None)
    # workspace_dir = luigi.Parameter(default=join(expanduser('~'), 'scratch'))

    def define_outputs(self):
        self.add_file_resource('empty_file')

    def run_impl(self):
        print 'Hello'

        # self.pipeline.subject = self.subject
        # self.pipeline.workspace_dir = join(self.workspace_dir, self.pipeline.subject)

        self.clear_output_file('empty_file')



"""
to run:
    python -m luigi --module fr1_report_luigi.Setup Setup --local-scheduler --subject=R1275D --workspace-dir=d:\sc_lui

"""



# pipeline = Pipeline(Params())
#
# class Setup(RamTaskL):
#
#     subject = luigi.Parameter(default='')
#
#     pipeline = pipeline
#     workspace_dir = luigi.Parameter(default=join(expanduser('~'),'scratch'))
#
#     def define_outputs(self):
#
#         self.add_file_resource('empty_file')
#
#     def run_impl(self):
#
#         print 'Hello'
#
#         self.pipeline.subject = self.subject
#         self.pipeline.workspace_dir = join(self.workspace_dir,self.pipeline.subject)
#
#         self.clear_output_file('empty_file')
