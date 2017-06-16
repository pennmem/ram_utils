import luigi

# # command line example:
# # python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT
#
# # python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT
#
# # python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
# import sys

#

from Params import Params
from Pipeline import Pipeline
from RamTaskL import RamTaskL
from FR1EventPreparation import FR1EventPreparation

from RepetitionRatio import RepetitionRatio
from MontagePreparation import MontagePreparation
from ComputeFR1Powers import ComputeFR1Powers
from ComputeFR1HFPowers import ComputeFR1HFPowers
from ComputeTTest import ComputeTTest
from ComputeClassifier import ComputeClassifier
from ComputeClassifier import ComputeJointClassifier
from ComposeSessionSummary import ComposeSessionSummary
from GenerateReportTasks import GeneratePlots
from GenerateReportTasks import GenerateTex
from GenerateReportTasks import GenerateReportPDF

from Setup import Setup
from FR1EventPreparation_1 import FR1EventPreparation_1

params = Params()
pipeline = Pipeline(params)


class EventCheck(RamTaskL):
    """
    Dummy task
    """
    def define_outputs(self):
        self.add_file_resource('mmm')

    def requires(self):
        yield Setup(pipeline=self.pipeline)
        yield FR1EventPreparation(pipeline=self.pipeline)

    def run_impl(self):
        print 'GOT HERE'
        events = self.get_passed_object(self.pipeline.task + '_all_events')
        self.clear_output_file('mmm')
        print events


# if __name__ == '__main__':
#     luigi.build([FR1EventPreparation(pipeline=pipeline, mark_as_completed=True),
#                  EventCheck(pipeline=pipeline),
#                  MontagePreparation(pipeline=pipeline, mark_as_completed=True),
#                  RepetitionRatio(pipeline=pipeline),
#                  ComputeFR1Powers(pipeline=pipeline, mark_as_completed=True),
#                  ComputeFR1HFPowers(pipeline=pipeline, mark_as_completed=True),
#                  ComputeTTest(pipeline=pipeline, mark_as_completed=False),
#                  ComputeClassifier(pipeline=pipeline, mark_as_completed=True),
#                  ComputeJointClassifier(pipeline=pipeline, mark_as_completed=True),
#                  ComposeSessionSummary(pipeline=pipeline, mark_as_completed=False),
#                  GeneratePlots(pipeline=pipeline, mark_as_completed=False),
#                  GenerateTex(pipeline=pipeline, mark_as_completed=False),
#                  # GenerateReportPDF(pipeline=pipeline,mark_as_completed=False)
#                  ],
#                 local_scheduler=True)


if __name__ == '__main__':

    # luigi.run(main_task_cls=FR1EventPreparation_1, cmdline_args=['--subject','R1065J'])



    luigi.build([
                    Setup(pipeline=pipeline, mark_as_completed=True,workspace_dir='d:\sc_lui', subject='R1065J'),
                    FR1EventPreparation_1(pipeline=pipeline, mark_as_completed=True)
                ],
                local_scheduler=True,
                )

    # luigi.build([
    # # FR1EventPreparation(pipeline=pipeline, mark_as_completed=True),
    # #              EventCheck(pipeline=pipeline),
    # #              MontagePreparation(pipeline=pipeline, mark_as_completed=True),
    # #              RepetitionRatio(pipeline=pipeline),
    # #              ComputeFR1Powers(pipeline=pipeline, mark_as_completed=True),
    # #              ComputeFR1HFPowers(pipeline=pipeline, mark_as_completed=True),
    # #              ComputeTTest(pipeline=pipeline, mark_as_completed=False),
    # #              ComputeClassifier(pipeline=pipeline, mark_as_completed=True),
    # #              ComputeJointClassifier(pipeline=pipeline, mark_as_completed=True),
    # #              ComposeSessionSummary(pipeline=pipeline, mark_as_completed=False),
    # #              GeneratePlots(pipeline=pipeline, mark_as_completed=False),
    # #              GenerateTex(pipeline=pipeline, mark_as_completed=False),
    #              Setup(pipeline=pipeline, mark_as_completed=False),
    #             FR1EventPreparation_1(pipeline=pipeline, mark_as_completed=True)
    #              ],
    #             local_scheduler=True)
