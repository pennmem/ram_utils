import sys
import time
import numpy as np

from os.path import *

from ramutils.pipeline import RamPipeline
from ps4_pal5_prompt import parse_command_line, Args
from system_3_utils.ram_tasks.CMLParserClosedLoop5 import CMLParserCloseLoop5
from tests.ps4_pal5_biomarker.PAL1EventPreparation import PAL1EventPreparation
from tests.ps4_pal5_biomarker.FREventPreparation import FREventPreparation
from tests.ps4_pal5_biomarker.CombinedEventPreparation import CombinedEventPreparation
from tests.ps4_pal5_biomarker.ComputePowers import ComputePowers
from tests.ps4_pal5_biomarker.MontagePreparation import MontagePreparation
from tests.ps4_pal5_biomarker.ComputeClassifier import ComputeClassifier
from tests.ps4_pal5_biomarker.ComputeClassifier import ComputePAL1Classifier
from tests.ps4_pal5_biomarker.ComputeClassifier import ComputeFullClassifier
from tests.ps4_pal5_biomarker.ComputeEncodingClassifier import ComputeEncodingClassifier
from tests.ps4_pal5_biomarker.LogResults import LogResults
from system_3_utils.ram_tasks.CheckElectrodeConfigurationClosedLoop3 import CheckElectrodeConfigurationClosedLoop3
from tests.ps4_pal5_biomarker.system3.ExperimentConfigGeneratorClosedLoop5_V1 import ExperimentConfigGeneratorClosedLoop5_V1


if sys.platform.startswith('win'):
    prefix = 'd:/'
else:
    prefix = '/'

args_list = []
try:
    args_obj = parse_command_line()
    args_list.append(args_obj)
except:
    import debug_args_obj
    args_list = debug_args_obj.args_list


class StimParams(object):
    def __init__(self, **kwds):
        pass


class Params(object):
    def __init__(self):
        self.version = '3.00'

        self.include_fr1 = True
        self.include_catfr1 = True
        self.include_fr3 = True
        self.include_catfr3 = True

        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.fr1_retrieval_start_time = -0.525
        self.fr1_retrieval_end_time = 0.0
        self.fr1_retrieval_buf = 0.524


        self.pal1_start_time = 0.3
        self.pal1_end_time = 2.00
        self.pal1_buf = 1.2

        self.pal1_retrieval_start_time = -0.625
        self.pal1_retrieval_end_time = -0.1
        self.pal1_retrieval_buf = 0.524

        self.encoding_samples_weight = 7.2
        self.pal_samples_weight = 1.93

        self.recall_period = 5.0

        self.sliding_window_interval = 0.1
        self.sliding_window_start_offset = 0.3

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(6), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200

        self.stim_params = StimParams(
        )


params = Params()


class ReportPipeline(RamPipeline):
    def __init__(self, subject, workspace_dir, mount_point=None, args=None):
        RamPipeline.__init__(self)
        self.subject = subject
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)
        self.args = args_obj


if __name__ == '__main__':

    for args_obj in args_list:
        args_obj.workspace_dir = join(args_obj.workspace_dir, args_obj.experiment, args_obj.subject)
        report_pipeline = ReportPipeline(subject=args_obj.subject,
                                         workspace_dir=join(args_obj.workspace_dir, args_obj.subject),
                                         mount_point=args_obj.mount_point,
                                         args=args_obj)
        report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))
        report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))
        report_pipeline.add_task(FREventPreparation(mark_as_completed=False))
        report_pipeline.add_task(CombinedEventPreparation(mark_as_completed=False))
        report_pipeline.add_task(CheckElectrodeConfigurationClosedLoop3(params=params, mark_as_completed=False))
        report_pipeline.add_task(ComputePowers(params=params, mark_as_completed=(True & args_obj.allow_fast_rerun)))
        report_pipeline.add_task(ComputeEncodingClassifier(params=params, mark_as_completed=False))
        report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=False))
        report_pipeline.add_task(ComputePAL1Classifier(params=params, mark_as_completed=False))
        report_pipeline.add_task(ExperimentConfigGeneratorClosedLoop5_V1(params=params, mark_as_completed=False))
        report_pipeline.add_task(ComputeFullClassifier(params=params, mark_as_completed=(True & args_obj.allow_fast_rerun)))
        report_pipeline.execute_pipeline()



