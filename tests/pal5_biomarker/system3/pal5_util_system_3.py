import time
from os.path import *
import numpy as np

from pal5_prompt import parse_command_line, Args

args_list = []

try:
    # raise KeyboardInterrupt
    args_obj = parse_command_line()
except KeyboardInterrupt:

    args_obj = Args()

    args_obj.subject = 'R1333N'
    args_obj.anodes = ['LPLT5', 'LAHD21']
    args_obj.cathodes = ['LPLT6', 'LAHD22']
    args_obj.electrode_config_file = '/Users/depalati/mnt/rhino/scratch/system3_configs/ODIN_configs/R1333N/R1333N_28AUG2017L0M0STIM.csv'
    args_obj.experiment = 'PAL5'
    args_obj.min_amplitudes = [0.25, 0.25]
    args_obj.max_amplitudes = [1.0, 1.0]
    args_obj.mount_point = '/Users/depalati/mnt/rhino'
    args_obj.pulse_frequency = 200
    args_obj.workspace_dir = '/Users/depalati/mnt/rhino/scratch/depalati/configs'

    args_list.append(args_obj)

from ramutils.pipeline import RamPipeline
from tests.pal5_biomarker.PAL1EventPreparation import PAL1EventPreparation
from tests.pal5_biomarker.ComputePAL1Powers import ComputePAL1Powers
from tests.pal5_biomarker.MontagePreparation import MontagePreparation
from system_3_utils.ram_tasks.CheckElectrodeConfigurationClosedLoop3 import CheckElectrodeConfigurationClosedLoop3
from tests.pal5_biomarker.ComputeClassifier import ComputeClassifier
from tests.pal5_biomarker.ComputeClassifier import ComputeFullClassifier
from tests.pal5_biomarker.ComputeEncodingClassifier import ComputeEncodingClassifier
from tests.pal5_biomarker.LogResults import LogResults
from tests.pal5_biomarker.ComputeBiomarkerThreshold import ComputeBiomarkerThreshold
from tests.pal5_biomarker.system3.ExperimentConfigGeneratorClosedLoop5 import ExperimentConfigGeneratorClosedLoop5


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

        self.pal1_start_time = 0.3
        self.pal1_end_time = 2.00
        self.pal1_buf = 1.2

        self.pal1_retrieval_start_time = -0.625
        self.pal1_retrieval_end_time = -0.1
        self.pal1_retrieval_buf = 0.524

        self.encoding_samples_weight = 1.0

        self.recall_period = 5.0

        self.sliding_window_interval = 0.1
        self.sliding_window_start_offset = 0.3

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(6), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 0.048

        self.n_perm = 200

        self.stim_params = StimParams()


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
        log_filename = join(args_obj.workspace_dir,args_obj.subject,time.strftime('%Y_%m_%d')+'.csv')
        report_pipeline = ReportPipeline(subject=args_obj.subject,
                                         workspace_dir=join(args_obj.workspace_dir,
                                            '{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                            args_obj.subject,args_obj.experiment,
                                            args_obj.anodes[0],args_obj.cathodes[0],args_obj.max_amplitudes[0],
                                            args_obj.anodes[1],args_obj.cathodes[1],args_obj.max_amplitudes[1]
                                            )
                                                            ),
                                         mount_point=args_obj.mount_point,
                                         args=args_obj)

        report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))
        report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))
        report_pipeline.add_task(CheckElectrodeConfigurationClosedLoop3(params=params, mark_as_completed=False))
        report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=False))
        report_pipeline.add_task(ComputeEncodingClassifier(params=params, mark_as_completed=False))
        report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=False))
        report_pipeline.add_task(ComputeBiomarkerThreshold(params=params, mark_as_completed=False))
        report_pipeline.add_task(ComputeFullClassifier(params=params, mark_as_completed=False))
        report_pipeline.add_task(ExperimentConfigGeneratorClosedLoop5(params=params, mark_as_completed=False))
        report_pipeline.execute_pipeline()
