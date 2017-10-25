
DEBUG = True

from os.path import *

from pal5_prompt import parse_command_line, Args

from system_3_utils.ram_tasks.CMLParserClosedLoop5 import CMLParserCloseLoop5
import sys

if sys.platform.startswith('win'):

    prefix = 'd:/'

else:

    prefix = '/'


try:
    args_obj = parse_command_line()
except:


    args_obj = Args()

    args_obj.subject = 'R1250N'
    args_obj.anodes = ['PG10', 'PG11']
    args_obj.cathodes = ['PG11','PG12']
    args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
    args_obj.experiment = 'PS4_PAL5'
    args_obj.min_amplitudes = [0.25,0.25]
    args_obj.max_amplitudes = [1.0,1.0]
    args_obj.mount_point = prefix
    args_obj.pulse_frequency = 200
    args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)


    # args_obj = Args()
    #
    # args_obj.subject = 'R1095N'
    # args_obj.anodes = ['RTT1', 'RTT3']
    # args_obj.cathodes = ['RTT2', 'RTT4']
    # args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
    # args_obj.experiment = 'PS4_PAL5'
    # args_obj.min_amplitudes = [0.25,0.25]
    # args_obj.max_amplitudes = [1.0,1.0]
    # args_obj.mount_point = prefix
    # args_obj.pulse_frequency = 200
    # args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)


    # args_obj = Args()
    #
    # args_obj.subject = 'R1284N'
    # args_obj.anodes = ['LMD1', 'LMD3']
    # args_obj.cathodes = ['LMD2','LMD4']
    # args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
    # args_obj.experiment = 'PS4_PAL5'
    # args_obj.min_amplitudes = [0.25,0.25]
    # args_obj.max_amplitudes = [1.0,1.0]
    # args_obj.mount_point = prefix
    # args_obj.pulse_frequency = 200
    # args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)


# ------------------------------- end of processing command line

from RamPipeline import RamPipeline

from reports.pal5_biomarker.PAL1EventPreparation import PAL1EventPreparation

from reports.pal5_biomarker.ComputePAL1Powers import ComputePAL1Powers

from reports.pal5_biomarker.MontagePreparation import MontagePreparation

from system_3_utils.ram_tasks.CheckElectrodeConfigurationClosedLoop3 import CheckElectrodeConfigurationClosedLoop3

from reports.pal5_biomarker.ComputeClassifier import ComputeClassifier

from reports.pal5_biomarker.ComputeClassifier import ComputeFullClassifier

from reports.pal5_biomarker.ComputeEncodingClassifier import ComputeEncodingClassifier

from reports.pal5_biomarker.ComputeBiomarkerThreshold import ComputeBiomarkerThreshold

from reports.pal5_biomarker.system3.ExperimentConfigGeneratorClosedLoop5 import ExperimentConfigGeneratorClosedLoop5

import numpy as np


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

        # self.retrieval_samples_weight = 2.5
        # self.encoding_samples_weight = 2.5
        self.encoding_samples_weight = 1.0

        self.recall_period = 5.0

        self.sliding_window_interval = 0.1
        self.sliding_window_start_offset = 0.3

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(6), np.log10(180), 8)
        # self.freqs = np.logspace(np.log10(3), np.log10(180), 8)  # TODO - remove it from production code

        self.log_powers = True

        self.penalty_type = 'l2'
        # self.C = 7.2e-4  # TODO - remove it from production code
        self.C = 0.048

        # self.n_perm = 200
        self.n_perm = 10  # TODO - remove it from production code

        self.stim_params = StimParams(
        )


params = Params()


class ReportPipeline(RamPipeline):
    def __init__(self, subject, workspace_dir, mount_point=None, args=None):
        RamPipeline.__init__(self)
        self.subject = subject
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)
        # self.args = args # todo original code
        self.args = args_obj


if __name__ == '__main__':
    # report_pipeline = ReportPipeline(subject=args.subject,
    #                                  workspace_dir=join(args.workspace_dir, args.subject), mount_point=args.mount_point,
    #                                  args=args)

    report_pipeline = ReportPipeline(subject=args_obj.subject,
                                     workspace_dir=join(args_obj.workspace_dir, args_obj.subject), mount_point=args_obj.mount_point,
                                     args=args_obj)


    report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))

    report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))
    #
    report_pipeline.add_task(CheckElectrodeConfigurationClosedLoop3(params=params, mark_as_completed=False))
    #
    report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeEncodingClassifier(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputeFullClassifier(params=params, mark_as_completed=True))



    # report_pipeline.add_task(ComputeBiomarkerThreshold(params=params, mark_as_completed=False))
    #
    #
    #
    # #
    # report_pipeline.add_task(ExperimentConfigGeneratorClosedLoop5(params=params, mark_as_completed=False))
    #
    # starts processing pipeline
    report_pipeline.execute_pipeline()
