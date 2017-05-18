DEBUG = True

from os.path import *

from pal5_prompt import parse_command_line, Args

from system_3_utils.ram_tasks.CMLParserClosedLoop5 import CMLParserCloseLoop5
import sys
import time

if sys.platform.startswith('win'):

    prefix = 'd:/'

else:

    prefix = '/'

try:
    args_obj = parse_command_line()
except:
    args_list = []

    # args_obj = Args()
    #
    # args_obj.subject = 'R1250N'
    # args_obj.anodes = ['PG10', 'PG11']
    # args_obj.cathodes = ['PG11', 'PG12']
    # args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv' % args_obj.subject)
    # args_obj.experiment = 'PS4_CatFR5'
    # args_obj.min_amplitudes = [0.25, 0.25]
    # args_obj.max_amplitudes = [1.0, 1.0]
    # args_obj.mount_point = prefix
    # args_obj.pulse_frequency = 200
    # args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)
    # args_obj.allow_fast_rerun = True
    #
    # args_list.append(args_obj)

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
    # args_obj.allow_fast_rerun = True
    #
    # args_list.append(args_obj)
    #
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
    #
    # args_list.append(args_obj)

    args_obj = Args()

    args_obj.subject = 'R1002P'
    args_obj.anodes = ['LPF1', 'LPF3']
    args_obj.cathodes = ['LPF2','LPF4']
    args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
    args_obj.experiment = 'PS4_CatFR5'
    args_obj.min_amplitudes = [0.25,0.25]
    args_obj.max_amplitudes = [1.0,1.0]
    args_obj.mount_point = prefix
    args_obj.pulse_frequency = 200
    args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)
    args_obj.allow_fast_rerun = True

    args_list.append(args_obj)
    #
    # args_obj = Args()
    #
    # args_obj.subject = 'R1065J'
    # args_obj.anodes = ['LS1', 'LS3']
    # args_obj.cathodes = ['LS2', 'LS4']
    # args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
    # args_obj.experiment = 'PS4_PAL5'
    # args_obj.min_amplitudes = [0.25,0.25]
    # args_obj.max_amplitudes = [1.0,1.0]
    # args_obj.mount_point = prefix
    # args_obj.pulse_frequency = 200
    # args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)
    #
    # args_list.append(args_obj)

    #
    # # messed up localization
    # # args_obj = Args()
    # #
    # # args_obj.subject = 'R1118N'
    # # args_obj.anodes = ['G11', 'G13']
    # # args_obj.cathodes = ['G12', 'G14']
    # # args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
    # # args_obj.experiment = 'PS4_PAL5'
    # # args_obj.min_amplitudes = [0.25,0.25]
    # # args_obj.max_amplitudes = [1.0,1.0]
    # # args_obj.mount_point = prefix
    # # args_obj.pulse_frequency = 200
    # # args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)
    #
    # # messed up data
    # args_obj = Args()
    #
    # args_obj.subject = 'R1121M'
    # args_obj.anodes = ['RFG1', 'RFG3']
    # args_obj.cathodes = ['RFG2', 'RFG4']
    # args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
    # args_obj.experiment = 'PS4_PAL5'
    # args_obj.min_amplitudes = [0.25,0.25]
    # args_obj.max_amplitudes = [1.0,1.0]
    # args_obj.mount_point = prefix
    # args_obj.pulse_frequency = 200
    # args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)
    #
    # args_list.append(args_obj)

    #
    # args_obj = Args()
    #
    # args_obj.subject = 'R1162N'
    # args_obj.anodes = ['G11', 'G13']
    # args_obj.cathodes = ['G12', 'G14']
    # args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
    # args_obj.experiment = 'PS4_PAL5'
    # args_obj.min_amplitudes = [0.25,0.25]
    # args_obj.max_amplitudes = [1.0,1.0]
    # args_obj.mount_point = prefix
    # args_obj.pulse_frequency = 200
    # args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)
    #
    # args_list.append(args_obj)
    #
    # args_obj = Args()
    #
    # args_obj.subject = 'R1175N'
    # args_obj.anodes = ['LAT1', 'LAT3']
    # args_obj.cathodes = ['LAT2','LAT4']
    # args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
    # args_obj.experiment = 'PS4_PAL5'
    # args_obj.min_amplitudes = [0.25,0.25]
    # args_obj.max_amplitudes = [1.0,1.0]
    # args_obj.mount_point = prefix
    # args_obj.pulse_frequency = 200
    # args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)
    #
    # args_list.append(args_obj)
    #
    #
    # args_obj = Args()
    #
    # args_obj.subject = 'R1212P'
    # args_obj.anodes = ['LXB1', 'LXB3']
    # args_obj.cathodes = ['LXB2','LXB4']
    # args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
    # args_obj.experiment = 'PS4_PAL5'
    # args_obj.min_amplitudes = [0.25,0.25]
    # args_obj.max_amplitudes = [1.0,1.0]
    # args_obj.mount_point = prefix
    # args_obj.pulse_frequency = 200
    # args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)
    #
    # args_list.append(args_obj)
    #
    # args_obj = Args()
    #
    # args_obj.subject = 'R1232N'
    # args_obj.anodes = ['LAT1', 'LAT3']
    # args_obj.cathodes = ['LAT2','LAT4']
    # args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
    # args_obj.experiment = 'PS4_PAL5'
    # args_obj.min_amplitudes = [0.25,0.25]
    # args_obj.max_amplitudes = [1.0,1.0]
    # args_obj.mount_point = prefix
    # args_obj.pulse_frequency = 200
    # args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)
    #
    # args_list.append(args_obj)

# ------------------------------- end of processing command line

from RamPipeline import RamPipeline

from tests.pal5_fr5_biomarker.PAL1EventPreparation import PAL1EventPreparation

from tests.pal5_fr5_biomarker.FREventPreparation import FREventPreparation

from tests.pal5_fr5_biomarker.CombinedEventPreparation import CombinedEventPreparation

from tests.pal5_fr5_biomarker.ComputePowers import ComputePowers

from tests.pal5_fr5_biomarker.MontagePreparation import MontagePreparation

from system_3_utils.ram_tasks.CheckElectrodeConfigurationClosedLoop3 import CheckElectrodeConfigurationClosedLoop3

from tests.pal5_fr5_biomarker.ComputeClassifier import ComputeClassifier

from tests.pal5_fr5_biomarker.ComputeClassifier import ComputeFullClassifier

from tests.pal5_fr5_biomarker.ComputeEncodingClassifier import ComputeEncodingClassifier

from tests.pal5_fr5_biomarker.LogResults import LogResults

from tests.pal5_biomarker.system3.ExperimentConfigGeneratorClosedLoop5_V1 import ExperimentConfigGeneratorClosedLoop5_V1

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

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.fr1_retrieval_start_time = -0.525
        self.fr1_retrieval_end_time = 0.0
        self.fr1_retrieval_buf = 0.524


        self.pal1_start_time = 0.3
        self.pal1_end_time = 2.00
        self.pal1_buf = 1.2

        # original code
        self.pal1_retrieval_start_time = -0.625
        self.pal1_retrieval_end_time = -0.1
        self.pal1_retrieval_buf = 0.524


        # # todo remove in the production code
        # self.pal1_retrieval_start_time = -0.600
        # self.pal1_retrieval_end_time = -0.1
        # self.pal1_retrieval_buf = 0.499


        # self.retrieval_samples_weight = 2.5
        # self.encoding_samples_weight = 2.5
        self.encoding_samples_weight = 7.2
        self.pal_samples_weight = 1.93

        self.recall_period = 5.0

        self.sliding_window_interval = 0.1
        self.sliding_window_start_offset = 0.3

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(6), np.log10(180), 8)
        # self.freqs = np.logspace(np.log10(3), np.log10(180), 8)  # TODO - remove it from production code

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4
        # self.C = 0.048 # TODO - remove it from production code - original PAL5 value


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
        self.args = args_obj


if __name__ == '__main__':
    # report_pipeline = ReportPipeline(subject=args.subject,
    #                                  workspace_dir=join(args.workspace_dir, args.subject), mount_point=args.mount_point,
    #                                  args=args)


    log_filename = join('D:/PAL5', 'PAL5_' + time.strftime('%Y_%m_%d_%H_%M_%S')+'.csv')

    for args_obj in args_list:
        report_pipeline = ReportPipeline(subject=args_obj.subject,
                                         workspace_dir=join(args_obj.workspace_dir, args_obj.subject),
                                         mount_point=args_obj.mount_point,
                                         args=args_obj)

        report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

        report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))

        report_pipeline.add_task(FREventPreparation(mark_as_completed=False))

        report_pipeline.add_task(CombinedEventPreparation(mark_as_completed=False))

        report_pipeline.add_task(CheckElectrodeConfigurationClosedLoop3(params=params, mark_as_completed=False))
        #
        report_pipeline.add_task(ComputePowers(params=params, mark_as_completed=(True & args_obj.allow_fast_rerun)))

        # report_pipeline.add_task(ComputeEncodingClassifier(params=params, mark_as_completed=False))
        #
        report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=False))

        report_pipeline.add_task(ExperimentConfigGeneratorClosedLoop5_V1(params=params, mark_as_completed=False))

        #
        # report_pipeline.add_task(LogResults(params=params, mark_as_completed=False, log_filename=log_filename))
        #
        report_pipeline.add_task(ComputeFullClassifier(params=params, mark_as_completed=(True & args_obj.allow_fast_rerun)))

        # starts processing pipeline
        report_pipeline.execute_pipeline()



#
# if __name__ == '__main__':
#     # report_pipeline = ReportPipeline(subject=args.subject,
#     #                                  workspace_dir=join(args.workspace_dir, args.subject), mount_point=args.mount_point,
#     #                                  args=args)
#
#     report_pipeline = ReportPipeline(subject=args_obj.subject,
#                                      workspace_dir=join(args_obj.workspace_dir, args_obj.subject), mount_point=args_obj.mount_point,
#                                      args=args_obj)
#
#
#     report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))
#
#     report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))
#     #
#     report_pipeline.add_task(CheckElectrodeConfigurationClosedLoop3(params=params, mark_as_completed=False))
#     #
#     report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))
#
#     report_pipeline.add_task(ComputeEncodingClassifier(params=params, mark_as_completed=False))
#
#     report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=False))
#
#     report_pipeline.add_task(ComputeFullClassifier(params=params, mark_as_completed=True))
#
#
#
#     # report_pipeline.add_task(ComputeBiomarkerThreshold(params=params, mark_as_completed=False))
#     #
#     #
#     #
#     # #
#     # report_pipeline.add_task(ExperimentConfigGeneratorClosedLoop5(params=params, mark_as_completed=False))
#     #
#     # starts processing pipeline
#     report_pipeline.execute_pipeline()