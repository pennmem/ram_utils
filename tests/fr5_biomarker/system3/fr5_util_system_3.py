# command line example:
# python fr3_util_system_3.py --workspace-dir=/scratch/busygin/FR3_biomarkers --subject=R1145J_1 --n-channels=128 --anode=RD2 --anode-num=34 --cathode=RD3 --cathode-num=35 --pulse-frequency=200 --pulse-duration=500 --target-amplitude=1000

print "ATTN: Wavelet params and interval length are hardcoded!! To change them, recompile"
print "Windows binaries from https://github.com/busygin/morlet_for_sys2_biomarker"
print "See https://github.com/busygin/morlet_for_sys2_biomarker/blob/master/README for detail."

from os.path import *
import sys
from system_3_utils.ram_tasks.CMLParserClosedLoop3 import CMLParserCloseLoop3

cml_parser = CMLParserCloseLoop3(arg_count_threshold=1)

subject = 'R1230J'
cml_parser.arg('--workspace-dir', '/home1/leond/fr5_config')
cml_parser.arg('--experiment', 'FR5')
cml_parser.arg('--mount-point', '/Volumes/rhino_root')
cml_parser.arg('--subject', subject)
cml_parser.arg('--electrode-config-file', '/home1/leond/fr5_config/contacts%s.csv' % subject)
cml_parser.arg('--pulse-frequency', '200')
cml_parser.arg('--target-amplitude', '1.0')
cml_parser.arg('--anodes', '3LAHD2', '12RGRD1')
cml_parser.arg('--cathodes', '3LAHD3', '12RGRD2')
cml_parser.arg('--min-amplitudes', '0.25')
cml_parser.arg('--max-amplitudes', '1.0')

args = cml_parser.parse()

# ------------------------------- end of processing command line

from RamPipeline import RamPipeline

from tests.fr5_biomarker.FREventPreparation import FREventPreparation

from tests.fr5_biomarker.ComputeFRPowers import ComputeFRPowers

from tests.fr5_biomarker.MontagePreparation import MontagePreparation

from system_3_utils.ram_tasks.CheckElectrodeConfigurationClosedLoop3 import CheckElectrodeConfigurationClosedLoop3

from tests.fr5_biomarker.ComputeClassifier import ComputeClassifier

from tests.fr5_biomarker.ComputeClassifier import ComputeFullClassifier

from tests.fr5_biomarker.system3.ExperimentConfigGeneratorClosedLoop5 import ExperimentConfigGeneratorClosedLoop5

import numpy as np


class StimParams(object):
    def __init__(self, **kwds):
        pass
        # self.n_channels = kwds['n_channels']
        # self.elec1 = kwds['anode_num']
        # self.anode = kwds.get('anode', '')
        # self.elec2 = kwds['cathode_num']
        # self.cathode = kwds.get('cathode', '')
        # self.pulseFrequency = kwds['pulse_frequency']
        # self.pulseCount = kwds['pulse_count']
        # self.amplitude = kwds['target_amplitude']
        #
        # self.duration = 300
        # self.trainFrequency = 1
        # self.trainCount = 1


class Params(object):
    def __init__(self):
        self.version = '3.00'

        self.include_fr1 = True
        self.include_catfr1 = True
        self.include_fr3 = False
        self.include_catfr3 = False

        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.fr1_retrieval_start_time = -0.525
        self.fr1_retrieval_end_time = 0.0
        self.fr1_retrieval_buf = 0.524

        self.encoding_samples_weight = 2.5

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(6), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200

        self.stim_params = StimParams(
            # n_channels=args.n_channels,
            # anode_num=args.anode_num,
            # anode=args.anode,
            # cathode_num=args.cathode_num,
            # cathode=args.cathode,
            # pulse_frequency=args.pulse_frequency,
            # pulse_count=args.pulse_frequency*args.pulse_duration/1000,
            # target_amplitude=args.target_amplitude
        )

        # self.stim_params = StimParams(
        #     n_channels=args.n_channels,
        #     anode_num=args.anode_num,
        #     anode=args.anode,
        #     cathode_num=args.cathode_num,
        #     cathode=args.cathode,
        #     pulse_frequency=args.pulse_frequency,
        #     pulse_count=args.pulse_frequency*args.pulse_duration/1000,
        #     target_amplitude=args.target_amplitude
        # )


params = Params()
# class ConfigError(Exception):
#     pass
#
#
# config_file = args.electrode_config_file
#
# if config_file.endswith('csv'):
#     if not isfile(config_file.replace('csv','bin')):
#         raise ConfigError('Needs binary config file along with csv config file')
# elif config_file.endswith('bin'):
#     if not isfile(config_file.replace('bin','csv')):
#         raise ConfigError('Needs csv config file along with binary file')
#
# else:
#     raise ConfigError('Unknown config file type')


# TODO - we need to check if all files need for bipolar referencing are ready before executing the whole pipeine

if args.bipolar:
    electrode_config_file = args.electrode_config_file
    electrode_config_file_dir = dirname(electrode_config_file)
    trans_matrix_fname = join(electrode_config_file_dir, 'monopolar_trans_matrix%s.h5' % args.subject)

    if not exists(trans_matrix_fname):
        print ('Bipolar referencing needs %s' % ('monopolar_trans_matrix%s.h5' % args.subject))
        print ('Please run jacksheet_2_configuration_csv.sh script located in clinical_affairs/syste,3 folder of the RAM_UTILS repository')
        sys.exit(1)


class ReportPipeline(RamPipeline):
    def __init__(self, subject, workspace_dir, mount_point=None, args=None):
        RamPipeline.__init__(self)
        self.subject = subject
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)
        self.args = args


report_pipeline = ReportPipeline(subject=args.subject,
                                 workspace_dir=args.workspace_dir, mount_point=args.mount_point, args=args)

report_pipeline.add_task(FREventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(mark_as_completed=False))

report_pipeline.add_task(CheckElectrodeConfigurationClosedLoop3(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeFRPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeFullClassifier(params=params, mark_as_completed=False))

report_pipeline.add_task(ExperimentConfigGeneratorClosedLoop5(params=params, mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
