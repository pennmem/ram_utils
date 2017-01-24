# command line example:
# python fr3_util_system_3.py --workspace-dir=/scratch/busygin/FR3_biomarkers --subject=R1145J_1 --n-channels=128 --anode=RD2 --anode-num=34 --cathode=RD3 --cathode-num=35 --pulse-frequency=200 --pulse-duration=500 --target-amplitude=1000

print "ATTN: Wavelet params and interval length are hardcoded!! To change them, recompile"
print "Windows binaries from https://github.com/busygin/morlet_for_sys2_biomarker"
print "See https://github.com/busygin/morlet_for_sys2_biomarker/blob/master/README for detail."

from os.path import *

from CMLParserSystem3 import CMLParserSystem3

cml_parser = CMLParserSystem3(arg_count_threshold=1)
cml_parser.arg('--workspace-dir','D:/scratch/FR3_utils_system_3')
cml_parser.arg('--experiment','FR3')
cml_parser.arg('--mount-point','D:/')
cml_parser.arg('--subject','R1247P_1')
# cml_parser.arg('--electrode-config-file',r'd:\experiment_configs2\R1234M\FR3\config_files\R1170J_ALLCHANNELSSTIM.bin')
cml_parser.arg('--electrode-config-file',r'd:\experiment_configs\R1247P_FR3.bin')
# cml_parser.arg('--stim-electrode-pair',r'LST1_LST2')
cml_parser.arg('--anode-num','95')
cml_parser.arg('--cathode-num','97')




# cml_parser.arg('--workspace-dir','/scratch/busygin/FR3_biomarkers')
# cml_parser.arg('--subject','R1145J_1')
# cml_parser.arg('--n-channels','128')
# cml_parser.arg('--anode-num','3')
# cml_parser.arg('--cathode-num','4')
# cml_parser.arg('--pulse-frequency','200')
# cml_parser.arg('--pulse-count','100')
# cml_parser.arg('--target-amplitude','1000')


args = cml_parser.parse()

# ------------------------------- end of processing command line

from RamPipeline import RamPipeline

from tests.fr3_biomarker_json.FREventPreparation import FREventPreparation

from tests.fr3_biomarker_json.ComputeFRPowers import ComputeFRPowers

from tests.fr3_biomarker_json.MontagePreparation import MontagePreparation

from system_3_utils.ram_tasks.CheckElectrodeConfigurationClosedLoop3 import CheckElectrodeConfigurationClosedLoop3

from tests.fr3_biomarker_json.ComputeClassifier import ComputeClassifier

from ExperimentConfigGenerator import ExperimentConfigGenerator


import numpy as np


class StimParams(object):
    def __init__(self,**kwds):
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
        self.version = '2.00'

        self.include_fr1 = True
        self.include_catfr1 = True
        self.include_fr3 = True
        self.include_catfr3 = True

        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

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


class ReportPipeline(RamPipeline):

    def __init__(self, subject, workspace_dir, mount_point=None, args=None):
        RamPipeline.__init__(self)
        self.subject = subject
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)
        self.args = args


report_pipeline = ReportPipeline(subject=args.subject,
                                       workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point, args=args)

report_pipeline.add_task(FREventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(mark_as_completed=False))

report_pipeline.add_task(CheckElectrodeConfigurationClosedLoop3(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeFRPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(ExperimentConfigGenerator(params=params, mark_as_completed=False))


#
# # report_pipeline.add_task(SaveMatlabFile(params=params, mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
