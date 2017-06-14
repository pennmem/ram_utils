# command line example:
# python fr3_util_system_3.py --workspace-dir=/scratch/busygin/FR3_biomarkers --subject=R1145J_1 --n-channels=128 --anode=RD2 --anode-num=34 --cathode=RD3 --cathode-num=35 --pulse-frequency=200 --pulse-duration=500 --target-amplitude=1000

print "ATTN: Wavelet params and interval length are hardcoded!! To change them, recompile"
print "Windows binaries from https://github.com/busygin/morlet_for_sys2_biomarker"
print "See https://github.com/busygin/morlet_for_sys2_biomarker/blob/master/README for detail."

from os.path import *

from ram_utils.system_3_utils.ram_tasks.CMLParserClosedLoop3 import CMLParserCloseLoop3

cml_parser = CMLParserCloseLoop3(arg_count_threshold=1)
cml_parser.arg('--workspace-dir','/scratch/leond/R1250N')
cml_parser.arg('--experiment','PAL3')
cml_parser.arg('--mount-point','/')
cml_parser.arg('--subject','R1250N')
cml_parser.arg('--electrode-config-file','/home1/leond/fr3_config/contactsR1250N.csv')
cml_parser.arg('--pulse-frequency','100')
cml_parser.arg('--target-amplitude','1000')
cml_parser.arg('--anode-num','10')
cml_parser.arg('--anode','PG10')
cml_parser.arg('--cathode-num','11')
cml_parser.arg('--cathode','PG11')



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

from ram_utils.RamPipeline import RamPipeline

from ram_utils.biomarkers.pal3_biomarker import ComputePAL1Powers

from ram_utils.biomarkers.pal3_biomarker import MontagePreparation, PAL1EventPreparation

from ram_utils.system_3_utils import CheckElectrodeConfigurationClosedLoop3

from ram_utils.biomarkers.pal3_biomarker import ComputeClassifier

from ram_utils.biomarkers.pal3_biomarker import ExperimentConfigGeneratorClosedLoop3


import numpy as np


class StimParams(object):
    def __init__(self,**kwds):
        pass

class Params(object):
    def __init__(self):
        self.version = '3.00'

        self.width = 5

        self.pal1_start_time = 0.3
        self.pal1_end_time = 2.0
        self.pal1_buf = 1.0

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

report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(mark_as_completed=False))

report_pipeline.add_task(CheckElectrodeConfigurationClosedLoop3(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(ExperimentConfigGeneratorClosedLoop3(params=params, mark_as_completed=False))


#
# # report_pipeline.add_task(SaveMatlabFile(params=params, mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
