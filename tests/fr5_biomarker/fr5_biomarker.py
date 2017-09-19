# command line example:
# python fr3_biomarker.py --workspace-dir=/scratch/busygin/FR3_biomarkers --subject=R1145J_1 --n-channels=128 --anode=RD2 --anode-num=34 --cathode=RD3 --cathode-num=35 --pulse-frequency=200 --pulse-duration=500 --target-amplitude=1000

print "ATTN: Wavelet params and interval length are hardcoded!! To change them, recompile"
print "Windows binaries from https://github.com/busygin/morlet_for_sys2_biomarker"
print "See https://github.com/busygin/morlet_for_sys2_biomarker/blob/master/README for detail."

from os.path import *
from BiomarkerUtils import CMLParserBiomarker


cml_parser = CMLParserBiomarker(arg_count_threshold=1)
cml_parser.arg('--workspace-dir','/scratch/leond/FR3_biomarkers_json')
cml_parser.arg('--subject','R1124J_1')
cml_parser.arg('--n-channels','128')
cml_parser.arg('--anode-nums','34','86')
cml_parser.arg('--anodes','LDA1','RDA3')
cml_parser.arg('--cathode-nums','35','87')
cml_parser.arg('--cathodes','LDA2','RDA4')
cml_parser.arg('--pulse-frequency','200')
cml_parser.arg('--pulse-duration','500')
cml_parser.arg('--target-amplitude','250')
cml_parser.arg('--anode-num','34')
cml_parser.arg('--cathode-num','35')


args = cml_parser.parse()


# ------------------------------- end of processing command line

from RamPipeline import RamPipeline

from FREventPreparation import FREventPreparation

from ComputeFRPowers import ComputeFRPowers

from MontagePreparation import MontagePreparation

from CheckElectrodeLabels import CheckElectrodeLabels

from ComputeClassifier import ComputeClassifier

from SaveMatlabFile import SaveMatlabFile

import numpy as np

class ArgumentError(Exception):
    pass




class StimParams(object):
    def __init__(self,**kwds):
        self.n_channels = kwds['n_channels']
        self.elec1 = kwds['anode_num']
        self.anode = kwds.get('anode', '')
        self.anodes = kwds.get('anodes',[self.anode])
        self.anode_nums = kwds.get('anode_nums',[self.elec1])
        self.elec2 = kwds['cathode_num']
        self.cathode = kwds.get('cathode', '')
        self.cathodes = kwds.get('cathodes',[self.cathode])
        self.cathode_nums=  kwds.get('cathode_nums',[self.elec2])
        self.pulseFrequency = kwds['pulse_frequency']
        self.pulseCount = kwds['pulse_count']
        self.amplitude = kwds['target_amplitude']

        self.duration = 300
        self.trainFrequency = 1
        self.trainCount = 1

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
            n_channels=args.n_channels,
            anode_num=args.anode_num,
            anode_nums=args.anode_nums,
            anode=args.anode,
            anodes= args.anodes,
            cathode_num=args.cathode_num,
            cathode_nums=args.cathode_nums,
            cathode=args.cathode,
            cathodes=args.cathodes,
            pulse_frequency=args.pulse_frequency,
            pulse_count=args.pulse_frequency*args.pulse_duration/1000,
            target_amplitude=args.target_amplitude
        )


params = Params()


class ReportPipeline(RamPipeline):

    def __init__(self, subject, workspace_dir, mount_point=None,args=None):
        RamPipeline.__init__(self)
        self.subject = subject
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir),
        self.args = args


report_pipeline = ReportPipeline(subject=args.subject,
                                       workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point,
                                 args=args)

report_pipeline.add_task(FREventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(mark_as_completed=False))

report_pipeline.add_task(CheckElectrodeLabels(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeFRPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=False))

report_pipeline.add_task(SaveMatlabFile(params=params, mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
