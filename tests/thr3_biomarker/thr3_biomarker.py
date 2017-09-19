# command line example:
# python fr3_biomarker_db.py --workspace-dir=/scratch/busygin/FR3_biomarkers --subject=R1145J_1 --n-channels=128 --anode=RD2 --anode-num=34 --cathode=RD3 --cathode-num=35 --pulse-frequency=200 --pulse-duration=500 --target-amplitude=1000

print "ATTN: Wavelet params and interval length are hardcoded!! To change them, recompile"
print "Windows binaries from https://github.com/busygin/morlet_for_sys2_biomarker"
print "See https://github.com/busygin/morlet_for_sys2_biomarker/blob/master/README for detail."

from os.path import *
from BiomarkerUtils import CMLParserBiomarker


cml_parser = CMLParserBiomarker()
cml_parser.arg('--workspace-dir','/scratch/jfm2/THR3_biomarkers')
cml_parser.arg('--subject','R1294C')
cml_parser.arg('--n-channels','128')
cml_parser.arg('--anode-num','75')
cml_parser.arg('--anode','RPC1')
cml_parser.arg('--cathode-num','76')
cml_parser.arg('--cathode','RPC2')
cml_parser.arg('--pulse-frequency','200')
cml_parser.arg('--pulse-duration','500')
cml_parser.arg('--target-amplitude','1000')


args = cml_parser.parse()


# ------------------------------- end of processing command line

from ReportUtils import ReportPipeline

from THREventPreparation import THREventPreparation

from ComputeTHRPowers import ComputeTHRPowers

from MontagePreparation import MontagePreparation

from CheckElectrodeLabels import CheckElectrodeLabels

from ComputeClassifier import ComputeClassifier

from SaveMatlabFile import SaveMatlabFile

import numpy as np


class StimParams(object):
    def __init__(self,**kwds):
        self.n_channels = kwds['n_channels']
        self.elec1 = kwds['anode_num']
        self.anode = kwds.get('anode', '')
        self.elec2 = kwds['cathode_num']
        self.cathode = kwds.get('cathode', '')
        self.pulseFrequency = kwds['pulse_frequency']
        self.pulseCount = kwds['pulse_count']
        self.amplitude = kwds['target_amplitude']

        self.duration = 300
        self.trainFrequency = 1
        self.trainCount = 1

class Params(object):
    def __init__(self):
        self.version = '2.00'

        # These don't do anything? Left in anyway
        self.include_thr = True
        self.include_thr3 = True

        self.width = 5

        self.thr_start_time = 0.0
        self.thr_end_time = 1.366
        self.thr_buf = 1.365

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200

        self.stim_params = StimParams(
            n_channels=args.n_channels,
            anode_num=args.anode_num,
            anode=args.anode,
            cathode_num=args.cathode_num,
            cathode=args.cathode,
            pulse_frequency=args.pulse_frequency,
            pulse_count=args.pulse_frequency*args.pulse_duration/1000,
            target_amplitude=args.target_amplitude
        )


params = Params()


report_pipeline = ReportPipeline(subject=args.subject, sessions = args.sessions,
                                       workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point)

report_pipeline.add_task(THREventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(mark_as_completed=False))

report_pipeline.add_task(CheckElectrodeLabels(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeTHRPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(SaveMatlabFile(params=params, mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
