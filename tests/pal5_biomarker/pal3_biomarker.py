# command line example:
# python pal3_biomarker_db.py --workspace-dir=/scratch/busygin/PAL3_biomarkers --subject=R1162N --n-channels=128 --anode=AD2 --anode-num=56 --cathode=AD3 --cathode-num=57 --pulse-frequency=200 --pulse-duration=500 --target-amplitude=1000

print "ATTN: Wavelet params and interval length are hardcoded!! To change them, recompile"
print "Windows binaries from https://github.com/busygin/morlet_for_sys2_biomarker"
print "See https://github.com/busygin/morlet_for_sys2_biomarker/blob/master/README for detail."

from os.path import *
from BiomarkerUtils import CMLParserBiomarker


cml_parser = CMLParserBiomarker(arg_count_threshold=1)
cml_parser.arg('--workspace-dir','scratch/PAL3_biomarkers')
cml_parser.arg('--mount-point','/Volumes/rhino_root')
cml_parser.arg('--subject','R1312N')
cml_parser.arg('--n-channels','128')
cml_parser.arg('--anode-num','3')
cml_parser.arg('--anode','G3')
cml_parser.arg('--cathode-num','4')
cml_parser.arg('--cathode','G4')
cml_parser.arg('--pulse-frequency','100')
cml_parser.arg('--pulse-duration','300')
cml_parser.arg('--target-amplitude','1250')
cml_parser.arg('--sessions','1','2')


args = cml_parser.parse()


# ------------------------------- end of processing command line

from RamPipeline import RamPipeline

from PAL1EventPreparation import PAL1EventPreparation

from ComputePAL1Powers import ComputePAL1Powers

from MontagePreparation import MontagePreparation

from CheckElectrodeLabels import CheckElectrodeLabels

from ComputeClassifier import ComputeClassifier

from ComputeEncodingClassifier import ComputeEncodingClassifier

from ComputeBiomarkerThreshold import ComputeBiomarkerThreshold

from system3.ExperimentConfigGeneratorClosedLoop5 import ExperimentConfigGeneratorClosedLoop5

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
        self.amplitude = kwds['target_amplitude']

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

        # original code
        self.pal1_retrieval_start_time = -0.625
        self.pal1_retrieval_end_time = -0.1
        self.pal1_retrieval_buf = 0.524


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


        self.n_perm = 200


        self.stim_params = StimParams(**vars(args)
        )


params = Params()


class ReportPipeline(RamPipeline):

    def __init__(self, subject, workspace_dir, mount_point=None,args=None):
        RamPipeline.__init__(self)
        self.subject = subject
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)
        self.args=args


report_pipeline = ReportPipeline(subject=args.subject,
                                       workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point,
                                 args=args)

report_pipeline.add_task(MontagePreparation(mark_as_completed=False,params=None))

report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))

report_pipeline.add_task(CheckElectrodeLabels(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeEncodingClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeBiomarkerThreshold(params=params, mark_as_completed=True))

report_pipeline.add_task(ExperimentConfigGeneratorClosedLoop5(params=params, mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
