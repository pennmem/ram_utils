# command line example:
# python pal3_biomarker.py --workspace-dir=/scratch/busygin/PAL3_biomarkers --subject=R1162N --n-channels=128 --anode=AD2 --anode-num=56 --cathode=AD3 --cathode-num=57 --pulse-frequency=200 --pulse-duration=500 --target-amplitude=1000

from os.path import *
from BiomarkerUtils import CMLParserBiomarker


cml_parser = CMLParserBiomarker(arg_count_threshold=1)
# cml_parser.arg('--workspace-dir','/scratch/busygin/PAL3_biomarkers')
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

from PAL1EventPreparation import PAL1EventPreparation

from ComputePAL1Powers import ComputePAL1Powers

from TalPreparation import TalPreparation

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


class ReportPipeline(RamPipeline):

    def __init__(self, subject, workspace_dir, mount_point=None):
        RamPipeline.__init__(self)
        self.subject = subject
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)


report_pipeline = ReportPipeline(subject=args.subject,
                                       workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point)

report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))

report_pipeline.add_task(TalPreparation(mark_as_completed=False))

report_pipeline.add_task(CheckElectrodeLabels(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(SaveMatlabFile(params=params, mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
