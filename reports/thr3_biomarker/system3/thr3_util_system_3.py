# command line example:
# python thr3_util_system_3.py --workspace-dir=/scratch/busygin/FR3_biomarkers --subject=R1145J_1 --n-channels=128 --anode=RD2 --anode-num=34 --cathode=RD3 --cathode-num=35 --pulse-frequency=200 --pulse-duration=500 --target-amplitude=1000

print "ATTN: Wavelet params and interval length are hardcoded!! To change them, recompile"
print "Windows binaries from https://github.com/busygin/morlet_for_sys2_biomarker"
print "See https://github.com/busygin/morlet_for_sys2_biomarker/blob/master/README for detail."

from os.path import *

from system_3_utils.ram_tasks import CMLParserClosedLoop3
cml_parser = CMLParserClosedLoop3.CMLParserCloseLoop3(arg_count_threshold=1)
cml_parser.arg('--workspace-dir','/Volumes/rhino_root/scratch/leond/THR3_biomarkers/R1328E')
cml_parser.arg('--experiment','THR3')
cml_parser.arg('--mount-point','/Users/leond')
cml_parser.arg('--subject','R1328E')
cml_parser.arg('--electrode-config-file',r'/Volumes/PATRIOT/R1328E/R1328E_19SEP2017L0M0STIM.csv')
cml_parser.arg('--pulse-frequency','100')
cml_parser.arg('--target-amplitude','1.0')
cml_parser.arg('--anode-num','48')
cml_parser.arg('--anode','5LD8')
cml_parser.arg('--cathode-num','49')
cml_parser.arg('--cathode','5LD9')




# ------------------------------- end of processing command line

from ReportUtils import ReportPipeline

from tests.thr3_biomarker.THREventPreparation import THREventPreparation

from tests.thr3_biomarker.ComputeTHRPowers import ComputeTHRPowers

from tests.thr3_biomarker.MontagePreparation import MontagePreparation

from system_3_utils.ram_tasks.CheckElectrodeConfigurationClosedLoop3 import CheckElectrodeConfigurationClosedLoop3

from tests.thr3_biomarker.ComputeClassifier import ComputeClassifier

from tests.thr3_biomarker.system3.ExperimentConfigGeneratorClosedLoop3 import ExperimentConfigGeneratorClosedLoop3

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
    def __init__(self,args):
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

        self.stim_params = None


if __name__ =='__main__':
    args = cml_parser.parse()

    params = Params(args)

    report_pipeline = ReportPipeline(subject=args.subject,
                                           workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point, args=args)

    report_pipeline.add_task(THREventPreparation(mark_as_completed=False))

    report_pipeline.add_task(MontagePreparation(mark_as_completed=False))

    report_pipeline.add_task(CheckElectrodeConfigurationClosedLoop3(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputeTHRPowers(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=False))

    report_pipeline.add_task(ExperimentConfigGeneratorClosedLoop3(params=params, mark_as_completed=False))


    #
    # # report_pipeline.add_task(SaveMatlabFile(params=params, mark_as_completed=False))

    # starts processing pipeline
    report_pipeline.execute_pipeline()
