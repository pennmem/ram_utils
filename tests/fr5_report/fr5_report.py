from ReportUtils import ReportPipeline,CMLParser

import os
import numpy as np

parser = CMLParser()
# Default-ish args here
parser.arg('--subject','R1293P')
parser.arg('--task','FR5')
parser.arg('--workspace-dir','/Users/leond/fr5_reports')
parser.arg('--mount-point','/Volumes/rhino_root')


#
args = parser.parse()


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
        self.version = '1.00'

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

        # self.retrieval_samples_weight = 0.5
        self.encoding_samples_weight =2.5


        self.filt_order = 4

        self.freqs = np.logspace(np.log10(6), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


params=Params()

from EventPreparation import FR1EventPreparation,FR5EventPreparation

from MontagePreparation import MontagePreparation

from ComputeFR5Powers import ComputeFR5Powers

from EvaluateClassifier import EvaluateClassifier

# from ComputePSPowers import ComputePSPowers

from ComputeFRPowers import ComputeFRPowers

from ComputeClassifier import ComputeClassifier

from ComputeFRStimTable import ComputeFRStimTable

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import GeneratePlots, GenerateTex, GenerateReportPDF


# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,experiment=args.task, sessions=args.sessions,
                                 workspace_dir=os.path.join(args.workspace_dir,args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)

report_pipeline.add_task(FR1EventPreparation())

report_pipeline.add_task(FR5EventPreparation())

report_pipeline.add_task(MontagePreparation(params=params,mark_as_completed=False))

report_pipeline.add_task(ComputeFRPowers(params=params,mark_as_completed=True))

report_pipeline.add_task(ComputeFR5Powers(params=params,mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params,mark_as_completed=True))

report_pipeline.add_task(EvaluateClassifier(params=params,mark_as_completed=True))

report_pipeline.add_task(ComputeFRStimTable(params=params,mark_as_completed=False))

report_pipeline.add_task(ComposeSessionSummary(params=params,mark_as_completed=False))

report_pipeline.add_task(GeneratePlots())

report_pipeline.add_task(GenerateTex(mark_as_completed=False))

report_pipeline.add_task(GenerateReportPDF())

report_pipeline.execute_pipeline()


