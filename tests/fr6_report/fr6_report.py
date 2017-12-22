import os
import numpy as np

from ReportUtils import ReportPipeline,CMLParser
from LoadEEG import LoadPostStimEEG
from EventPreparation import FR1EventPreparation, FR6EventPreparation
from MontagePreparation import MontagePreparation
from ComputeFR6Powers import ComputeFR6Powers
from EvaluateClassifier import EvaluateClassifier
from ComputeFRPowers import ComputeFRPowers
from ComputeClassifier import ComputeClassifier,ComputeFullClassifier
from ComputeFRStimTable import ComputeFRStimTable
from ComposeSessionSummary import ComposeSessionSummary
from GenerateReportTasks import GeneratePlots, GenerateTex, GenerateReportPDF


parser = CMLParser()
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

        self.post_stim_start_time = 0.030
        self.post_stim_end_time = 0.555
        self.post_stim_buf = 0.524

        self.encoding_samples_weight =2.5

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(6), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


params = Params()

report_pipeline = ReportPipeline(subject=args.subject,
                                 task=args.task,
                                 experiment=args.task,
                                 sessions=args.sessions,
                                 workspace_dir=os.path.join(args.workspace_dir,args.subject),
                                 mount_point=args.mount_point,
                                 exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)
report_pipeline.add_task(FR1EventPreparation())
report_pipeline.add_task(FR6EventPreparation())
report_pipeline.add_task(MontagePreparation(params=params,mark_as_completed=False))
report_pipeline.add_task(ComputeFRPowers(params=params,mark_as_completed=True))
report_pipeline.add_task(ComputeFullClassifier(params=params,mark_as_completed=True)) # This only applies to the FR1 events. This could be simplified to just load the classifier
report_pipeline.add_task(LoadPostStimEEG(params=params,mark_as_completed=True))
report_pipeline.add_task(ComputeFR6Powers(params=params,mark_as_completed=True)) 
report_pipeline.add_task(EvaluateClassifier(params=params,mark_as_completed=True)) 
report_pipeline.add_task(ComputeFRStimTable(params=params,mark_as_completed=True))
report_pipeline.add_task(ComposeSessionSummary(params=params,mark_as_completed=False))
#report_pipeline.add_task(ComputeStimEffect(params=params, mark_as_completed=False))
report_pipeline.add_task(GeneratePlots())
report_pipeline.add_task(GenerateTex(mark_as_completed=False))
report_pipeline.add_task(GenerateReportPDF())
report_pipeline.execute_pipeline()