# command line example:
# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
import sys
from os.path import *


from ReportUtils import CMLParser,ReportPipeline

cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--subject','R1111M')
cml_parser.arg('--task','FR1')
cml_parser.arg('--workspace-dir','scratch/FR1_reports/')
cml_parser.arg('--mount-point','/Volumes/rhino_root/')
#cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')


print sys.path
args = cml_parser.parse()


from ReportTasks.EventPreparation import FREventPreparation

from RepetitionRatio import RepetitionRatio

from ReportTasks.ComputePowers import ComputePowers

from MontagePreparation import MontagePreparation

from ComputeFR1HFPowers import ComputeFR1HFPowers

from ComputeTTest import ComputeTTest

from ComputeClassifier import ComputeClassifier

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5
        self.task = args.task

        self.start = 0.0
        self.end = 1.366
        self.buf = 1.365


        self.filt_order = 4
        self.butter_freqs = [58.,62.]


        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200

class HFParams(Params):
    def __init__(self):
        super(HFParams, self).__init__()
        self.end = 1.6
        self.buf = 1.0

        self.freqs = np.logspace(np.log10(2), np.log10(200), 50)
        self.freqs = self.freqs[self.freqs>=70.0]



params = Params()

hf_params = HFParams()

# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,experiment=args.task,sessions=args.sessions,
                                 workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)


report_pipeline.add_task(FREventPreparation(task=args.task,sessions = args.sessions,))

report_pipeline.add_task(MontagePreparation(params, mark_as_completed=False))

if 'cat' in args.task:
    report_pipeline.add_task(RepetitionRatio(mark_as_completed=False))

report_pipeline.add_task(ComputePowers(params=params, mark_as_completed=True,name='ComputeFR1Powers'))

report_pipeline.add_task(ComputePowers(params=hf_params, mark_as_completed=True,name='ComputeFR1HFPowers'))

report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

report_pipeline.add_task(GenerateTex(mark_as_completed=False))

report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))


# starts processing pipeline
report_pipeline.execute_pipeline()
