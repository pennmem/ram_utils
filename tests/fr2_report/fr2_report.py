# command line example:
# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
import sys
from os.path import *


from ReportUtils import CMLParser,ReportPipeline

cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--subject','R1235E')
cml_parser.arg('--task','FR2')
cml_parser.arg('--workspace-dir','/scratch/leond/FR2_reports')
cml_parser.arg('--mount-point','')
cml_parser.arg('--hf_num')
cml_parser.arg('--stim')
#cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')

print sys.path
args = cml_parser.parse()

args.task = args.task.upper()

if 'CAT' in args.task:
    args.task='cat'+args.task.split('CAT')[1]



if args.use_matlab_events:
    from FR2MatEventPreparation import FR2EventPreparation
else:
    from FR2EventPreparation import FR2EventPreparation

from RepetitionRatio import RepetitionRatio

from ComputeFR2Powers import ComputeFR2Powers

from MontagePreparation import MontagePreparation

from ComputeFR2HFPowers import ComputeFR2HFPowers

from ComputeTTest import ComputeTTest

from ComputeClassifier import ComputeClassifier

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.hfs_start_time = 0.0
        self.hfs_end_time = 1.6
        self.hfs_buf = 1.0

        self.filt_order = 4
        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        if not args.hf_num:
            self.hfs = np.logspace(np.log10(2), np.log10(200), 50)
            self.hfs = self.hfs[self.hfs>=70.0]
        else:
            self.hfs = np.logspace(np.log10(float(args.hf_min)),np.log10(float(args.hf_max)),int(args.hf_num))

        self.log_powers = True

        self.stim = True if args.stim.capitalize()=='True' else (False if args.stim.capitalize()=='False' else None)
        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


params = Params()



# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,experiment=args.task,
                                 workspace_dir=join(args.workspace_dir,args.task+'_'+args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)

report_pipeline.add_task(FR2EventPreparation(params, mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(params, mark_as_completed=False))

if 'cat' in args.task:
    report_pipeline.add_task(RepetitionRatio(mark_as_completed=False))
name = '' if params.stim is None else('Stim' if params.stim is True else 'NoStim')

report_pipeline.add_task(ComputeFR2Powers(params=params, mark_as_completed=True, name='ComputeFR1' + name + 'Powers'))


report_pipeline.add_task(ComputeFR2HFPowers(params=params, mark_as_completed=True, name='ComputeFR1HF' + name + 'Powers'))

report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False, name='Compute' + name + 'TTest'))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True,name='Compute'+name+'Classifier'))

report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

report_pipeline.add_task(GenerateTex(mark_as_completed=False))

report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))


# starts processing pipeline
report_pipeline.execute_pipeline()
