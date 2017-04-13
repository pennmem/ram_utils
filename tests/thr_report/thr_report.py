# command line example:
# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
import sys
from os.path import *


from ReportUtils import CMLParser,ReportPipeline

cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--subject','R1294C')
cml_parser.arg('--task','THR')
cml_parser.arg('--workspace-dir','/scratch/jfm2/THR_reports')
cml_parser.arg('--mount-point','')
#cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')

args = cml_parser.parse()


from THREventPreparation import THREventPreparation

from ComputeTHRPowers import ComputeTHRPowers

from MontagePreparation import MontagePreparation

from ComputeTHRHFPowers import ComputeTHRHFPowers

from ComputeTTest import ComputeTTest

from ComputeClassifier import ComputeClassifier

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.thr_start_time = 0.0
        self.thr_end_time = 1.3
        self.thr_buf = 1.299

        self.ttest_start_time = 0.0
        self.ttest_end_time = 1.5
        self.ttest_buf = 1.0

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(1), np.log10(200), 8)

        self.ttest_names = ['Low Theta', 'High Theta', 'Gamma', 'HFA']
        self.ttest_freqs = np.logspace(np.log10(1), np.log10(200), 30)
        self.ttest_frange = np.array([[1.0, 3.0], [3.0, 9.0], [40.0, 70.0], [70.0, 200.0]])

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 250


params = Params()



# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,experiment=args.task,sessions =args.sessions,
                                 workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)


report_pipeline.add_task(THREventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(params, mark_as_completed=False))

report_pipeline.add_task(ComputeTHRPowers(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeTHRHFPowers(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=False))

report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

report_pipeline.add_task(GenerateTex(mark_as_completed=False))

report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))


# starts processing pipeline
report_pipeline.execute_pipeline()
