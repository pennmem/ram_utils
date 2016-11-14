# command line example:
# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
import sys
from os.path import *

# sys.path.append(join(dirname(__file__),'..','..'))

from ReportUtils import CMLParser,ReportPipeline

import numpy as np


cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--subject','R1236J')
cml_parser.arg('--workspace-dir','/scratch/leond/FR_connectivity_report')
cml_parser.arg('--mount-point','')
#cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')


args = cml_parser.parse()


from FR1EventPreparation import FR1EventPreparation

from MontagePreparation import MontagePreparation

from ComputeFR1PhaseDiff import ComputeFR1PhaseDiff

#from LoadESPhaseDiff import LoadESPhaseDiff

from ComputePhaseDiffSignificance import ComputePhaseDiffSignificance

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import GenerateTex, GenerateReportPDF


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.fr1_start_time = -1.0
        self.fr1_end_time = 2.8
        self.fr1_buf = 1.0
        self.fr1_n_bins = 19

        self.filt_order = 4

        self.freqs = np.linspace(45.0, 95.0, 11)

        self.n_perms = 500

        self.save_fstat_and_zscore_mats = False


params = Params()



# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject,
                                 workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)


report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(params, mark_as_completed=True))

report_pipeline.add_task(ComputeFR1PhaseDiff(params=params, mark_as_completed=True))

#report_pipeline.add_task(LoadESPhaseDiff(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputePhaseDiffSignificance(params=params, mark_as_completed=True))

report_pipeline.add_task(ComposeSessionSummary(mark_as_completed=False))

report_pipeline.add_task(GenerateTex(mark_as_completed=False))

report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))


# starts processing pipeline
report_pipeline.execute_pipeline()
