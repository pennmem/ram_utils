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
cml_parser.arg('--subject','R1156D')
cml_parser.arg('--task','RAM_FR1')
cml_parser.arg('--workspace-dir','/scratch/busygin/FR1_ppc')
cml_parser.arg('--mount-point','')
#cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')


args = cml_parser.parse()


from FR1EventPreparation import FR1EventPreparation

from MontagePreparation import MontagePreparation

from ComputeFR1Wavelets import ComputeFR1Wavelets

from ComputePPCFeatures import ComputePPCFeatures

from ComputeOutsamplePPCFeatures import ComputeOutsamplePPCFeatures

from ComputeTTest import ComputeTTest

from ComputeClassifier import ComputeClassifier


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)
        #self.freqs = np.array([180.0])

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


params = Params()



# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject, task=args.task,experiment=args.task,
                                 workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)


report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(params, mark_as_completed=True))

report_pipeline.add_task(ComputeFR1Wavelets(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputePPCFeatures(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeOutsamplePPCFeatures(params=params, mark_as_completed=True))

#report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=False))


# starts processing pipeline
report_pipeline.execute_pipeline()
