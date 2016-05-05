# command line example:
# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
import sys
import os
import numpy as np

# from setup_utils import parse_command_line, configure_python_paths

from ReportUtils import CMLParser,ReportPipeline

cml_parser = CMLParser(arg_count_threshold=1)

cml_parser.arg('--subject','R1162N')
cml_parser.arg('--workspace-dir','/scratch/mswat/automated_reports/PAL1_biomarker')
cml_parser.arg('--mount-point','')



# cml_parser.arg('--subject','R1162N')
# cml_parser.arg('--workspace-dir','/Users/m/automated_reports/PAL1_biomarker')
# cml_parser.arg('--mount-point','/Volumes/rhino_root')
# cml_parser.arg('--python-path','/Users/m/PTSA_NEW_GIT/')
# cml_parser.arg('--python-path','/Users/m/RAM_UTILS_GIT')



# cml_parser.arg('--subject','R1162N')
# cml_parser.arg('--workspace-dir','/scratch/busygin/PAL3_biomarkers')
# cml_parser.arg('--mount-point','')
# cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')

args = cml_parser.parse()



from PAL1EventPreparation import PAL1EventPreparation

from ComputePAL1Powers import ComputePAL1Powers

from TalPreparation import TalPreparation

from ComputeClassifier import ComputeClassifier

from SaveMatlabFile import SaveMatlabFile


# turn it into command line options

class StimParams(object):
    def __init__(self):
        self.n_channels = 128
        self.elec1 = 3
        self.elec2 = 4
        self.amplitude = 500
        self.duration = 300
        self.trainFrequency = 1
        self.trainCount = 1
        self.pulseFrequency = 200
        self.pulseCount = 100

class Params(object):
    def __init__(self):
        self.version = '2.00'

        self.width = 5

        self.pal1_start_time = 0.4
        self.pal1_end_time = 2.7
        self.pal1_buf = 1.0

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200

        self.stim_params = StimParams()


params = Params()


# sets up processing pipeline
report_pipeline = ReportPipeline(subject=args.subject,
                                 workspace_dir=os.path.join(args.workspace_dir, args.subject),
                                 mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)

report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))

report_pipeline.add_task(TalPreparation(mark_as_completed=False))

report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(SaveMatlabFile(params=params, mark_as_completed=False))


# starts processing pipeline
report_pipeline.execute_pipeline()
