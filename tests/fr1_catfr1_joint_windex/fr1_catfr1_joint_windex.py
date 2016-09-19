import sys
import os
from glob import glob
import re
import numpy as np

from ReportUtils import CMLParser,ReportPipeline

cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--workspace-dir','/scratch/busygin/fr1_catfr1_joint_base')
cml_parser.arg('--mount-point','')

args = cml_parser.parse()


from FR1EventPreparation import FR1EventPreparation

from ComputeFR1Powers import ComputeFR1Powers

from MontagePreparation import MontagePreparation

from ComputeClassifier import ComputeClassifier


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
        self.hfs = np.logspace(np.log10(2), np.log10(200), 50)
        self.hfs = self.hfs[self.hfs>=70.0]

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200

        self.windex_cleanup = False
        self.windex_ied_cleanup = False


params = Params()


def find_subjects_by_task(task):
    ev_files = glob(args.mount_point + ('/data/events/%s/R*_events.mat' % task))
    return [re.search(r'R\d\d\d\d[A-Z](_\d+)?', f).group() for f in ev_files]


subjects = np.unique(find_subjects_by_task('RAM_FR1') + find_subjects_by_task('RAM_CatFR1'))
subjects = subjects[:-2]
subjects = subjects[(subjects!='R1061T') & (subjects!='R1070T') & (subjects!='R1092J_2') & (subjects!='R1093J_1') & (subjects!='R1108J') & (subjects!='R1135E_1')]

subjects = ['R1201P_1']

for subject in subjects:
    #print '--Generating FR1&CatFR1 joint report for', subject
    if args.skip_subjects is not None and subject in args.skip_subjects:
        continue

    report_pipeline = ReportPipeline(
        args=args,
        subject=subject,
        workspace_dir=os.path.join(args.workspace_dir,  subject)
    )

    report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))

    report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

    # starts processing pipeline
    report_pipeline.execute_pipeline()
