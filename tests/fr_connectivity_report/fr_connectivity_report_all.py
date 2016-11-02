# command line example:
# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
import sys
import os
from os.path import *
from ptsa.data.readers.IndexReader import JsonIndexReader
# sys.path.append(join(dirname(__file__),'..','..'))

from ReportUtils import CMLParser,ReportPipeline

import numpy as np


cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--subject','R1050M')
cml_parser.arg('--workspace-dir','/scratch/busygin/FR_connectivity_report')
cml_parser.arg('--mount-point','')
#cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')


args = cml_parser.parse()


from ReportUtils import ReportSummaryInventory

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


params = Params()


json_reader = JsonIndexReader(os.path.join(args.mount_point,'data/eeg/protocols/r1.json'))
subject_set = json_reader.aggregate_values('subjects', experiment='FR1') | json_reader.aggregate_values('subjects', experiment='catFR1')
subjects = []
for s in subject_set:
    montages = json_reader.aggregate_values('montage', subject=s, experiment='FR1') | json_reader.aggregate_values('montage', subject=s, experiment='catFR1')
    for m_ in montages:
        m = str(m_)
        subject = str(s)
        if m!='0':
            subject += '_' + m
        subjects.append(subject)
subjects.sort()


rsi = ReportSummaryInventory(label='FR_connectivity')

for subject in subjects:
    print '--Generating FR1&CatFR1 joint report for', subject
    if args.skip_subjects is not None and subject in args.skip_subjects:
        continue

    # sets up processing pipeline
    report_pipeline = ReportPipeline(
        args=args,
        subject=subject,
        task='FR1_catFR1_joint',
        experiment='FR1_catFR1_joint',
        workspace_dir=join(args.workspace_dir, subject)
    )

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

    rsi.add_report_summary(report_summary=report_pipeline.get_report_summary())


print 'this is summary for all reports report ', rsi.compose_summary(detail_level=1)

rsi.output_json_files(dir=args.status_output_dir)
