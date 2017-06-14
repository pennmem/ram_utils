# command line example:
# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
import sys
from os.path import *

# sys.path.append(join(dirname(__file__),'..','..'))

from ...ReportUtils import CMLParser,ReportPipeline,ReportSummaryInventory

import numpy as np


cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--subject','R1026D')
cml_parser.arg('--workspace-dir','/scratch/RAM_maint/automated_connectivity_reports/FR_connectivity_reports')
cml_parser.arg('--mount-point','')
#cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')


args = cml_parser.parse()

#from LoadESPhaseDiff import LoadESPhaseDiff
from .FR1EventPreparation import FR1EventPreparation
from .MontagePreparation import MontagePreparation
from .ComputeFR1PhaseDiff import ComputeFR1PhaseDiff
from .ComputePhaseDiffSignificance import ComputePhaseDiffSignificance
from .ComposeSessionSummary import ComposeSessionSummary
from .GenerateReportTasks import GenerateTex, GenerateReportPDF,DeployReportPDF


from ptsa.data.readers import IndexReader


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

        self.save_fstat_and_zscore_mats = True



def run_report(args):
    report_pipeline = build_pipeline(args)


    # starts processing pipeline
    report_pipeline.execute_pipeline()


def build_pipeline(args):
    params = Params()
    # sets up processing pipeline
    report_pipeline = ReportPipeline(subject=args.subject,
                                     workspace_dir=join(args.workspace_dir, args.subject), mount_point=args.mount_point,
                                     exit_on_no_change=args.exit_on_no_change,
                                     recompute_on_no_status=args.recompute_on_no_status)
    report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))
    report_pipeline.add_task(MontagePreparation(params, mark_as_completed=True))
    report_pipeline.add_task(ComputeFR1PhaseDiff(params=params, mark_as_completed=True))
    # report_pipeline.add_task(LoadESPhaseDiff(params=params, mark_as_completed=True))
    report_pipeline.add_task(ComputePhaseDiffSignificance(params=params, mark_as_completed=True))
    report_pipeline.add_task(ComposeSessionSummary(mark_as_completed=False))
    report_pipeline.add_task(GenerateTex(mark_as_completed=False))
    report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))
    return report_pipeline

def run_all_reports(args):
    rsi = ReportSummaryInventory()
    jr = IndexReader.JsonIndexReader(join(args.mount_point,'protocols','r1.json'))
    subjects  = set(jr.subjects(experiment='FR1')+jr.subjects(experiment='catFR1'))
    for subject in subjects:
        montages  = set(jr.montages(subject=subject,experiment='FR1')+jr.montages(subject=subject,experiment='catFR1'))
        for montage in montages:
            subject += '_%s'%str(montage) if montage>0 else ''
            args.subject=subject
            report_pipeline= build_pipeline(args)
            report_pipeline.add_task(DeployReportPDF(False))
            report_pipeline.execute_pipeline()
            rsi.add_report_summary(report_pipeline.get_report_summary())

    rsi.output_json_files(args.report_status_dir)