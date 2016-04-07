import sys
from glob import glob
import re


from ReportUtils import CMLParser

cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--workspace-dir','/scratch/mswat/automated_reports/PAL1_reports')
cml_parser.arg('--mount-point','')
cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')

args = cml_parser.parse()


# from setup_utils import parse_command_line, configure_python_paths
#
# # -------------------------------processing command line
# if len(sys.argv)>1:
#
#     args = parse_command_line()
#
#
# else: # emulate command line
#     # command_line_emulation_argument_list = ['--subject','R1028M',
#     #                                         '--task','RAM_PAL1',
#     #                                         '--workspace-dir','/scratch/busygin/PAL1_reports',
#     #                                         '--mount-point','',
#     #                                         '--python-path','/home1/busygin/ram_utils_new_ptsa',
#     #                                         '--python-path','/home1/busygin/python/ptsa_latest',
#     #                                         #'--exit-on-no-change'
#     #                                         ]
#
#     command_line_emulation_argument_list = [
#                                             '--task','RAM_PAL1',
#                                             '--workspace-dir','/scratch/mswat/PAL1_reports',
#                                             '--mount-point','',
#                                             '--python-path','/home1/mswat/RAM_UTILS_GIT',
#                                             '--python-path','/home1/mswat/PTSA_NEW_GIT',
#                                             '--python-path','/home1/mswat/extra_libs'
#                                             #'--exit-on-no-change'
#                                             ]
#     args = parse_command_line(command_line_emulation_argument_list)
#
# configure_python_paths(args.python_path)

# ------------------------------- end of processing command line



from ReportUtils import ReportSummaryInventory, ReportSummary
from ReportUtils import ReportPipelineBase


from PAL1EventPreparation import PAL1EventPreparation

from MathEventPreparation import MathEventPreparation

from ComputePAL1Powers import ComputePAL1Powers

from TalPreparation import TalPreparation

from GetLocalization import GetLocalization

from ComputePAL1HFPowers import ComputePAL1HFPowers

from ComputeTTest import ComputeTTest

from ComputeClassifier import ComputeClassifier

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.pal1_start_time = 1.0
        self.pal1_end_time = 3.0
        self.pal1_buf = 1.0

        self.hfs_start_time = 1.0
        self.hfs_end_time = 3.0
        self.hfs_buf = 1.0

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)
        self.hfs = np.logspace(np.log10(2), np.log10(200), 50)
        self.hfs = self.hfs[self.hfs>=70.0]

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


params = Params()


# class ReportPipeline(RamPipeline):
#     def __init__(self, subject, task, workspace_dir, mount_point=None, exit_on_no_change=False):
#         RamPipeline.__init__(self)
#         self.exit_on_no_change = exit_on_no_change
#         self.subject = subject
#         self.task = self.experiment = task
#         self.mount_point = mount_point
#         self.set_workspace_dir(workspace_dir)
#         dependency_tracker = DependencyChangeTrackerLegacy(subject=subject, workspace_dir=workspace_dir, mount_point=mount_point)
#
#         self.set_dependency_tracker(dependency_tracker=dependency_tracker)

# class ReportPipeline(ReportPipelineBase):
#     def __init__(self, subject, task, workspace_dir, mount_point=None, exit_on_no_change=False):
#         super(ReportPipeline,self).__init__(subject=subject, workspace_dir=workspace_dir, mount_point=mount_point, exit_on_no_change=exit_on_no_change)
#         self.task = task
#         self.experiment = task

class ReportPipeline(ReportPipelineBase):
    def __init__(self, subject, task, workspace_dir, mount_point=None, exit_on_no_change=False,recompute_on_no_status=False):
        super(ReportPipeline,self).__init__(subject=subject, workspace_dir=workspace_dir, mount_point=mount_point, exit_on_no_change=exit_on_no_change,recompute_on_no_status=recompute_on_no_status)
        self.task = task
        self.experiment = task



task = 'RAM_PAL1'


def find_subjects_by_task(task):
    ev_files = glob(args.mount_point + ('/data/events/%s/R*_events.mat' % task))
    return [re.search(r'R\d\d\d\d[A-Z](_\d+)?', f).group() for f in ev_files]


subjects = find_subjects_by_task(task)
# subjects.remove('R1050M')
# subjects.remove('R1136N')
subjects.sort()

rsi = ReportSummaryInventory(label=task)

for subject in subjects:
    print '--Generating', task, 'report for', subject

    # # sets up processing pipeline
    # report_pipeline = ReportPipeline(subject=subject, task=task,
    #                                        workspace_dir=join(args.workspace_dir,task+'_'+subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change)

    report_pipeline = ReportPipeline(subject=subject, task=task,
                                     workspace_dir=join(args.workspace_dir, task + '_' + subject),
                                     mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
                                     recompute_on_no_status=args.recompute_on_no_status)

    report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))

    report_pipeline.add_task(MathEventPreparation(mark_as_completed=False))

    report_pipeline.add_task(TalPreparation(mark_as_completed=False))

    report_pipeline.add_task(GetLocalization(mark_as_completed=False))

    report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputePAL1HFPowers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

    report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

    report_pipeline.add_task(GenerateTex(mark_as_completed=False))

    report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

    # report_pipeline.add_task(DeployReportPDF(mark_as_completed=False))

    report_pipeline.execute_pipeline()

    rsi.add_report_summary(report_summary=report_pipeline.get_report_summary())


print 'this is summary for all reports report ', rsi.compose_summary(detail_level=1)

rsi.output_json_files(dir=args.status_output_dir)
rsi.send_email_digest()
