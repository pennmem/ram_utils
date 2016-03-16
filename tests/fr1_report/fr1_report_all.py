import re
import sys
from glob import glob

from setup_utils import parse_command_line, configure_python_paths

# -------------------------------processing command line
if len(sys.argv)>2:

    args = parse_command_line()


else: # emulate command line
    command_line_emulation_argument_list = ['--subject','R1086M',
                                            '--task','RAM_FR1',
                                            '--workspace-dir','/scratch/busygin/FR1_reports_new_new',
                                            '--mount-point','',
                                            '--python-path','/home1/busygin/ram_utils_new_ptsa',
                                            '--python-path','/home1/busygin/python/ptsa_new'
                                            ]
    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line

from FR1EventPreparation import FR1EventPreparation

from MathEventPreparation import MathEventPreparation

from ComputeFR1Powers import ComputeFR1Powers

from TalPreparation import TalPreparation

from GetLocalization import GetLocalization

from ComputeTTest import ComputeTTest

#from CheckTTest import CheckTTest

from ComputeClassifier import ComputeClassifier

#from CheckClassifier import CheckClassifier

from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.ttest_frange = (70.0, 200.0)

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


params = Params()


class ReportPipeline(RamPipeline):
    def __init__(self, subject, task, workspace_dir, mount_point=None):
        RamPipeline.__init__(self)
        self.subject = subject
        self.task = self.experiment = task
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)


task = 'RAM_FR1'


def find_subjects_by_task(task):
    ev_files = glob(args.mount_point + ('/data/events/%s/R*_events.mat' % task))
    return [re.search(r'R\d\d\d\d[A-Z](_\d+)?', f).group() for f in ev_files]


subjects = find_subjects_by_task(task)
subjects.sort()

for subject in subjects:
    print '--Generating', task, 'report for', subject

    # sets up processing pipeline
    report_pipeline = ReportPipeline(subject=subject, task=task,
                                           workspace_dir=join(args.workspace_dir,task+'_'+subject), mount_point=args.mount_point)

    report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))

    report_pipeline.add_task(MathEventPreparation(mark_as_completed=False))

    report_pipeline.add_task(TalPreparation(mark_as_completed=False))

    report_pipeline.add_task(GetLocalization(mark_as_completed=False))

    report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=True))

    #report_pipeline.add_task(CheckTTest(params=params, mark_as_completed=False))

    #report_pipeline.add_task(XValTTest(params=params, mark_as_completed=False))

    #report_pipeline.add_task(XValPlots(params=params, mark_as_completed=False))

    #
    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))
    #
    # #report_pipeline.add_task(CheckClassifier(params=params, mark_as_completed=False))
    #
    report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))
    #
    report_pipeline.add_task(GeneratePlots(mark_as_completed=False))
    #
    report_pipeline.add_task(GenerateTex(mark_as_completed=False))
    #
    report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

    # starts processing pipeline
    try:
        report_pipeline.execute_pipeline()
    except KeyboardInterrupt:
        print 'GOT KEYBOARD INTERUPT. EXITING'
        sys.exit()
    except:
        print 'Failed for', subject
