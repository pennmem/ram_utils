# command line example:

import sys
from setup_utils import parse_command_line, configure_python_paths
from os.path import join

# -------------------------------processing command line
if len(sys.argv)>2:

    args = parse_command_line()


else: # emulate command line
    command_line_emulation_argument_list = ['--workspace-dir','/scratch/busygin/ps_aggregator_joint',
                                            '--mount-point','',
                                            '--python-path','/home1/busygin/ram_utils_new_ptsa'
                                            ]

    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)

# ------------------------------- end of processing command line


import numpy as np
from RamPipeline import RamPipeline

from BuildAggregatePSTable import BuildAggregatePSTable

from CountSessions import CountSessions

from RunAnalysis import RunAnalysis

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.baseline_correction = True
        self.output_param = 'perf_diff'
        self.output_title = '$\Delta$ Post-Pre Expected Performance (%)'

        #self.output_param = 'prob_diff'
        #self.output_title = '$\Delta$ Post-Pre Classifier Output'

        #self.frequency_plot_regions = ['CA1', 'DG', 'PRC']
        self.frequency_plot_regions = []
        self.frequency_plot_areas = ['HC', 'MTLC', 'Frontal']

        #self.duration_plot_regions = ['CA1', 'DG', 'PRC']
        self.duration_plot_regions = []
        self.duration_plot_areas = ['HC', 'MTLC', 'Frontal']

        #self.amplitude_plot_regions = ['CA1', 'DG', 'PRC']
        self.amplitude_plot_regions = []
        self.amplitude_plot_areas = ['HC', 'MTLC']

        self.anova_areas = ['HC', 'MTLC', 'Frontal']


params = Params()


class ReportPipeline(RamPipeline):
    def __init__(self, workspace_dir, mount_point=None):
        RamPipeline.__init__(self)
        self.task = 'RAM_FR1'
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)



# sets up processing pipeline
report_pipeline = ReportPipeline(workspace_dir=args.workspace_dir, mount_point=args.mount_point)

report_pipeline.add_task(BuildAggregatePSTable(params=params, mark_as_completed=True))

report_pipeline.add_task(CountSessions(params=params, mark_as_completed=False))

report_pipeline.add_task(RunAnalysis(params=params, mark_as_completed=False))

report_pipeline.add_task(GeneratePlots(params=params, mark_as_completed=False))

report_pipeline.add_task(GenerateTex(mark_as_completed=False))

report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
