import sys
import time
from os.path import *

from pal5_prompt import parse_command_line, Args

from ....RamPipeline import RamPipeline

from ..MontagePreparation import MontagePreparation

from ....system_3_utils.ram_tasks.CheckElectrodeConfigurationClosedLoop3 import CheckElectrodeConfigurationClosedLoop3

from ..ComputeClassifier import ComputeClassifier,ComputeFullClassifier

from ..ComputeBiomarkerThreshold import ComputeBiomarkerThreshold

from ..ComputePAL1Powers import ComputePAL1Powers

from ..PAL1EventPreparation import PAL1EventPreparation

from ..LogResults import LogResults

from ..ComputeEncodingClassifier import ComputeEncodingClassifier


from .ExperimentConfigGeneratorClosedLoop5 import ExperimentConfigGeneratorClosedLoop5

import numpy as np


class StimParams(object):
    def __init__(self, **kwds):
        pass


class Params(object):
    def __init__(self):
        self.version = '3.00'

        self.include_fr1 = True
        self.include_catfr1 = True
        self.include_fr3 = True
        self.include_catfr3 = True

        self.width = 5

        self.pal1_start_time = 0.3
        self.pal1_end_time = 2.00
        self.pal1_buf = 1.2

        # original code
        self.pal1_retrieval_start_time = -0.625
        self.pal1_retrieval_end_time = -0.1
        self.pal1_retrieval_buf = 0.524


        self.encoding_samples_weight = 1.0

        self.recall_period = 5.0

        self.sliding_window_interval = 0.1
        self.sliding_window_start_offset = 0.3

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(6), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 0.048


        self.n_perm = 200


        self.stim_params = StimParams(
        )






def make_biomarker(args):    # report_pipeline = ReportPipeline(subject=args.subject,
    #                                  workspace_dir=join(args.workspace_dir, args.subject), mount_point=args.mount_point,
    #                                  args=args)
    params=Params()
    try:
        args.min_amplitudes = [args.min_amplitude_1, args.min_amplitude_2]
        args.max_amplitudes = [args.max_amplitude_1, args.max_amplitude_2]
    except AttributeError:
        args.min_amplitudes = [args.min_amplitude]
        args.max_amplitudes = [args.max_amplitude]

    class ReportPipeline(RamPipeline):
        def __init__(self, subject, workspace_dir, mount_point=None, args=None):
            RamPipeline.__init__(self)
            self.subject = subject
            self.mount_point = mount_point
            self.set_workspace_dir(workspace_dir)
            self.args = args

    log_filename = join(args.workspace_dir, args.subject, time.strftime('%Y_%m_%d') + '.csv')
    report_pipeline = ReportPipeline(subject=args.subject,
                                     workspace_dir=join(args.workspace_dir,
                                        '{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                        args.subject,args.experiment,
                                        args.anodes[0],args.cathodes[0],args.max_amplitudes[0],
                                        args.anodes[1],args.cathodes[1],args.max_amplitudes[1]
                                        )
                                                        ),
                                     mount_point=args.mount_point,
                                     args=args)

    report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

    report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))

    #
    report_pipeline.add_task(CheckElectrodeConfigurationClosedLoop3(params=params, mark_as_completed=False))
    #
    report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeEncodingClassifier(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeBiomarkerThreshold(params=params, mark_as_completed=True))

    report_pipeline.add_task(LogResults(params=params, mark_as_completed=False, log_filename=log_filename))

    report_pipeline.add_task(ComputeFullClassifier(params=params, mark_as_completed=True))

    report_pipeline.add_task(ExperimentConfigGeneratorClosedLoop5(params=params, mark_as_completed=False))

    # starts processing pipeline
    report_pipeline.execute_pipeline()


