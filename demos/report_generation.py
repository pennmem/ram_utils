from __future__ import print_function

import functools
import os.path
from pkg_resources import resource_filename

from ramutils.parameters import FilePaths, FRParameters
from ramutils.pipelines.report import make_report
from ramutils.pipelines.ramulator_config import make_stim_params

from ramutils.tasks import memory

memory.cachedir = "/Users/zduey/tmp/"


getpath = functools.partial(resource_filename, 'ramutils.test.test_data')

subject = 'R1374T'
rhino = os.path.expanduser('/Volumes/rhino')
pairs_path = os.path.join(
    'protocols', 'r1', 'subjects', subject,
    'localizations', str(0),
    'montages', str(0),
    'neuroradiology', 'current_processed', 'pairs.json')

paths = FilePaths(
    root='/Volumes/RHINO/',
    electrode_config_file='/scratch/system3_configs/ODIN_configs/R1374T'
                          '/R1374T_12DEC2017L0M0STIM.csv',
    pairs=pairs_path,
    dest='scratch/zduey/samplefr1_reports'
)

params = FRParameters()
make_report(subject, "catFR1", paths, exp_params=params, stim_params=None,
            joint_report=True)