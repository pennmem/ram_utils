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

subject = 'R1345D'
rhino = os.path.expanduser('/Volumes/rhino')
pairs_path = os.path.join(
    'protocols', 'r1', 'subjects', subject,
    'localizations', str(0),
    'montages', str(0),
    'neuroradiology', 'current_processed', 'pairs.json')

paths = FilePaths(
    root='/Volumes/RHINO/',
    electrode_config_file='/scratch/system3_configs/ODIN_configs/R1345D'
                          '/R1345D_10OCT2017L0M0STIM.csv',
    pairs=pairs_path,
    dest='scratch/zduey/samplefr1_reports'
)

params = FRParameters()
stim_params = make_stim_params(subject, ['LTG27', 'LTG20'], ['LTG28', 'LTG21'],
                               target_amplitudes=[1.0, 1.0],
                               root=paths.root)
make_report(subject, "FR1", paths, exp_params=params, stim_params=stim_params,
            sessions=[0], joint_report=False)