from __future__ import print_function

import functools
import os.path
from pkg_resources import resource_filename

from ramutils.parameters import FilePaths, FRParameters
from ramutils.pipelines.ramulator_config import make_ramulator_config, make_stim_params

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
    dest='scratch/zduey/sample_catFR5_biomarkers/'
)

params = FRParameters()
stim_params = make_stim_params(subject, ['RB6'], ['RB7'],
                               target_amplitudes=[0.5, 0.5],
                               root=paths.root)
make_ramulator_config(subject, "catFR5", paths, stim_params, exp_params=params)

