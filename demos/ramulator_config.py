from __future__ import print_function

import functools
import os.path
from pkg_resources import resource_filename

from ramutils.parameters import FilePaths, FRParameters
from ramutils.pipelines.ramulator_config import make_ramulator_config, make_stim_params

from ramutils.tasks import memory

memory.cachedir = "/Users/zduey/tmp/"


getpath = functools.partial(resource_filename, 'ramutils.test.test_data')

subject = 'R1354E'
rhino = os.path.expanduser('/Volumes/rhino')
pairs_path = os.path.join(
    rhino, 'protocols', 'r1', 'subjects', subject,
    'localizations', str(0),
    'montages', str(0),
    'neuroradiology', 'current_processed', 'pairs.json')

paths = FilePaths(
    root='/Volumes/RHINO/',
    electrode_config_file='/scratch/system3_configs/ODIN_configs/R1354E'
                          '/R1354E_26OCT2017L0M0STIM.csv',
    pairs=pairs_path,
    dest='scratch/zduey/sample_fr6_biomarkers'
)

params = FRParameters()

stim_params = make_stim_params(subject, ['1LD9', '5LD7'], ['1LD10', '5LD8'],
                               target_amplitudes=[1.0,1.0],
                               root=paths.root)
make_ramulator_config(subject, "FR6", paths, stim_params, exp_params=params)
