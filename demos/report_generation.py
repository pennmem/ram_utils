from __future__ import print_function

import functools
import os.path
from pkg_resources import resource_filename

from ramutils.parameters import FilePaths, FRParameters
from ramutils.pipelines.report import make_report

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
    dest='scratch/zduey/sample_fr1_reports'
)

params = FRParameters()
params.combine_events = False
make_report(subject, "FR1", paths, sessions=[0], exp_params=params)