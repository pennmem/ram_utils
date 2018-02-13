from __future__ import print_function

import os.path
from ramutils.parameters import FilePaths, FRParameters, PS5Parameters
from ramutils.pipelines.report import make_report

from ramutils.tasks import memory

memory.cachedir = "/Users/zduey/tmp/"


subject = 'R1385E'
rhino = os.path.expanduser('/Volumes/rhino')
pairs_path = os.path.join(
    'protocols', 'r1', 'subjects', subject,
    'localizations', str(0),
    'montages', str(0),
    'neuroradiology', 'current_processed', 'pairs.json')

paths = FilePaths(
    root='/Volumes/RHINO/',
    pairs=pairs_path,
    dest='/scratch/zduey/FR5_CatFR5/reports/',
    data_db='/scratch/zduey/FR5_CatFR5/'
)

params = FRParameters()
make_report(subject, "catFR5", paths, exp_params=params, stim_params=None,
            joint_report=False, sessions=[0], rerun=True, retrain=True)