from __future__ import print_function

import os.path
from ramutils.parameters import FilePaths, FRParameters, PS5Parameters
from ramutils.pipelines.report import make_report

from ramutils.tasks import memory

memory.cachedir = "/Users/zduey/tmp/"


subject = 'R1401J'
rhino = os.path.expanduser('/Volumes/rhino')

paths = FilePaths(
    root='/Volumes/RHINO/',
    dest='/scratch/zduey/',
    data_db='/scratch/zduey/'
)

params = FRParameters()
make_report(subject, "FR1", paths, exp_params=params, stim_params=None,
            joint_report=True, rerun=True, sessions=[0, 1, 100, 102])