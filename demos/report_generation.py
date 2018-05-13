from __future__ import print_function

import os.path
from ramutils.parameters import FilePaths, FRParameters, PS5Parameters
from ramutils.pipelines.report import make_report

from ramutils.tasks import memory

memory.cachedir = "~"



paths = FilePaths(
    root='/',
    dest='/scratch/zduey/',
    data_db='/scratch/zduey/'
)

params = FRParameters()
make_report("R1001P", "FR1", paths, exp_params=params, stim_params=None,
            joint_report=False, rerun=False, sessions=None)
