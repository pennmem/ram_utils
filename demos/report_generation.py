from __future__ import print_function

from ramutils.parameters import FilePaths, FRParameters, PS5Parameters
from ramutils.pipelines.report import make_report
from ramutils.pipelines.aggregated_report import make_aggregated_report

from ramutils.tasks import memory

memory.cachedir = "/home/zachduey/ramutils"


paths = FilePaths(
    root='/mnt/rhino/',
    dest='/scratch/zduey/',
    data_db='/scratch/report_database/'
)

params = FRParameters()
# make_report('R1384J', "catFR5", paths, exp_params=params, stim_params=None,
#             joint_report=False, sessions=[1], rerun=False, retrain=False)

make_aggregated_report('R1384J', 'catFR5', joint=True, paths=paths)
