from __future__ import print_function

from ramutils.parameters import FilePaths, FRParameters, PS5Parameters
from ramutils.pipelines.report import make_report
from ramutils.pipelines.aggregated_report import make_aggregated_report

from ramutils.tasks import memory

memory.cachedir = "/home/zachduey/ramutils"



params = FRParameters()

paths = FilePaths(
    root='/Volumes/RHINO/',
    dest='/scratch/zduey/',
    data_db='/scratch/report_database/'
)

params = FRParameters()
# make_report(subject, "catFR1", paths, exp_params=params, rerun=True,
#             joint_report=False, sessions=None, use_classifier_excluded_leads=True)
make_aggregated_report(experiments=['catFR5', 'FR5'], fit_model=False, paths=paths)
