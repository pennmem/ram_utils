from __future__ import print_function

from ramutils.pipelines.aggregated_report import  make_aggregated_report
from ramutils.parameters import FilePaths
from ramutils.tasks import memory

memory.cachedir = "/Users/zduey/tmp/"


paths = FilePaths(
    root='/Volumes/RHINO/',
    dest='/scratch/zduey/',
    data_db='/data10/RAM/report_database/'
)

make_aggregated_report(None, ['FR5', 'CatFR5'], None, fit_model=False, paths=paths)
