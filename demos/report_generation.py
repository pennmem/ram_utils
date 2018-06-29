from ramutils.parameters import FilePaths, FRParameters
from ramutils.pipelines.report import make_report

from ramutils.tasks import memory

memory.cachedir = "scratch/ramutils_test"

paths = FilePaths(
    root="~/mnt/rhino",
    dest="scratch/ramutils_test/",
    data_db="/scratch/report_database",
)

params = FRParameters()
make_report("R1384J", "FR5", paths, exp_params=params, stim_params=None,
            joint_report=False, rerun=False, sessions=[0], clinical=True)
