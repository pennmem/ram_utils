"""Script for generating reports.

The ``--excluded-contacts`` option is intended to be used in non-stim
experiments to see how a classifier would perform when excluding the contacts
chosen for stimulation.

"""

from __future__ import print_function
import os.path as osp

from ramutils.cli import make_parser
from ramutils.exc import UnsupportedExperimentError, TooManySessionsError, CommandLineError
from ramutils.log import get_logger
from ramutils.montage import make_stim_params
from ramutils.parameters import FilePaths, FRParameters, PS5Parameters
from ramutils.pipelines.report import make_report
from ramutils.utils import timer, is_stim_experiment
from ramutils.tasks import memory
import logging


parser = make_parser("Generate a report")
parser.add_argument("--retrain", "-R", action="store_true", default=False,
                    help="retrain classifier rather than loading from disk")
parser.add_argument('--joint-report', '-j', action='store_true', default=False,
                    help='include CatFR/FR for FR reports (default: off)')
parser.add_argument('--rerun', '-C', action="store_true", default=False,
                    help='do not use previously generated output')
parser.add_argument('--report_db_location',
                    help='location of report data database',
                    type=str, default="/scratch/report_database/")
parser.add_argument('--trigger-electrode', '-t', type=str,
                    help='Label of the electrode to use for triggering '
                         'stimulation in PS5')

logger = get_logger("reports")


def create_report(input_args=None):
    args = parser.parse_args(input_args)
    if args.debug:
        import dask
        dask.set_options(get=dask.get)
        logger.setLevel(logging.DEBUG)

    paths = FilePaths(
        root=osp.expanduser(args.root),
        dest=args.dest,
        data_db=args.report_db_location
    )

    # Stim params (used in make_report below) is really just used for excluding
    # pairs.
    stim_params = []

    # Extract sessions
    stim_experiment = is_stim_experiment(args.experiment)
    if (args.sessions is None or len(args.sessions) != 1):
        if stim_experiment:
            raise TooManySessionsError("Stim reports must be built one "
                                       "session at a time")
        elif args.sessions is not None:
            sessions = [int(session) for session in args.sessions]
        else:
            sessions = None
    else:
        sessions = [int(session) for session in args.sessions]

    if 'PS5' in args.experiment:
        exp_params = PS5Parameters()
    elif 'FR' in args.experiment or 'LocationSearch' in args.experiment:
        exp_params = FRParameters()
    elif 'PAL' in args.experiment:
        raise NotImplementedError("PAL experiments are not supported yet")
    else:
        raise UnsupportedExperimentError(
            "Unsupported experiment: " + args.experiment)

    if 'PS5' in args.experiment and args.trigger_electrode is None:
        raise CommandLineError("Must specify a trigger electrode for PS5 "
                               "experiments")

    # Generate report!
    # FIXME: stim_params should be called something different/just be a list of contacts to exclude
    with timer():
        path = make_report(
            subject=args.subject,
            experiment=args.experiment,
            paths=paths,
            joint_report=args.joint_report,
            retrain=args.retrain,
            stim_params=stim_params,
            exp_params=exp_params,
            sessions=sessions,
            vispath=args.vispath,
            rerun=args.rerun,
            trigger_electrode=args.trigger_electrode,
            use_classifier_excluded_leads=args.use_classifier_excluded_leads
        )
        logger.info("Wrote report to %s\n", path)
        memory.clear()  # remove cached intermediate results if build succeeds


if __name__ == "__main__":
    create_report()
