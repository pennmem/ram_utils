"""Script for generating reports.

The ``--excluded-contacts`` option is intended to be used in non-stim
experiments to see how a classifier would perform when excluding the contacts
chosen for stimulation.

"""

from __future__ import print_function
import os.path as osp

from ramutils.cli import make_parser, configure_caching
from ramutils.exc import UnsupportedExperimentError, TooManySessionsError
from ramutils.log import get_logger, get_warning_accumulator
from ramutils.montage import make_stim_params
from ramutils.parameters import FilePaths, FRParameters
from ramutils.pipelines.report import make_report
from ramutils.utils import timer, is_stim_experiment

parser = make_parser("Generate a report")
parser.add_argument('--sessions', '-S', nargs='+',
                    help='sessions to read data from (default: use all)')
parser.add_argument("--retrain", "-R", action="store_true", default=False,
                    help="retrain classifier rather than loading from disk")
parser.add_argument('--excluded-contacts', '-E', nargs='+',
                    help='contacts to exclude from classifier')
parser.add_argument('--joint-report', '-j', action='store_true', default=False,
                    help='include CatFR/FR for FR reports (default: off)')

logger = get_logger("reports")


def create_report(input_args=None):
    args = parser.parse_args(input_args)
    configure_caching(args.cachedir, args.force_rerun)
    warning_accumulator = get_warning_accumulator()

    paths = FilePaths(
        root=osp.expanduser(args.root),
        dest=args.dest,
    )

    # Stim params (used in make_report below) is really just used for excluding
    # pairs.
    if args.excluded_contacts is not None:
        stim_params = make_stim_params(args.subject,
                                       anodes=args.excluded_contacts,
                                       cathodes=args.excluded_contacts,
                                       root=paths.root)
    else:
        stim_params = []

    # Extract sessions
    stim_experiment = is_stim_experiment(args.experiment)
    if args.sessions is not None:
        if stim_experiment and len(args.sessions) != 1:
            raise TooManySessionsError("Stim reports must be built one "
                                       "session at a time")
        sessions = [int(session) for session in args.sessions]
    else:
        sessions = None

    if 'FR' in args.experiment:
        exp_params = FRParameters()
    elif 'PAL' in args.experiment:
        raise NotImplementedError("PAL experiments are not supported yet")
    else:
        raise UnsupportedExperimentError("Unsupported experiment: " + args.experiment)

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
        )
        logger.info("Wrote report to %s\n", path)

    warnings = '\n' + warning_accumulator.format_all()
    if warnings is not None:
        logger.info(warnings)


if __name__ == "__main__":
    create_report()
