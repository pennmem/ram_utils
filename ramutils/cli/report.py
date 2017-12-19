"""Script for generating reports."""

import os.path as osp

from ramutils.cli import make_parser, configure_caching
from ramutils.log import get_logger
from ramutils.pipelines.report import make_report

parser = make_parser("Generate a report")
parser.add_argument('--sessions', '-S', nargs='+',
                    help='sessions to read data from (default: use all)')
parser.add_argument("--retrain", "-R", action="store_true",
                    help="retrain classifier rather than loading from disk")

logger = get_logger("reports")


def main(input_args=None):
    from ramutils.parameters import FilePaths

    args = parser.parse_args(input_args)

    configure_caching(args.dest, args.force_rerun)

    paths = FilePaths(
        root=osp.expanduser(args.root),
        dest=args.dest,
    )

    # FIXME: Construct stim parameters
    stim_params = None

    # Extract sessions
    if args.sessions is not None:
        sessions = [int(session) for session in args.sessions]
    else:
        sessions = None

    # FIXME: Select experiment parameters from command-line option
    exp_params = None

    # Generate report!
    path = make_report(
        subject=args.subject,
        experiment=args.experiment,
        paths=paths,
        retrain=args.retrain,
        stim_params=stim_params,
        exp_params=exp_params,
        sessions=sessions,
        vispath=args.vispath,
    )
    logger.info("Wrote report to %s", path)


if __name__ == "__main__":
    root = '~/mnt/rhino'
    dest = 'scratch/ramutils/demo'

    main([
        '--root', root, '--dest', dest,
        '-s', 'R1111M',
        '-x', 'FR1',
    ])
