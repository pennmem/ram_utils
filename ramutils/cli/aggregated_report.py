""" Script for generating aggregated stim reports """

from __future__ import print_function
import os.path as osp

from ramutils.cli import make_parser
from ramutils.exc import CommandLineError
from ramutils.log import get_logger
from ramutils.parameters import FilePaths
from ramutils.pipelines.aggregated_report import make_aggregated_report
from ramutils.utils import timer
from ramutils.tasks import memory


parser = make_parser("Generate a report", agg=True)
parser.add_argument('--fit-model', '-f', action='store_true', default=False,
                    help='Fit model to estimate behavioral effects of stim (very slow)')
parser.add_argument('--report_db_location',
                    help='location of report data database',
                    type=str, default="/scratch/report_database/")

logger = get_logger("reports")


def create_aggregate_report(input_args=None):
    args = parser.parse_args(input_args)

    paths = FilePaths(
        root=osp.expanduser(args.root),
        dest=args.dest,
        data_db=args.report_db_location
    )

    # Valid options
    if args.subject is None and args.experiment is None and args.sessions is None:
        raise CommandLineError("Insufficient information. Must specify at least subject(s) or experiment(s)")

    if args.subject is not None:
        if len(args.subject) > 1 and args.sessions is not None:
            raise CommandLineError("Sessions may only be specified for single subject/experiment aggregations")

    if args.experiment is not None and args.sessions is not None:
        if len(args.experiment) > 1:
            raise CommandLineError("Sessions may only be specified for single subject/experiment aggregations")

    with timer():
        path = make_aggregated_report(
            subjects=args.subject,
            experiments=args.experiment,
            sessions=args.sessions,
            fit_model=args.fit_model,
            paths=paths,
        )
        logger.info("Wrote report to %s\n", path)
        memory.clear()


if __name__ == "__main__":
    create_aggregate_report()
