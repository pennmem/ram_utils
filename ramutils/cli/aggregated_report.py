""" Script for generating aggregated stim reports """

from __future__ import print_function
import os.path as osp

from ramutils.cli import make_parser
from ramutils.exc import CommandLineError
from ramutils.log import get_logger, get_warning_accumulator
from ramutils.parameters import FilePaths
from ramutils.pipelines.aggregated_report import make_aggregated_report
from ramutils.utils import timer
from ramutils.tasks import memory


parser = make_parser("Generate a report", agg=True)
parser.add_argument('--sessions', '-S', nargs='+')
parser.add_argument('--fit-model', '-f', action='store_true', default=False)
parser.add_argument('--report_db_location',
                    help='location of report data database',
                    type=str, default="/scratch/report_database/")

logger = get_logger("reports")


def create_aggregate_report(input_args=None):
    args = parser.parse_args(input_args)
    warning_accumulator = get_warning_accumulator()

    paths = FilePaths(
        root=osp.expanduser(args.root),
        dest=args.dest,
        data_db=args.report_db_location
    )

    # Valid options
    if args.subjects is None and args.experiments is None and args.sessions is None:
        raise CommandLineError("Insufficient information. Must specify at least subject(s) or experiment(s)")

    if args.subjects is not None and args.sessions is not None:
        raise CommandLineError("Sessions may only be specified for single subject/experiment aggregations")

    if args.experiments is not None and args.sessions is not None:
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

    warnings = '\n' + warning_accumulator.format_all()
    if warnings is not None:
        logger.info(warnings)


if __name__ == "__main__":
    create_aggregate_report()
