"""Script for generating reports."""

from classiflib import ClassifierContainer

from ramutils.cli import make_parser, configure_caching
from ramutils.log import get_logger
from ramutils.pipelines.report import make_report

parser = make_parser("Generate a report")
parser.add_argument('--classifier', '-c', default=None, type=str, help="path to classifier container")

logger = get_logger("reports")


def main():
    args = parser.parse_args()

    configure_caching(args.dest, args.force_rerun)

    if args.classifier is not None:
        container = ClassifierContainer.load(args.classifier)
    else:
        container = None

    path = make_report(args.subject, args.experiment, container)
    logger.info("Wrote report to %s", path)


if __name__ == "__main__":
    main()
