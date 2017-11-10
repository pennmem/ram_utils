"""Script for generating reports."""

from ramutils.cli import make_parser, configure_caching, EXPERIMENTS

parser = make_parser("Generate a report",
                     [exp[group] for group, exp in EXPERIMENTS.items()][0])


def main():
    args = parser.parse_args()

    configure_caching(args.dest, args.force_rerun)

    print("TODO: run report")


if __name__ == "__main__":
    main()
