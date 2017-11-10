"""Script for generating reports."""

from ramutils.cli import make_parser, EXPERIMENTS

parser = make_parser("Generate a report",
                     [exp[group] for group, exp in EXPERIMENTS.items()][0])


def main():
    from ramutils.tasks import memory

    args = parser.parse_args()

    memory.cachedir = args.dest
    if args.force_rerun:
        memory.clear()

    print("TODO: run report")


if __name__ == "__main__":
    main()
