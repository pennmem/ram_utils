"""Script to batch create a bunch of experiment configuration files for testing
new Ramulator releases.

"""

import functools
from operator import add

SUBJECT = "R1384J"
EXPERIMENTS = [
    "AmplitudeDeterimation",
    "FR1",
    "CatFR1",
    "FR5",
    "CatFR5",
    "PS4_FR5",
    "PS4_CatFR5",
    "PS5_FR",
    "PS5_CatFR",
    "TICL_FR",
#    "TICL_CatFR",
]
ANODES = ["LF7", "RS1", "LM5", "LS1"]
CATHODES = ["LF8", "RS2", "LM6", "LS2"]
MIN_AMPLITUDES = [0.1] * len(ANODES)
MAX_AMPLITUDES = [0.5] * len(ANODES)


def make_args(experiment, **kwargs):
    root = kwargs.pop('root', '/')
    dest = kwargs.pop('dest', None)

    args = [
        '-s', SUBJECT,
        '-x', experiment,
        '--root', root
    ]

    if dest is not None:
        args += ['-d', dest]

    def check_for(key):
        if key in kwargs:
            return ['--' + key.replace('_', '-'), ' '.join(str(a) for a in kwargs[key])]
        else:
            return []

    keys = ['trigger_pairs', 'anodes', 'cathodes', 'min_amplitudes', 'max_amplitudes', 'target_amplitudes']
    return functools.reduce(add, [args] + [check_for(key) for key in keys])


def generate_configs(executor, root=None, dest=None):
    """Submits several config generations to the executor."""
    from ramutils.cli.expconf import create_expconf

    futures = []
    submit = functools.partial(executor.submit, create_expconf)

    make_args_ = functools.partial(make_args, root=root, dest=dest) \
        if root is not None else make_args

    for experiment in EXPERIMENTS:
        if "FR1" in experiment:
            submit(make_args_(experiment))

        if experiment in ["FR5", "CatFR5"] or "TICL" in experiment:
            anodes = ANODES[:1]
            cathodes = CATHODES[:1]
            min_amplitudes = MIN_AMPLITUDES[:1]
            amplitudes = MAX_AMPLITUDES[:1]

            get_args = functools.partial(make_args_, experiment,
                                         anodes=anodes, cathodes=cathodes,
                                         target_amplitudes=amplitudes)

            if "TICL" not in experiment:
                futures.append(submit(get_args()))
            else:
                futures.append(submit(get_args(trigger_pairs=["LM5_LM6"])))

    return futures


if __name__ == "__main__":
    from argparse import ArgumentParser
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = ArgumentParser()
    parser.add_argument('--root', '-r', default=None, help='root path')
    parser.add_argument('--dest', '-d', default='scratch/ramutils_test', help='destination path')
    parser.add_argument('--max-workers', '-w', type=int, default=2, help='max worker processes')
    args = parser.parse_args()

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = generate_configs(executor, root=args.root, dest=args.dest)
        print(futures)
        for future in as_completed(futures):
            print(future.result())
