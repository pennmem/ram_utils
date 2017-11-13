from argparse import ArgumentParser

EXPERIMENTS = {
    'record_only': [
        'FR1',
        'catFR1',
        'PAL1'
    ],
    'ps': [
        # 'PS2',
        'PS4_FR5',
        'PS4_catFR5'
    ],
    'closed_loop': [
        'FR3',
        'catFR4',
        'PAL3',
        'FR5',
        'catFR5',
        'PAL5',
        'FR6',
        'catFR6',
    ]
}


class ValidationError(Exception):
    """Raised when command-line arguments are invalid."""


def make_parser(description, allowed_experiments=sum([exps for exps in EXPERIMENTS.values()], [])):
    """Create a stub parser containing common options.

    Parameters
    ----------
    description : str
        Passed along to :class:`ArgumentParser`
    allowed_experiments : List[str]
        List of allowed experiments.

    Returns
    -------
    parser : ArgumentParser

    """
    parser = ArgumentParser(description=description)
    parser.add_argument('--dest', '-d', default='.', help='directory to write output to')
    parser.add_argument('--subject', '-s', required=True, type=str, help='subject ID')
    parser.add_argument('--force-rerun', action='store_true', help='force re-running all tasks')
    parser.add_argument('--experiment', '-x', required=True, type=str,
                        choices=allowed_experiments, help='experiment')
    parser.add_argument('--vispath', default=None, type=str,
                        help='Path to save task graph visualization to')
    return parser


def configure_caching(cachedir, invalidate=False):
    """Setup task caching.

    Parameters
    ----------
    cachedir : str
        Location to cache task outputs to.
    invalidate : bool
        Clear all cached files.

    Returns
    -------
    Configured caching object.

    """
    from ramutils.tasks import memory

    memory.cachedir = cachedir

    if invalidate:
        memory.clear()

    return memory
