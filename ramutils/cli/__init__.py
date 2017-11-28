from argparse import ArgumentParser
import os.path
import tempfile

from ramutils.constants import EXPERIMENTS


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

    Notes
    -----
    Paths are relative to the root argument except for cachedir when specified.
    When not given, cachedir will default to::

        os.path.join(tempfile.gettempdir(), 'ramutils')

    """
    parser = ArgumentParser(description=description)
    parser.add_argument('--root', default='/', help='path to rhino root')
    parser.add_argument('--dest', '-d', default='.', help='directory to write output to')
    parser.add_argument('--cachedir', default=os.path.join(tempfile.gettempdir(), 'ramutils'),
                        help='absolute path for caching dir')
    parser.add_argument('--subject', '-s', required=True, type=str, help='subject ID')
    parser.add_argument('--force-rerun', action='store_true', help='force re-running all tasks')
    parser.add_argument('--experiment', '-x', required=True, type=str,
                        choices=allowed_experiments, help='experiment')
    parser.add_argument('--vispath', default=None, type=str,
                        help='path to save task graph visualization to')
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

    if invalidate and os.path.isdir(cachedir):
        memory.clear()

    return memory
