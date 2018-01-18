from argparse import ArgumentParser
import os
import os.path

from ramutils import __version__
from ramutils.constants import EXPERIMENTS


class ValidationError(Exception):
    """Raised when command-line arguments are invalid."""


class RamArgumentParser(ArgumentParser):
    """Parse arguments and run common things afterwards."""
    def __init__(self, **kwargs):
        allowed_experiments = kwargs.pop('allowed_experiments')
        super(RamArgumentParser, self).__init__(**kwargs)
        default_cache_dir = os.path.expanduser(os.path.join('~', '.ramutils', 'cache'))

        self.add_argument('--root', default='/', help='path to rhino root (default: /)')
        self.add_argument('--dest', '-d', default='scratch/ramutils',
                          help='directory to write output to (default: scratch/ramutils)')
        self.add_argument('--cachedir', default=default_cache_dir,
                          help='absolute path for caching dir')
        self.add_argument('--subject', '-s', required=True, type=str, help='subject ID')
        self.add_argument('--force-rerun', action='store_true', help='force re-running all tasks')
        self.add_argument('--experiment', '-x', required=True, type=str,
                          choices=allowed_experiments, help='experiment')
        self.add_argument('--vispath', default=None, type=str,
                          help='path to save task graph visualization to')
        self.add_argument('--version', action='version',
                          version='ramutils version {}'.format(__version__))

    def _create_dirs(self, path):
        if os.path.exists(path):
            if os.path.isfile(path):
                raise RuntimeError("{} is a file but must be a directory".format(path))
        else:
            try:
                os.makedirs(path)
            except OSError:
                pass

    @staticmethod
    def _configure_caching(cachedir, invalidate=False):
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

    def parse_args(self, args=None, namespace=None):
        args = super(RamArgumentParser, self).parse_args(args, namespace)
        self._create_dirs(args.dest)
        self._create_dirs(args.cachedir)
        self._configure_caching(args.cachedir, args.force_rerun)
        return args


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

    return RamArgumentParser(description=description,
                             allowed_experiments=allowed_experiments)
