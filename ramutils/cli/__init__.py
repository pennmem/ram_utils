from argparse import ArgumentParser
import os
import os.path

from ramutils import __version__
from ramutils.constants import EXPERIMENTS


class RamArgumentParser(ArgumentParser):
    """Parse arguments and run common things afterwards."""

    def __init__(self, agg=False, **kwargs):
        allowed_experiments = kwargs.pop('allowed_experiments')
        super(RamArgumentParser, self).__init__(**kwargs)
        default_cache_dir = os.path.expanduser(
            os.path.join('~', '.ramutils', 'cache'))

        self.add_argument('--root', default='/',
                          help='path to rhino root (default: /)')
        self.add_argument('--dest', '-d', default='scratch/ramutils',
                          help='directory to write output to relative to ROOT \n (default: scratch/ramutils)')
        self.add_argument('--cachedir', default=default_cache_dir,
                          help='absolute path for caching dir')
        self.add_argument('--use-cached', action='store_true',
                          help='allow cached results from previous run to be reused')
        self.add_argument('--use-classifier-excluded-leads', '-u', action='store_true', default=False,
                          help='Exclude channels in classifier_excluded_leads.txt from classifier')
        self.add_argument('--vispath', default=None, type=str,
                          help='path to save task graph visualization to')
        self.add_argument('--version', action='version',
                          version='ramutils version {}'.format(__version__))
        self.add_argument('--sessions', '-S', nargs='+',
                          help='sessions to read data from (default: use all)')
        self.add_argument('--debug', '-D', action='store_true',
                          help='Run in debug mode')

        # Number of args, type, and required flag are different so it is easier to do set them up
        # completely separately
        if agg:
            self.add_argument('--subject', '-s', nargs='+',
                              help='List of subjects')
            self.add_argument('--experiment', '-x', nargs='+',
                              help='List of experiments')

        else:
            self.add_argument('--subject', '-s', required=True,
                              type=str, help='subject ID')
            self.add_argument('--experiment', '-x', required=True, type=str,
                              choices=allowed_experiments, help='experiment')

    def _create_dirs(self, path):
        if os.path.exists(path):
            if os.path.isfile(path):
                raise RuntimeError(
                    "{} is a file but must be a directory".format(path))
        else:
            try:
                os.makedirs(path)
            except OSError:
                pass

    @staticmethod
    def _configure_caching(cachedir, use_cached=False):
        """Setup task caching.

        Parameters
        ----------
        cachedir : str
            Location to cache task outputs to.
        use_cached : bool
            Use cached results from previous runs

        Returns
        -------
        Configured caching object.

        """
        from ramutils.tasks import memory

        try:
            memory.store_backend.location = cachedir
        except AttributeError: # joblib v0.11-
            memory.cachedir = cachedir

        if not use_cached and os.path.isdir(cachedir):
            memory.clear()

        return memory

    def parse_args(self, args=None, namespace=None):
        args = super(RamArgumentParser, self).parse_args(args, namespace)
        self._create_dirs(args.dest)
        self._create_dirs(args.cachedir)
        self._configure_caching(args.cachedir, args.use_cached)
        return args


def make_parser(description, agg=False, allowed_experiments=sum([exps for exps in EXPERIMENTS.values()], [])):
    """Create a stub parser containing common options.

    Parameters
    ----------
    description : str
        Passed along to :class:`ArgumentParser`
    agg: bool
        If True, then subject and experiment can be lists instead of single items
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
                             agg=agg,
                             allowed_experiments=allowed_experiments)
