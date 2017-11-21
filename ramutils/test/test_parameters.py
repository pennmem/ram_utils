import functools
import os.path
from ramutils.parameters import FilePaths


class TestFilePaths:
    def test_getattr(self):
        kwargs = dict(
            root='/root',
            dest='dest',
            electrode_config_file='/tmp/ec',
            pairs='/tmp/pairs',
            excluded_pairs='/tmp/excluded_pairs'
        )
        paths = FilePaths(**kwargs)

        assert paths.root == kwargs['root']

        rjoin = functools.partial(os.path.join, paths.root)
        assert paths.dest == rjoin(kwargs['dest'].lstrip('/'))
        assert paths.electrode_config_file == rjoin(kwargs['electrode_config_file'].lstrip('/'))
        assert paths.pairs == rjoin(kwargs['pairs'].lstrip('/'))
        assert paths.excluded_pairs == rjoin(kwargs['excluded_pairs'].lstrip('/'))
