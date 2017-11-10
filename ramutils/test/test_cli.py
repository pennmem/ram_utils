import os
import pytest
from ramutils.cli import *


def test_make_parser():
    parser = make_parser('test')
    args = parser.parse_args(['-s', 'R0000M', '-d', '.', '-x', 'FR1'])
    assert args.subject == 'R0000M'
    assert args.dest == '.'
    assert args.experiment == 'FR1'


@pytest.mark.parametrize('invalidate', [True, False])
def test_configure_caching(invalidate, tmpdir):
    from ramutils.tasks import memory

    path = str(tmpdir)
    configure_caching(path)

    @memory.cache
    def foo():
        return "bar"

    # put something in the cache dir
    foo()

    assert len(os.listdir(path))

    # Re-configure, possibly clearing
    configure_caching(path, invalidate)

    if invalidate:
        assert not len(os.listdir(path))
    else:
        assert len(os.listdir(path))
