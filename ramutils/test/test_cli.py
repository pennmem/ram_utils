from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import os
from socket import gethostname

import pytest

from ramutils.cli import *
from ramutils.cli.expconf import *


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


class TestExpConf:
    @pytest.mark.parametrize(
        "experiment",
        ["AmplitudeDetermination", "PS4_FR5", "CatFR5"]
    )
    def test_validate_stim_settings(self, experiment):
        class Args:
            anodes = ['A1', 'B1']
            cathodes = ['A2', 'B2']
            target_amplitudes = None

            @contextmanager
            def copy(self):
                copied = deepcopy(self)
                yield copied
                del copied

        args = Args()
        setattr(args, 'experiment', experiment)

        # Mismatch in number of anodes/cathodes
        with args.copy() as args2:
            args2.cathodes = ['A2']
            with pytest.raises(ValidationError):
                validate_stim_settings(args2)

        if experiment == 'AmplitudeDetermination':
            with args.copy() as args2:
                # Not enough min/max amplitudes
                setattr(args2, 'min_amplitudes', [0.1])
                setattr(args2, 'max_amplitudes', [1.0])
                with pytest.raises(ValidationError):
                    validate_stim_settings(args2)

                # Sufficient min/max amplitudes
                args2.min_amplitudes = [0.1, 0.1]
                args2.max_amplitudes = [1.0, 1.0]
                validate_stim_settings(args2)
        elif 'PS4' in experiment:
            pass  # FIXME
        else:
            with args.copy() as args2:
                # no target amplitudes given
                with pytest.raises(RuntimeError):
                    validate_stim_settings(args2)

                # invalid number of target amplitudes
                args2.target_amplitudes = [0.5]
                with pytest.raises(ValidationError):
                    validate_stim_settings(args2)

                # correct number of target amplitudes
                args2.target_amplitudes = [0.5, 0.5]
                validate_stim_settings(args2)

    @pytest.mark.rhino
    @pytest.mark.slow
    @pytest.mark.trylast
    @pytest.mark.parametrize(
        'experiment,subject,postfix,anodes,cathodes',
        [
            ('AmplitudeDetermination', 'R1364C', '06NOV2017L0M0STIM', ['AMY7', 'TOJ7'], ['AMY8', 'TOJ8']),
            ('CatFR5', 'R1364C', '06NOV2017L0M0STIM', ['AMY7'], ['AMY8']),
            ('FR6', 'R1364C', '06NOV2017L0M0STIM', ['AMY7', 'TOJ7'], ['AMY8', 'TOJ8']),
            ('PS4_FR5', 'R1364C', '06NOV2017L0M0STIM', ['AMY7', 'TOJ7'], ['AMY8', 'TOJ8']),
            ('PAL5', 'R1365N', '16NOV2017L0M0STIM', ['LAD12'], ['LAD13'])
        ]
    )
    def test_main(self, experiment, subject, postfix, anodes, cathodes):
        rhino_root = defaultdict(lambda: '/Volumes/RHINO')

        # If you put your rhino mount point somewhere other than /Volumes/RHINO,
        # put it here.
        rhino_root['krieger'] = os.path.expanduser('~/mnt/rhino')
        rhino_root['rhino2'] = '/'

        root = rhino_root[gethostname().split('.')[0]]

        args = [
            "-s", subject, "-x", experiment,
            "-e",
            "scratch/system3_configs/ODIN_configs/{subject:s}/{subject:s}_{postfix:s}.csv".format(
                subject=subject, postfix=postfix),
            "--target-amplitudes", "0.5", "0.75",
            "--root", root, "--dest", "scratch/ramutils2/tests", "--force-rerun"
        ]

        args += ["--anodes"] + anodes
        args += ["--cathodes"] + cathodes

        if experiment != 'AmplitudeDetermination' and 'PS4' not in experiment:
            args += ['--target-amplitudes'] + ['0.5'] * len(anodes)
        else:
            args += ['--min-amplitudes'] + ['0.1'] * len(anodes)
            args += ['--max-amplitudes'] + ['1.0'] * len(anodes)

        main(args)
