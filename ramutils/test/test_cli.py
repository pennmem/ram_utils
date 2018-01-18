from contextlib import contextmanager
from copy import deepcopy
import os

import pytest

from ramutils.cli import *
from ramutils.cli.expconf import *
from ramutils.cli.report import *


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
    @pytest.mark.output
    @pytest.mark.parametrize(
        'experiment,subject,postfix,anodes,cathodes',
        [
            ('AmplitudeDetermination', 'R1364C', '06NOV2017L0M0STIM', ['AMY7', 'TOJ7'], ['AMY8', 'TOJ8']),
            ('CatFR5', 'R1364C', '06NOV2017L0M0STIM', ['AMY7'], ['AMY8']),
            ('FR6', 'R1364C', '06NOV2017L0M0STIM', ['AMY7', 'TOJ7'], ['AMY8', 'TOJ8']),
            ('PS4_FR5', 'R1364C', '06NOV2017L0M0STIM', ['AMY7', 'TOJ7'], ['AMY8', 'TOJ8']),
            ('PAL5', 'R1318N', 'R1318N11JUL17M0L0STIM', ['LAIIH2'], ['LAIIH3'])
        ]
    )
    def test_create_expconf(self, experiment, subject, postfix, anodes,
                            cathodes, rhino_root, output_dest):

        args = [
            "-s", subject, "-x", experiment,
            "-e",
            "scratch/system3_configs/ODIN_configs/{subject:s}/{subject:s}_{postfix:s}.csv".format(
                subject=subject, postfix=postfix),
            "--target-amplitudes", "0.5", "0.75",
            "--root", rhino_root, "--dest", output_dest, "--force-rerun"
        ]

        args += ["--anodes"] + anodes
        args += ["--cathodes"] + cathodes

        if experiment != 'AmplitudeDetermination' and 'PS4' not in experiment:
            args += ['--target-amplitudes'] + ['0.5'] * len(anodes)
        else:
            args += ['--min-amplitudes'] + ['0.1'] * len(anodes)
            args += ['--max-amplitudes'] + ['1.0'] * len(anodes)

        create_expconf(args)


class TestCreateReports:
    @pytest.mark.rhino
    @pytest.mark.slow
    @pytest.mark.output
    @pytest.mark.parametrize('subject, experiment, sessions, joint', [
        ('R1001P', 'FR1', None, False),
        ('R1354E', 'FR1', [0], False),
        ('R1354E', 'FR1', [0, 1], False),
        ('R1354E', 'CatFR1', [0], False),
        ('R1354E', 'FR1', None, True),
        ('R1345D', 'FR1', None, False),
        ('R1374T', 'CatFR1', None, False),
        ('R1374T', 'CatFR1', None, True)
    ])
    def test_create_open_loop_report(self, subject, experiment, sessions, joint,
                                     rhino_root, output_dest):
        args = [
            '--root', rhino_root,
            '--dest', output_dest,
            '-s', subject,
            '-x', experiment,
        ]

        if joint:
            args += ['-j']

        if sessions is not None:
            args += ['-S'] + [str(session) for session in sessions]

        create_report(args)
        return

    @pytest.mark.rhino
    @pytest.mark.output
    @pytest.mark.parametrize('subject, experiment, sessions', [
        ('R1374T', 'CatFR5', [0]),
        ('R1345D', 'FR5', [0]),
        ('R1374T', 'PS4_CatFR5', None)
    ])
    def test_create_stim_session_report(self, subject, experiment, sessions,
                                        rhino_root, output_dest):

        args = [
            '--root', rhino_root,
            '--dest', output_dest,
            '-s', subject,
            '-x', experiment
        ]

        if sessions is not None:
            args += ['-S'] + sessions

        create_report(args)
        return
