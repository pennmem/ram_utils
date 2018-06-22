from contextlib import contextmanager
from copy import deepcopy

import pytest

from ramutils.cli import *
from ramutils.cli.expconf import *
from ramutils.cli.report import *
from ramutils.cli.aggregated_report import *


def test_make_parser():
    parser = make_parser('test')
    args = parser.parse_args(['-s', 'R0000M', '-d', '.', '-x', 'FR1'])
    assert args.subject == 'R0000M'
    assert args.dest == '.'
    assert args.experiment == 'FR1'


@pytest.mark.parametrize('use_cached', [True, False])
def test_configure_caching(use_cached, tmpdir):
    from ramutils.tasks import memory

    path = str(tmpdir)
    RamArgumentParser._configure_caching(path)

    @memory.cache
    def foo():
        return "bar"

    # put something in the cache dir
    foo()

    assert len(os.listdir(path))

    # Re-configure, possibly clearing
    RamArgumentParser._configure_caching(path, use_cached)

    if use_cached:
        assert len(os.listdir(path))
    else:
        assert not len(os.listdir(path))


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

    @pytest.mark.expconf
    @pytest.mark.rhino
    @pytest.mark.slow
    @pytest.mark.trylast
    @pytest.mark.output
    @pytest.mark.parametrize(
        'experiment,subject,postfix,anodes,cathodes,use_classifier_excluded_leads, sessions',
        [
            ('AmplitudeDetermination', 'R1364C', '06NOV2017L0M0STIM', ['AMY7', 'TOJ7'], ['AMY8', 'TOJ8'], False, None),
            ('TICL_FR', 'R1364C', '06NOV2017L0M0STIM', ['AMY7'], ['AMY8'], False, None),
            ('TICL_CatFR', 'R1364C', '06NOV2017L0M0STIM', ['AMY7'], ['AMY8'], False, None),
            ('CatFR5', 'R1364C', '06NOV2017L0M0STIM', ['AMY7'], ['AMY8'], False, None),
            ('CatFR5', 'R1364C', '06NOV2017L0M0STIM', ['AMY7'], ['AMY8'], True, [0, 1, 100]), # limited sessions config
            ('FR6', 'R1364C', '06NOV2017L0M0STIM', ['AMY7', 'TOJ7'], ['AMY8', 'TOJ8'], False, None),
            ('PS4_FR5', 'R1364C', '06NOV2017L0M0STIM', ['AMY7', 'TOJ7'], ['AMY8', 'TOJ8'], False, None),
            ('PS4_CatFR5', 'R1364C', '06NOV2017L0M0STIM', ['AMY7', 'TOJ7'], ['AMY8', 'TOJ8'], False, None),
            ('PAL5', 'R1318N', 'R1318N11JUL17M0L0STIM', ['LAIIH2'], ['LAIIH3'], False, None),
            ('PS5_FR', 'R1378T', '18DEC2017L0M0STIM', ['LC8'], ['LC9'], False, None)
        ]
    )
    def test_create_expconf(self, experiment, subject, postfix, anodes,
                            cathodes, use_classifier_excluded_leads, sessions, rhino_root, output_dest):

        args = [
            "-s", subject, "-x", experiment,
            "-e",
            "scratch/system3_configs/ODIN_configs/{subject:s}/{subject:s}_{postfix:s}.csv".format(
                subject=subject, postfix=postfix),
            "--target-amplitudes", "0.5", "0.75",
            "--root", rhino_root, "--dest", output_dest,
        ]

        args += ["--anodes"] + anodes
        args += ["--cathodes"] + cathodes

        if experiment == 'PS5_FR':
            args += ['--trigger-pairs'] + ['LX15_LX16'] + ['LT8_LT9']

        if experiment != 'AmplitudeDetermination' and 'PS' not in experiment:
            args += ['--target-amplitudes'] + ['0.5'] * len(anodes)

        else:
            args += ['--min-amplitudes'] + ['0.1'] * len(anodes)
            args += ['--max-amplitudes'] + ['1.0'] * len(anodes)

        if use_classifier_excluded_leads:
            args += ['-u']

        if sessions is not None:
            args += ['-S'] + [str(session) for session in sessions]

        create_expconf(args)

    @pytest.mark.rhino
    @pytest.mark.parametrize("subject,anodes,cathodes,target_amplitudes,xfail", [
        ("R1384J", ["LF7", "LS8"], ["LF8", "LS9"], [0.5, 0.5], False),
        ("R1384J", ["{}"] * 7, ["{}"] * 7, [0.5] * 7, True)
    ])
    def test_create_expconf_locsearch(self, subject, anodes, cathodes,
                                      target_amplitudes, xfail, rhino_root,
                                      output_dest):
        args = [
            "-s", subject,
            "-x", "LocationSearch",
            "--root", rhino_root,
            "-d", output_dest,
        ]
        args += ["--anodes"] + anodes
        args += ["--cathodes"] + cathodes
        args += ["--target-amplitudes"] + [str(ampl) for ampl in target_amplitudes]
        args += ["--default-area", "5"]

        if xfail:
            with pytest.raises(ValidationError):
                create_expconf(args)
        else:
            create_expconf(args)


class TestCreateReports:
    @pytest.mark.rhino
    @pytest.mark.slow
    @pytest.mark.output
    @pytest.mark.parametrize('rerun', [True, False])
    @pytest.mark.parametrize('subject, experiment, sessions, joint, use_classifier_excluded_leads', [
        ('R1001P', 'FR1', None, False, False),
        ('R1354E', 'FR1', [0], False, False),
        ('R1354E', 'FR1', [0, 1], False, False),
        ('R1354E', 'CatFR1', [1], False, False),
        # test case for cross-experiment session subset
        ('R1354E', 'FR1', [0, 100], True, False),
        ('R1354E', 'FR1', None, True, False),
        ('R1345D', 'FR1', None, False, False),
        ('R1374T', 'CatFR1', None, False, False),
        ('R1374T', 'CatFR1', None, True, False),
        # Test case for re-localized subject
        ('R1394E_1', 'FR1', None, True, False),
        # use excluded classifier leads test
        ('R1364C', 'FR1', [0], False, True)
    ])
    def test_create_open_loop_report(self, subject, experiment, sessions,
                                     joint, use_classifier_excluded_leads, rerun, rhino_root,
                                     output_dest):
        args = [
            '--root', rhino_root,
            '--dest', output_dest,
            '-s', subject,
            '-x', experiment,
            '--report_db_location', output_dest
        ]

        if rerun is True:
            args += ['--rerun']

        if joint:
            args += ['-j']

        if sessions is not None:
            args += ['-S'] + [str(session) for session in sessions]

        if use_classifier_excluded_leads:
            args += ['-u']

        create_report(args)
        return

    @pytest.mark.rhino
    @pytest.mark.output
    @pytest.mark.parametrize('rerun', [True, False])
    @pytest.mark.parametrize('subject, experiment, sessions', [
        ('R1111M', 'FR2', [0]),
        ('R1154D', 'FR3', [0]),
        ('R1374T', 'CatFR5', [0]),
        ('R1345D', 'FR5', [0]),
        ('R1364C', 'PS4_FR', [1]),
        ('R1001P', 'FR2', [0]),
        ('R1384J', 'PS5_CatFR', [0])
    ])
    def test_create_stim_session_report(self, subject, experiment, sessions,
                                        rerun, rhino_root, output_dest):

        args = [
            '--root', rhino_root,
            '--dest', output_dest,
            '-s', subject,
            '-x', experiment,
            '--report_db_location', output_dest
        ]

        if rerun is True:
            args += ['--rerun']

        if sessions is not None:
            args += ['-S'] + [str(session) for session in sessions]

        if experiment == 'PS5_CatFR':
            args += ['--trigger-electrode', 'LF1-LF2']

        create_report(args)
        return

    @pytest.mark.rhino
    @pytest.mark.output
    @pytest.mark.parametrize('subjects, experiments, sessions', [
        (['R1384J'], ['FR5'], None),
        (['R1384J'], ['FR5'], [0, 1]),
        (['R1384J'], ['FR5', 'CatFR5'], None),
        (None, ['FR5'], None),
        (['R1384J', 'R1395M'], ['FR5'], None)
    ])
    def test_create_aggregated_stim_session_report(self, subjects, experiments, sessions, rhino_root, output_dest):

        args = [
            '--root', rhino_root,
            '--dest', output_dest,
        ]

        if subjects is not None:
            args += ['-s'] + subjects

        if experiments is not None:
            args += ['-x'] + experiments

        if sessions is not None:
            args += ['-S'] + [str(session) for session in sessions]

        create_aggregate_report(args)
        return
