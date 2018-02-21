import os
import pytest
import functools
import numpy as np

from pkg_resources import resource_filename

from ramutils.tasks import memory
from ramutils.utils import load_event_test_data
from ramutils.tasks.events import *
from ramutils.parameters import FRParameters, PALParameters, FilePaths


datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data.input.events')


class TestEvents:
    @classmethod
    def teardown_class(cls):
        memory.clear(warn=False)

    @pytest.mark.rhino
    @pytest.mark.parametrize('subject, experiment, params', [
        ('R1354E', 'FR1', FRParameters),
        ('R1350D', 'FR1', FRParameters),
        ('R1353N', 'PAL1', PALParameters),
        ('R1348J', 'catFR1', FRParameters)
    ])
    def test_training_legacy_regression(self, subject, experiment, params,
                                        rhino_root):
        expected = datafile("/{}_task_events_rhino.npy".format(subject))
        expected_events = load_event_test_data(expected, rhino_root)

        paths = FilePaths(root=rhino_root)
        extra_kwargs = params().to_dict()
        actual_events = build_training_data(subject, experiment, paths,
                                            **extra_kwargs).compute()
        assert len(actual_events) == len(expected_events)
        assert np.array_equal(actual_events.recalled,
                              expected_events.recalled)

    @pytest.mark.rhino
    @pytest.mark.parametrize('subject, experiment, params, combine_events', [
        ('R1350D', 'FR1', FRParameters, True),  # multi-session FR
        ('R1353N', 'PAL1', PALParameters, True),  # pal
        ('R1354E', 'FR1', FRParameters, True), # fr and catfr combined
        ('R1354E', 'FR1', FRParameters, False) # only FR when catFR available
    ])
    def test_build_training_data_regression(self, subject, experiment,
                                            params, combine_events, rhino_root):

        expected = datafile('/train/{}_{}_combined_training_events.npy'.format(
            subject, experiment))

        if not combine_events:
            expected = expected.replace("_combined", "")

        paths = FilePaths(root=rhino_root)
        extra_kwargs = params().to_dict()

        current_events = build_training_data(subject, experiment, paths,
                                             **extra_kwargs).compute()

        expected_events = load_event_test_data(expected, rhino_root)
        assert len(current_events) == len(expected_events)
        assert np.array_equal(current_events.recalled, expected_events.recalled)

        return

    @pytest.mark.rhino
    @pytest.mark.parametrize('subject, experiment, params, joint_report, '
                             'sessions', [
        ('R1350D', 'FR1', FRParameters, True, None), # multi-session FR
        ('R1354E', 'FR1', FRParameters, True, None), # fr and catfr combined
        ('R1354E', 'FR1', FRParameters, False, None), # FR1 only
        ('R1354E', 'catFR1', FRParameters, False, [0]), # catFR1 only
        ('R1353N', 'PAL1', PALParameters, True, [0]), # PAL1 only
        ('R1345D', 'FR5', FRParameters, False, [0]), # FR5 only
    ])
    def test_build_test_data_regression(self, subject, experiment, params,
                                        joint_report, sessions, rhino_root):
        expected = datafile('/test/{}_{}_combined_test_events.npy'.format(
            subject, experiment))

        if not joint_report:
            expected = expected.replace("_combined", "")

        assert os.path.exists(expected)

        paths = FilePaths(root=rhino_root)
        extra_kwargs = params().to_dict()

        all_events, task_events, stim_params = build_test_data(
            subject, experiment, paths, joint_report=joint_report,
            sessions=sessions, **extra_kwargs).compute()

        expected_events = load_event_test_data(expected, rhino_root)
        assert len(expected_events) == len(task_events)
        assert np.array_equal(task_events.recalled, expected_events.recalled)

    @pytest.mark.rhino
    def test_build_ps_events(self, rhino_root):
        actual = build_ps_data('R1374T', 'catFR5', 'ps4_events', None,
                               rhino_root).compute()
        assert len(actual) == 3068
