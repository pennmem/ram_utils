import functools
import random
import pytest

from pkg_resources import resource_filename

from ramutils.events import *
from sklearn.externals import joblib
from ramutils.parameters import FRParameters

datafile = functools.partial(resource_filename, 'ramutils.test.test_data')


class TestEvents:
    @classmethod
    def setup_class(cls):
        cls.subject = 'R1354E'
        cls.rootdir = datafile('')
        cls.n_word, cls.n_rec_word, cls.n_rec_base = (
            [random.randint(1, 10) for _ in range(3)])

        data = [('FR1', 0, -1, 'WORD', 1000 + t, 0, 0, True) for t in range(
            cls.n_word)]
        data += [('FR1', 0, 0, 'REC_WORD', 1000 + t + cls.n_word, 0, -1, False) for
                 t in range(cls.n_rec_word)]
        data += [('FR1', 0, 0, 'REC_BASE', 1000 + t + cls.n_word +
                  cls.n_rec_word, 0, 0, False) for t in range(cls.n_rec_base)]

        dtype = [
            ('experiment', '|U256'),
            ('session', '<i8'),
            ('list',  '<i8'),
            ('type', '|U256'),
            ('mstime', '<i8'),
            ('intrusion', '<i8'),
            ('eegoffset', '<i8'),
            ('recalled', '?')
        ]

        cls.test_data = np.rec.array(np.array(data, dtype=dtype))

    @pytest.mark.parametrize("subject, experiment, sessions", [
        ('R1354E', 'FR1', None),
        ('R1354E', 'FR1', [1]),
        ('R1354E', 'catFR1', None),
        ('R1354E', 'catFR1', [101])
    ])
    def test_load_events(self, subject, experiment, sessions):
        events = load_events(subject, experiment, sessions=sessions,
                             rootdir=self.rootdir)
        n_sessions = len(np.unique(events.session))

        assert len(events) > 0

        if sessions is None:
            if experiment == 'FR1':
                assert n_sessions == 2
            elif experiment == 'catFR1':
                assert n_sessions == 4
        else:
            assert n_sessions == 1

        return

    def test_concatenate_events_for_single_experiment(self):
        fr_events = load_events(self.subject, 'FR1', rootdir=self.rootdir)

        combined_events = concatenate_events_for_single_experiment([fr_events,
                                                                    fr_events])
        assert combined_events.shape == (2*len(fr_events),)
        return

    def test_extract_experiment(self):
        experiment = extract_experiment_from_events(self.test_data)
        assert len(experiment) == 1
        assert 'FR1' in experiment
        assert 'catFR1' not in experiment
        return

    def test_concatenate_events_across_experiments(self):
        fr_events = load_events(self.subject, 'FR1', rootdir=self.rootdir)
        catfr_events = load_events(
            self.subject, 'catFR1', rootdir=self.rootdir)

        combined_events = concatenate_events_across_experiments([fr_events,
                                                                 catfr_events])
        assert combined_events.shape == (8363,)

        unique_sessions = np.unique(combined_events.session)
        assert [sess_num in unique_sessions for sess_num in [0, 1, 100, 101]]

        # Check that sessions were updated correctly when combining events
        assert 0 not in combined_events[combined_events.experiment ==
                                        'catFR1'].session
        assert 1 not in combined_events[combined_events.experiment ==
                                        'catFR1'].session
        assert 100 in combined_events[combined_events.experiment ==
                                      'catFR1'].session
        assert 101 in combined_events[combined_events.experiment ==
                                      'catFR1'].session
        assert 0 in combined_events[combined_events.experiment ==
                                    'FR1'].session
        assert 1 in combined_events[combined_events.experiment ==
                                    'FR1'].session
        assert 100 not in combined_events[combined_events.experiment ==
                                          'FR1'].session
        assert 101 not in combined_events[combined_events.experiment ==
                                          'FR1'].session

        return

    @pytest.mark.parametrize('encoding_only', [True, False])
    def test_select_word_events(self, encoding_only):
        word_events = select_word_events(self.test_data, encoding_only)

        # No valid retrieval events will be found because time between events
        # is explicitly made to be 1ms
        if encoding_only:
            assert len(word_events) == self.n_word
        else:
            assert len(word_events) == (self.n_word +
                                        self.n_rec_base + self.n_rec_word)

        return

    def test_find_free_time_periods(self):
        # TODO: This is a more complicated algorithm to test
        return

    def test_match_baseline_retrieval_events(self,rhino_root):
        events = np.rec.array(
            np.load(datafile("input/events/R1409D_pre_baseline_event_insertion_events.npy")))
        events['eegfile'] = [os.path.join(rhino_root,ev['eegfile'])
                             for ev in events]
        params = FRParameters()
        final_events = insert_baseline_retrieval_events(
            events,
            params.baseline_removal_start_time,
            params.retrieval_time,
            params.empty_epoch_duration,
            params.pre_event_buf,
            params.post_event_buf
        )
        retrieval_mask = get_all_retrieval_events_mask(final_events)
        assert (final_events[retrieval_mask].recalled.sum() ==
                (final_events[retrieval_mask].recalled==0).sum())

    @pytest.mark.rhino
    def test_insert_baseline_retrieval_events(self,rhino_root):
        # This is just a regression test. There should be something more
        # comprehensive. This does not look like it would be using rhino, but
        # under the hood a sample of eeg data is loaded to determine the sample
        # rate
        events = np.rec.array(
            np.load(datafile("input/events/R1409D_pre_baseline_event_insertion_events.npy")))
        events['eegfile'] = [os.path.join(rhino_root,ev['eegfile'])
                             for ev in events]
        params = FRParameters()
        final_events = insert_baseline_retrieval_events(events,
                                                        params.baseline_removal_start_time,
                                                        params.retrieval_time,
                                                        params.empty_epoch_duration,
                                                        params.pre_event_buf,
                                                        params.post_event_buf)
        expected_events = np.rec.array(
            np.load(datafile("input/events/R1409D_post_retrieval_baseline_event_insertion_events.npy")))

        assert len(final_events) == len(expected_events)

        return

    @pytest.mark.rhino
    @pytest.mark.parametrize("subject,experiment,session,samplerate", [
        ('R1409D', 'FR1', '1', 1000),
        ('R1001P', 'FR1', '0', 500)
    ])
    def test_lookup_sample_rates(self, subject, experiment, session, samplerate,
                                 rhino_root):
        actual_samplerate = lookup_sample_rate(subject, experiment, session,
                                               rootdir=rhino_root)
        assert actual_samplerate == samplerate

    def test_remove_incomplete_lists(self):
        # TODO: There are two ways to do this. In addition to unit tests,
        # we need to identify the most robust way of doing this procedure
        return

    @pytest.mark.rhino
    def test_extract_sample_rate(self):
        return

    def test_update_pal_retrieval_events(self):
        # Requires creating some sample PAL events
        return

    def test_get_pal_retrieval_events_mask(self):
        # Need some sample PAL events for this test
        return

    def test_get_fr_retrieval_events_mask(self):
        retrieval_event_mask = get_fr_retrieval_events_mask(self.test_data)
        assert max(retrieval_event_mask) == False
        return

    def test_remove_negative_offsets(self):
        cleaned_events = remove_negative_offsets(self.test_data)
        assert len(cleaned_events) == (len(self.test_data) - self.n_rec_word)
        return

    def test_remove_practice_lists(self):
        cleaned_events = remove_practice_lists(self.test_data)
        assert len(cleaned_events) == (len(self.test_data) - self.n_word)
        return

    def test_get_time_between_events(self):
        time_between_events = get_time_between_events(self.test_data)
        # By construction, all test events are 1 ms apart, except first event
        # which is 0ms away from itself
        assert all(time_between_events == (np.append(
            [0], np.ones(len(self.test_data) - 1))))
        return

    def test_select_encoding_events(self):
        encoding_events = select_encoding_events(self.test_data)
        assert len(encoding_events) == self.n_word
        return

    def test_select_baseline_retrieval_events(self):
        baseline_retrieval_events = select_baseline_retrieval_events(
            self.test_data)
        assert len(baseline_retrieval_events) == self.n_rec_base

        # Check that calling this function on events with no baseline retrieval
        # rases a runtime error
        data = [('WORD', t * 1001, 0, 0) for t in range(5)]
        dtype = [
            ('type', '|U256'),
            ('mstime', '<i8'),
            ('intrusion', '<i8'),
            ('eegoffset', '<i8')
        ]
        no_baseline_retrieval_events = np.rec.array(
            np.array(data, dtype=dtype))
        try:
            baseline_retrieval_events = select_baseline_retrieval_events(
                no_baseline_retrieval_events)
        except RuntimeError:
            pass

        return

    def test_select_all_retrieval_events(self):
        all_retrieval_events = select_all_retrieval_events(self.test_data)
        assert len(all_retrieval_events) == self.n_rec_base + self.n_rec_word
        return

    def test_select_retrieval_events(self):
        retrieval_events = select_retrieval_events(self.test_data)
        # By construction, no events are more than 1000ms apart, so this should
        # return no events
        assert len(retrieval_events) == 0
        return

    def test_select_vocalization_events(self):
        vocalization_events = select_vocalization_events(self.test_data)
        assert len(vocalization_events) == self.n_rec_word
        return

    def test_select_session_events(self):
        return

    def test_get_session_mask(self):
        return

    def test_extract_lists(self):
        return

    def test_get_recall_events_mask(self):
        recall_mask = get_recall_events_mask(self.test_data)
        assert sum(recall_mask) == self.n_word
        return

    def test_append_field(self):
        appended_data = add_field(self.test_data, 'item_name', 'X', '<U256')
        assert 'item_name' in appended_data.dtype.names
        assert appended_data['item_name'].dtype.str == '<U256'
        return

    # Four possible partitions. Be sure to check all
    def test_partition_events(self):
        dtypes = [
            ('experiment', '|U256'),
            ('type', '|U256'),
            ('list', '<i8')
        ]
        test_fr_encoding = np.rec.array(
            np.array([('FR1', 'WORD', 1)], dtype=dtypes))
        test_fr_retrieval = np.rec.array(np.array([('FR1', 'REC_EVENT', 1)],
                                                  dtype=dtypes))
        test_pal_encoding = np.rec.array(np.array([('PAL1', 'WORD', 1)],
                                                  dtype=dtypes))
        test_pal_retrieval = np.rec.array(np.array([('PAL1', 'REC_EVENT', 1)],
                                                   dtype=dtypes))

        for subset in [test_fr_encoding, test_fr_retrieval,
                       test_pal_encoding, test_pal_retrieval]:
            partitions = partition_events(subset)
            combined_event_length = sum([len(v)
                                         for k, v in partitions.items()])
            assert combined_event_length == 1
            assert len(partitions['post_stim']) == 0

        encoding_retrieval_partitions = partition_events(
            np.rec.array(np.concatenate([test_fr_encoding, test_fr_retrieval])))
        assert sum([len(v)
                    for k, v in encoding_retrieval_partitions.items()]) == 2

        pal_fr_partitions = partition_events(np.rec.array(np.concatenate([
            test_fr_encoding, test_pal_encoding])))
        assert sum([len(v) for k, v in pal_fr_partitions.items()]) == 2

        pal_fr_encoding_retrieval_partitions = partition_events(
            np.rec.array(np.concatenate([test_fr_encoding, test_fr_retrieval,
                                         test_pal_encoding])))
        assert sum([len(v)
                    for k, v in pal_fr_encoding_retrieval_partitions.items()]) == 3

        all_partitions = partition_events(np.rec.array(np.concatenate([
            test_fr_encoding, test_fr_retrieval, test_pal_encoding,
            test_pal_retrieval])))
        assert sum([len(v) for k, v in all_partitions.items()]) == 4

        return

    def test_extract_stim_and_post_stim_masks(self):
        # TODO: Fill in with comparison to legacy outputs
        return

    @pytest.mark.parametrize("experiment, session_list, exp_sessions", [
        ('FR1', [0, 1, 100, 102, 203], [0, 1]),
        ('catfr1', [0, 1, 100, 102, 203], [0, 2]),
        ('PAL1', [0, 1, 100, 102, 203], [3]),
        ('PS', [0, 1, 100, 102, 203], [0, 1])
    ])
    def test_remove_session_number_offsets(self, experiment, session_list, exp_sessions):
        extracted_sessions = remove_session_number_offsets(
            experiment, session_list)
        assert extracted_sessions == exp_sessions
        return

    @pytest.mark.rhino
    @pytest.mark.slow
    def test_get_repetition_ratios_dict(self, rhino_root):
        # Note: If data for any subject stored in the cached repetition
        # ratios changes, this could result in the test failing. It would be
        # better to just calculate the dict for a few subjects whose data we
        # have cached in the test data directory
        current_repetitions_dict = get_repetition_ratio_dict(
            rootdir=rhino_root)

        cached_reptitions_dict = joblib.load(datafile(
            '/input/events/repetition_ratios.pkl'))

        for subject, ratios in cached_reptitions_dict.items():
            # Only check a few older subjects so we know new data hasn't been
            # added that will affect the test
            if subject in ['R1204T', 'R1343M', 'R1330D']:
                current = np.nan_to_num(current_repetitions_dict[subject])
                old = np.nan_to_num(ratios)
                assert np.allclose(current, old)
