import functools
import random
import pytest

from pkg_resources import resource_filename
from sklearn.externals import joblib

from ramutils.events import *
from ramutils.parameters import FRParameters, PALParameters

datafile = functools.partial(resource_filename, 'ramutils.test.test_data')


class TestEvents:
    @classmethod
    def setup_class(cls):
        cls.subject = 'R1354E'
        cls.rootdir = datafile('')
        cls.n_word, cls.n_rec_word, cls.n_rec_base = (
            [random.randint(1, 10) for _ in range(3)])

        data = [('FR1', 0, -1, 'WORD', 1000 + t, 0, 0) for t in range(
            cls.n_word)]
        data += [('FR1', 0, 0, 'REC_WORD', 1000 + t + cls.n_word, 0, -1) for
                 t in range(cls.n_rec_word)]
        data += [('FR1', 0, 0, 'REC_BASE', 1000 + t + cls.n_word +
                  cls.n_rec_word, 0, 0) for t in range(cls.n_rec_base)]

        dtype = [
            ('experiment', '|S256'),
            ('session', '<i8'),
            ('list',  '<i8'),
            ('type', '|S256'),
            ('mstime', '<i8'),
            ('intrusion', '<i8'),
            ('eegoffset', '<i8')
        ]

        cls.test_data = np.array(data, dtype=dtype).view(np.recarray)

    @pytest.mark.parametrize("subject, experiment, sessions", [
        ('R1354E', 'FR1', None),
        ('R1354E', 'FR1', [1]),
        ('R1354E', 'catFR1', None),
        ('R1354E', 'catFR1', [1])
    ])
    def test_load_events(self, subject, experiment, sessions):
        events = load_events(subject, experiment, sessions,
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

    def test_concatenate_events_across_experiments(self):
        fr_events = load_events(self.subject, 'FR1', rootdir=self.rootdir)
        catfr_events = load_events(self.subject, 'catFR1', rootdir=self.rootdir)

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
            assert len(word_events) == (self.n_word + self.n_rec_base + self.n_rec_word)

        return

    def test_find_free_time_periods(self):
        # TODO: This is a more complicated algorithm to test
        return

    def test_insert_baseline_retrieval_events(self):
        # TODO: This is another somehwat complicated algorithm to test
        return

    def test_remove_incomplete_lists(self):
        # TODO: There are two ways to do this. In addition to unit tests,
        # we need to identify the most robust way of doing this procedure
        return

    @pytest.mark.skip(reason='rhino')
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
            ('type', '|S256'),
            ('mstime', '<i8'),
            ('intrusion', '<i8'),
            ('eegoffset', '<i8')
        ]
        no_baseline_retrieval_events = np.array(data, dtype=dtype).view(
            np.recarray)
        try:
            baseline_retrieval_events = select_baseline_retrieval_events(no_baseline_retrieval_events)
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

    # Four possible partitions. Be sure to check all
    def test_partition_events(self):
        dtypes = [
            ('experiment', '|S256'),
            ('type', '|S256')
        ]
        test_fr_encoding = np.array([('FR1', 'WORD')], dtype=dtypes).view(
            np.recarray)
        test_fr_retrieval = np.array([('FR1', 'REC_EVENT')],
                                     dtype=dtypes).view(np.recarray)
        test_pal_encoding = np.array([('PAL1', 'WORD')],
                                     dtype=dtypes).view(np.recarray)
        test_pal_retrieval = np.array([('PAL1', 'REC_EVENT')],
                                      dtype=dtypes).view(np.recarray)

        for subset in [test_fr_encoding, test_fr_retrieval,
                       test_pal_encoding, test_pal_retrieval]:
            partitions = partition_events(subset)
            combined_event_length = sum([len(v) for k, v in partitions.items()])
            assert combined_event_length == 1

        encoding_retrieval_partitions = partition_events(np.concatenate([
            test_fr_encoding, test_fr_retrieval]).view(np.recarray))
        assert sum([len(v) for k, v in encoding_retrieval_partitions.items()]) == 2

        pal_fr_partitions = partition_events(np.concatenate([
            test_fr_encoding, test_pal_encoding]).view(np.recarray))
        assert sum([len(v) for k, v in pal_fr_partitions.items()]) == 2

        pal_fr_encoding_retrieval_partitions = partition_events(
            np.concatenate([test_fr_encoding, test_fr_retrieval,
                            test_pal_encoding]).view(np.recarray))
        assert sum([len(v) for k, v in pal_fr_encoding_retrieval_partitions.items()]) == 3

        all_partitions = partition_events(np.concatenate([test_fr_encoding,
                                                          test_fr_retrieval,
                                                          test_pal_encoding,
                                                          test_pal_retrieval]).view(np.recarray))
        assert sum([len(v) for k, v in all_partitions.items()]) == 4

        return

    @pytest.mark.parametrize("subject, experiment, parameters, encoding_only, "
                             "combine_events", [
        ("R1354E", "FR6", FRParameters, False, True),
        ("R1354E", "FR6", FRParameters, True, True),
        ("R1016M", "PAL6", PALParameters, True, True),
        ("R1016M", "PAL6", PALParameters, True, False),
        ("R1016M", "PAL6", PALParameters, False, True),
        ("R1016M", "PAL6", PALParameters, False, False)
    ])
    def test_preprocess_events(self, subject, experiment, parameters,
                               encoding_only, combine_events):
        parameters = parameters().to_dict()
        events = preprocess_events(subject,
                                   experiment,
                                   parameters['baseline_removal_start_time'],
                                   parameters['retrieval_time'],
                                   parameters['empty_epoch_duration'],
                                   parameters['pre_event_buf'],
                                   parameters['post_event_buf'],
                                   encoding_only=encoding_only,
                                   combine_events=combine_events,
                                   root=datafile(''))
        assert len(events) > 0

        if encoding_only:
            assert "REC_EVENT" not in np.unique(events.type)
        else:
            assert "REC_EVENT" in np.unique(events.type)

        if combine_events and subject == "R1016M":
            # There are some experiment fields that are blank, so checking
            # that there are two unique experiment types would fail
            assert len(np.unique(events.experiment)) > 1

        return

    @pytest.mark.parametrize("subject, parameters, experiment", [
        ("R1354E", FRParameters, "FR6"),  # catFR and FR
        ("R1350D", FRParameters, "FR6"),  # FR only
        ("R1348J", FRParameters, "FR6"),  # catFR only
        ("R1353N", PALParameters, "PAL6")  # PAL only
    ])
    def test_regression_event_processing(self, subject, parameters, experiment):
        parameters = parameters().to_dict()

        old_events = joblib.load(datafile('/input/events/{}_events.pkl'.format(
            subject)))
        new_events = preprocess_events(subject,
                                       experiment,
                                       parameters['baseline_removal_start_time'],
                                       parameters['retrieval_time'],
                                       parameters['empty_epoch_duration'],
                                       parameters['pre_event_buf'],
                                       parameters['post_event_buf'],
                                       root=datafile(''))
        assert len(old_events) == len(new_events)

        return

