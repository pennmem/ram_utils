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
            assert n_sessions == 2 # 2 sessions of FR and catFR
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
        assert combined_events.shape == (6053,)

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
        return

    def test_insert_baseline_retrieval_events(self):
        return

    def test_remove_incomplete_lists(self):
        return

    def test_extract_sample_rate(self):
        return

    def test_update_pal_retrieval_events(self):
        return

    def test_get_pal_retrieval_events_mask(self):
        return

    def test_get_fr_retrieval_events_mask(self):
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
        test_fr_encoding = np.array(['FR1', 'WORD'], dtype=dtypes).view(
            np.recarray)
        test_fr_retrieval = np.array(['FR1', 'REC_EVENT'],
                                     dtype=dtypes).view(np.recarray)
        test_pal_encoding = np.array(['PAL1', 'WORD'],
                                     dtype=dtypes).view(np.recarray)
        test_pal_retrieval = np.array(['PAL1', 'REC_EVENT'],
                                      dtype=dtypes).view(np.recarray)

        for subset in [test_fr_encoding, test_fr_retrieval,
                       test_pal_encoding, test_pal_retrieval]:
            # TODO add assertion here
            pass


        return

    @pytest.mark.parametrize("experiment, encoding_only, combine_events", [
        ("FR6", False, True),
        ("FR6", True, True),
        ("PAL6", True, True),
        ("PAL6", True, False),
        ("PAL6", False, True),
        ("PAL6", False, False)
    ])
    def test_preprocess_events(self, experiment, encoding_only, combine_events):
        return

    @pytest.mark.skip(reason='rhino')
    @pytest.mark.parametrize("subject, parameters, experiment", [
        ("R1354E", FRParameters, "FR6"), # catFR and FR
        ("R1350D", FRParameters, "FR6"), # FR only
        ("R1348J", FRParameters, "FR6"), # catFR only
        ("R1353N", PALParameters, "PAL6") # PAL only
    ])
    def test_regression_event_processing(self, subject, parameters, experiment):
        parameters = parameters().to_dict()

        old_events = joblib.load(datafile('{}_events.pkl'.format(subject)))
        new_events = preprocess_events(subject,
                                       experiment,
                                       parameters['baseline_removal_start_time'],
                                       parameters['retrieval_time'],
                                       parameters['empty_epoch_duration'],
                                       parameters['pre_event_buf'],
                                       parameters['post_event_buf'],
                                       root='/Volumes/RHINO/')
        assert len(old_events) == len(new_events)

        return

