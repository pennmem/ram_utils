import pytest
import random
import functools
import numpy as np

from pkg_resources import resource_filename

from ptsa.data.readers import JsonIndexReader, BaseEventReader
from ramutils.tasks import memory
from ramutils.tasks.events import *


datafile = functools.partial(resource_filename, 'ramutils.test.test_data.protocols')


class TestEvents:
    @classmethod
    def setup_class(cls):
        cls.subject = 'R1354E'
        cls.index = JsonIndexReader(datafile('r1.json'))
        cls.n_word, cls.n_rec_word, cls.n_rec_base = (
            [random.randint(1, 10) for _ in range(3)])

        data = [(0, 0, 'WORD', 1000 + t, 0, 0) for t in range(cls.n_word)]
        data += [(0, 0, 'REC_WORD', 1000 + t + cls.n_word, 0, -1) for t in
                 range(
            cls.n_rec_word)]
        data += [(0, 0, 'REC_BASE', 1000 + t + cls.n_word + cls.n_rec_word, 0,
                 0) for t in
                 range(cls.n_rec_base)]

        dtype = [
            ('session', '<i8'),
            ('list',  '<i8'),
            ('type', '|S256'),
            ('mstime', '<i8'),
            ('intrusion', '<i8'),
            ('eegoffset', '<i8')
        ]

        cls.test_data = np.array(data, dtype=dtype).view(np.recarray)

    @classmethod
    def teardown_class(cls):
        memory.clear(warn=False)

    @pytest.mark.parametrize('catfr', [True, False])
    def test_read_fr_events(self, catfr):
        events = read_fr_events(self.index, self.subject, cat=catfr).compute()

        # There are 2 sessions each of FR1 and catFR1
        assert len(events) == 2

    def test_concatenate_events(self):
        fr_events = [
            BaseEventReader(filename=datafile('R1354E_FR1_{}.json'.format(n))).read()
            for n in range(2)
        ]
        catfr_events = [
            BaseEventReader(filename=datafile('R1354E_catFR1_{}.json'.format(n))).read()
            for n in range(2)
        ]

        concated = concatenate_events(fr_events, catfr_events).compute()
        assert concated.shape == (5804,)

    def test_combine_events(self):
        combined_events = combine_events([self.test_data, self.test_data])
        assert len(combined_events) == (2 * len(self.test_data))
        return

    @pytest.mark.parametrize('retrieval', [True, False])
    def test_select_word_events(self, retrieval):
        word_events = select_word_events(self.test_data,
                                         retrieval).compute()

        # No valid retrieval events will be found because time between events
        # is explicitly made to be 1ms
        if retrieval:
            assert len(word_events) == (self.n_word + self.n_rec_base)
        else:
            assert len(word_events) == self.n_word

        return

    def test_find_free_time_periods(self):
        return

    def test_insert_baseline_retrieval_events(self):
        return

    def test_remove_incomplete_lists(self):
        return

    def test_remove_negative_offsets(self):
        cleaned_events = remove_negative_offsets(self.test_data)
        assert len(cleaned_events) == (len(self.test_data) - self.n_rec_word)
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
