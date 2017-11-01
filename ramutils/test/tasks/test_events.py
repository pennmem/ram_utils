import functools
import random
from pkg_resources import resource_filename
import pytest

from ptsa.data.readers import JsonIndexReader, BaseEventReader

from ramutils.tasks import memory
from ramutils.tasks.events import *

datafile = functools.partial(resource_filename, 'ramutils.test.test_data.protocols')


class TestEvents:
    @classmethod
    def setup_class(cls):
        cls.subject = 'R1354E'
        cls.index = JsonIndexReader(datafile('r1.json'))

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
        assert concated.shape == (380,)

    @pytest.mark.parametrize('retrieval', [True, False])
    def test_select_word_events(self, retrieval):
        n_word, n_rec_word, n_rec_base = [random.randint(1, 10) for _ in range(3)]

        data = [('WORD', t * 1001, 0) for t in range(n_word)]
        data += [('REC_WORD', t * 1001, 0) for t in range(n_rec_word)]
        data += [('REC_BASE', t * 1001, 0) for t in range(n_rec_base)]

        dtype = [
            ('type', '|S256'),
            ('mstime', '<i8'),
            ('intrusion', '<i8')
        ]

        events = np.array(data, dtype=dtype).view(np.recarray)

        word_events = select_word_events(events, retrieval).compute()

        if retrieval:
            # TODO: understand why -1
            assert len(word_events) == n_word + n_rec_word + n_rec_base - 1
        else:
            assert len(word_events) == n_word
