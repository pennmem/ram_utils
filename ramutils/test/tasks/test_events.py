from pkg_resources import resource_filename
import pytest

from ptsa.data.readers import JsonIndexReader

from ramutils.tasks import memory
from ramutils.tasks.events import *
# from ramutils.test import Mock


class TestEvents:
    @classmethod
    def setup_class(cls):
        cls.subject = 'R1354E'
        cls.index = JsonIndexReader(resource_filename('ramutils.test.test_data.protocols', 'r1.json'))

    @classmethod
    def teardown_class(cls):
        memory.clear(warn=False)

    @pytest.mark.parametrize('catfr', [True, False])
    def test_read_fr_events(self, catfr):
        events = read_fr_events(self.index, self.subject, cat=catfr).compute()

        # There are 2 sessions each of FR1 and catFR1
        assert len(events) == 2

    # def test_concatenate_events(self):
    #     pass
    #
    # def test_select_word_events(self):
    #     pass
