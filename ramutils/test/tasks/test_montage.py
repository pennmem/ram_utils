import json
import pytest
import functools
import numpy as np

from pkg_resources import resource_filename

from classiflib import dtypes
from ramutils.tasks.montage import *
from ramutils.tasks import memory
from ramutils.parameters import StimParameters


datafile = functools.partial(resource_filename, 'ramutils.test.test_data')


class TestMontage:
    @classmethod
    def setup_class(cls):
        cls.test_pairs_str = '{"test_subject": {' \
                             '  "pairs": {' \
                             '      "LAD1-LAD2": {' \
                             '          "channel_1": 9,' \
                             '          "channel_2": 10' \
                             '      },' \
                             '      "LAD3-LAD4": {' \
                             '          "channel_1": 11,' \
                             '          "channel_2": 12' \
                             '      }' \
                             '    }' \
                             '  }' \
                             '}'
        cls.test_pairs = json.loads(cls.test_pairs_str)
        cls.test_pairs_recarray = np.rec.fromrecords([
            ("9", "10", "LAD1", "LAD2"),
            ("11", "12", "LAD3", "LAD4")
        ], dtype=dtypes.pairs)

        cls.stim_params = [StimParameters(label='LAD1-LAD2',
                                          anode=9,
                                          cathode=10)
                           ]

    @classmethod
    def teardown_class(cls):
        memory.clear(warn=False)

    def test_load_pairs(self):
        filename = datafile('R1328E_pairs.json')
        pairs = load_pairs_from_json(filename).compute()

        with open(filename) as f:
            data = json.load(f)

        assert 'R1328E' in data
        assert 'R1328E' in pairs
        assert 'pairs' in data['R1328E']
        assert 'pairs' in pairs['R1328E']
        return

    def test_extract_pairs_dict(self):
        no_pairs_str = "{}"
        no_pairs = json.loads(no_pairs_str)
        assert extract_pairs_dict(no_pairs) == {}

        extracted_pairs = extract_pairs_dict(self.test_pairs)
        assert len(extracted_pairs.keys()) == 2
        assert "LAD1-LAD2" in extracted_pairs.keys()
        assert "LAD3-LAD4" in extracted_pairs.keys()

        return

    @pytest.mark.parametrize("return_excluded, expected", [
        (True,["LAD1-LAD2"]),
        (False, ["LAD3-LAD4"]),
    ])
    def test_reduce_pairs(self, return_excluded, expected):
        reduced_pairs = reduce_pairs(self.test_pairs, self.stim_params,
                                     return_excluded=return_excluded).compute()
        assert (reduced_pairs.keys() == expected)
        return

    def test_generate_pairs_for_classifier(self):
        pairs_for_classifier = generate_pairs_for_classifier(self.test_pairs,
                                                             []).compute()
        assert all(pairs_for_classifier == self.test_pairs_recarray)

        return

    @pytest.mark.parametrize('excluded_pairs, expected', [
        ([], [True, True]),
        (['LAD1-LAD2'], [False, True]),
        (['LAD1-LAD2', 'LAD3-LAD4'], [False, False])
    ])
    def test_get_used_pair_mask(self, excluded_pairs, expected):
        pair_mask = get_used_pair_mask(self.test_pairs,
                                       excluded_pairs).compute()
        assert all(pair_mask == expected)

        return


