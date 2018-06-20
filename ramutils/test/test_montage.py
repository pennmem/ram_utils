import functools
from pkg_resources import resource_filename
import pytest

from numpy.testing import assert_equal

from ramutils.montage import *
from ramutils.parameters import StimParameters, FilePaths

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
        ], dtype=dtypes.pairs, names=['contact0', 'contact1', 'label0',
                                      'label1'])

        cls.stim_params = [StimParameters(anode_label='LAD1',
                                          cathode_label='LAD2',
                                          anode=9,
                                          cathode=10)
                           ]

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
        (True, ["LAD1-LAD2"]),
        (False, ["LAD3-LAD4"]),
    ])
    def test_reduce_pairs(self, return_excluded, expected):
        reduced_pairs = reduce_pairs(self.test_pairs, self.stim_params,
                                     return_excluded=return_excluded)
        assert (list(reduced_pairs.keys()) == expected)
        return

    def test_generate_pairs_for_classifier(self):
        pairs_for_classifier = generate_pairs_for_classifier(self.test_pairs,
                                                             [])
        assert all(pairs_for_classifier == self.test_pairs_recarray)

        return

    @pytest.mark.parametrize('excluded_pairs, expected', [
        ([], [True, True]),
        (['LAD1-LAD2'], [False, True]),
        (['LAD3-LAD4'], [True, False]),
        (['LAD1-LAD2', 'LAD3-LAD4'], [False, False])
    ])
    def test_get_used_pair_mask(self, excluded_pairs, expected):
        # Constructed test pairs are not an Ordered Dict, so this should
        # raise a flag
        with pytest.raises(RuntimeError):
            pair_mask = get_used_pair_mask(self.test_pairs,
                                           excluded_pairs)

        ordered_pairs = OrderedDict()
        ordered_pairs['LAD1-LAD2'] = 'test'
        ordered_pairs['LAD3-LAD4'] = 'test'

        mock_pairs = OrderedDict({'test_subject': {'pairs': ordered_pairs}})

        pair_mask = get_used_pair_mask(mock_pairs, excluded_pairs)
        for i in range(len(pair_mask)):
            assert expected[i] == pair_mask[i]

        return

    def test_compare_recorded_with_all_pairs(self):
        ordered_pairs = OrderedDict()
        ordered_pairs['LAD1-LAD2'] = {'channel_1': 9, 'channel_2': 10}
        ordered_pairs['LAD3-LAD4'] = {'channel_1': 11, 'channel_2': 12}
        mock_pairs = OrderedDict({'test_subject': {'pairs': ordered_pairs}})

        mask = compare_recorded_with_all_pairs(mock_pairs,
                                               self.test_pairs_recarray)
        assert sum(mask) == 2

        ordered_pairs = OrderedDict()
        ordered_pairs['LAD1-LAD2'] = {'channel_1': 9, 'channel_2': 10}
        ordered_pairs['LAD3-LAD4'] = {'channel_1': 11, 'channel_2': 12}
        ordered_pairs['LAD4-LAD5'] = {'channel_1': 13, 'channel_2': 14}
        mock_pairs = OrderedDict({'test_subject': {'pairs': ordered_pairs}})

        mask = compare_recorded_with_all_pairs(mock_pairs,
                                               self.test_pairs_recarray)
        assert sum(mask) == 2

        return

    @pytest.mark.parametrize('subject', [
        'R1354E',
    ])
    def test_load_pairs_from_json(self, subject):
        test_pairs = load_pairs_from_json(subject, 'FR1', rootdir=datafile(''))
        assert len(test_pairs.keys()) > 10
        assert '11LD1-11LD2' in test_pairs

        return

    @pytest.mark.parametrize('subject', [
        'R1354E'
    ])
    def test_build_montage_metadata_table(self, subject):
        with open(datafile('/input/configs/{}_pairs_from_ec.json'.format(subject))) as f:
            pairs_from_ec = json.load(f)

        metadata_table = build_montage_metadata_table(subject,
                                                      'FR1',
                                                      [0],
                                                      pairs_from_ec,
                                                      root=datafile(''))
        assert len(metadata_table) == len(
            pairs_from_ec[subject]['pairs'].keys())

        return

    @pytest.mark.parametrize('subject', [
        'R1354E'
    ])
    def test_build_montage_metadata_table_regression(self, subject):
        with open(datafile('/input/configs/{}_pairs_from_ec.json'.format(subject))) as f:
            pairs_from_ec = json.load(f)

        table = build_montage_metadata_table(subject,
                                             'FR1',
                                             [0],
                                             pairs_from_ec,
                                             root=datafile(''))
        old = pd.read_csv(
            datafile('/input/montage/{}_montage_metadata.csv'.format(subject)))

        # Check correspondence my merging
        # merged = table.merge(old, how='outer', indicator=True)
        # assert 'left_only' not in merged._merge
        # assert 'right_only' not in merged._merge
        # ^-- I'm not really sure what correspondence we were supposed to be
        # checking here, so instead I'm just going to check that relevant fields
        # give the same values - MVD 2018-06-20

        assert_equal(table.channel_1.values.astype(int), old.channel_1.values)
        assert_equal(table.channel_2.values.astype(int), old.channel_2.values)
        assert_equal(table.label.dropna().values, old.label.dropna().values)
        assert_equal(table.location.dropna().values,
                     old.location.dropna().values)

    @pytest.mark.rhino
    @pytest.mark.parametrize('subject, experiment', [
        ('R1375C', 'catFR1'),  # Will use electrode config
        ('R1320D', 'catFR1'),  # Will fall back to pairs.json
    ])
    def test_get_pairs(self, subject, experiment, rhino_root):
        paths = FilePaths(root=rhino_root)
        pairs = get_pairs(subject, experiment, None, paths)
        assert len(pairs.keys()) > 0
        assert pairs[subject] is not None

        return

    def test_generate_pairs_from_electrode_config(self):
        paths = FilePaths(root=datafile(''),
                          electrode_config_file='/input/configs/R1354E_26OCT2017L0M0STIM.csv',
                          pairs='/input/montage/R1354E_pairs.json')
        config_pairs = generate_pairs_from_electrode_config('R1354E',
                                                            'FR1',
                                                            [0],
                                                            paths)
        assert len(config_pairs['R1354E']['pairs'].keys()) > 0
        return

    def test_get_classifier_excluded_leads(self):
        paths = FilePaths(root=datafile(''))

        excluded_contacts = get_classifier_excluded_leads(
            'test_subject', self.test_pairs, paths.root)
        assert len(excluded_contacts) == 1
        assert excluded_contacts[0].anode_label == 'LAD1'
        assert excluded_contacts[0].cathode_label == 'LAD2'
