import functools
import json
from pkg_resources import resource_string

from ramutils.tasks.odin import save_montage_files


def jsondata(s):
    return json.loads(resource_string('ramutils.test.test_data', s))


def test_save_montage_files(tmpdir):
    pairs = jsondata('R1328E_pairs.json')
    excluded = jsondata('R1328E_excluded_pairs.json')
    save_montage_files(pairs, excluded, str(tmpdir)).compute()

    with open(str(tmpdir.join('pairs.json')), 'r') as f:
        assert json.loads(f.read()) == pairs

    with open(str(tmpdir.join('excluded_pairs.json')), 'r') as f:
        assert json.loads(f.read()) == excluded
