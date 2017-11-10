import functools
import json
from pkg_resources import resource_filename

from ramutils.tasks.montage import load_pairs_from_json

datafile = functools.partial(resource_filename, 'ramutils.test.test_data')


def test_load_pairs():
    filename = datafile('R1328E_pairs.json')
    pairs = load_pairs_from_json(filename)

    with open(filename) as f:
        data = json.load(f)

    assert 'R1328E' in data
    assert 'pairs' in data['R1328E']
