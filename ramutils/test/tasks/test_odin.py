import json
from pkg_resources import resource_string
import pytest

from ramutils.tasks.odin import save_montage_files


def jsondata(s):
    return json.loads(resource_string('ramutils.test.test_data', s))


@pytest.mark.parametrize('mkdir', [True, False])
def test_save_montage_files(mkdir, tmpdir):
    pairs = jsondata('R1328E_pairs.json')
    excluded = jsondata('R1328E_excluded_pairs.json')

    dest = str(tmpdir.join('otherdir')) if mkdir else str(tmpdir)
    save_montage_files(pairs, excluded, dest).compute()

    with open(str(tmpdir.join('pairs.json')), 'r') as f:
        assert json.loads(f.read()) == pairs

    with open(str(tmpdir.join('excluded_pairs.json')), 'r') as f:
        assert json.loads(f.read()) == excluded

    # FIXME
    # with pytest.raises(AssertionError):
    #     path = tmpdir.join('file')
    #     path.write_binary(b'')
    #     save_montage_files(pairs, excluded, str(path))
