from copy import deepcopy
import json

import pytest

from ramutils import conf

CONFS = {
    'empty': "{}",
    'paths_modified': '{"PATHS": {"root": "/root"}}'
}


@pytest.mark.parametrize('key', CONFS.keys())
def test_load_user_settings(key, tmpdir):
    paths = deepcopy(conf.PATHS)  # original configuration
    conf_path = str(tmpdir.join('user_settings.py'))

    with open(conf_path, 'w') as f:
        f.write(CONFS[key])

    conf.load_user_settings(conf_path)

    if key == 'empty':
        assert conf.PATHS == paths
    elif key == 'paths_modified':
        assert conf.PATHS['root'] == '/root'
        assert paths['cachedir'] == conf.PATHS['cachedir']
        assert paths['logdir'] == conf.PATHS['logdir']


def test_save_user_settings(tmpdir):
    conf.PATHS['root'] = '/root'
    path = str(tmpdir.join('user_settings.json'))

    conf.save_user_settings(path)

    with open(path, 'r') as f:
        loaded = json.loads(f.read())

    assert loaded['PATHS'] == conf.PATHS
