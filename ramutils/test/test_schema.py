import json
import pytest
import numpy as np
from numpy.testing import assert_equal
import h5py

from traits.api import Array, String

from ramutils.schema import Schema


class SomeSchema(Schema):
    x = Array(dtype=np.float)
    name = String()


@pytest.mark.parametrize('mode', ['w', 'a'])
@pytest.mark.parametrize('desc', ['a number', None])
def test_to_hdf(mode, desc, tmpdir):
    class MySchema(Schema):
        x = Array(dtype=np.float64, desc=desc)
        y = Array(dtype=np.int32, desc=desc)
    obj = MySchema(x=np.random.random(100), y=np.random.random(100))
    filename = str(tmpdir.join('test.h5'))

    obj.to_hdf(filename, mode)

    with h5py.File(filename, 'r') as hfile:
        assert_equal(hfile['/x'][:], obj.x)
        assert_equal(hfile['/y'][:], obj.y)


def test_from_hdf(tmpdir):
    x = np.arange(10)
    y = np.arange(10, dtype=np.int32)

    path = str(tmpdir.join('test.h5'))

    with h5py.File(path, 'w') as hfile:
        hfile.create_dataset('/x', data=x, chunks=True)
        hfile.create_dataset('/y', data=y, chunks=True)

    class MySchema(Schema):
        x = Array(dtype=np.float64)
        y = Array(dtype=np.int32)

    instance = MySchema.from_hdf(path)

    assert_equal(instance.x, x)
    assert_equal(instance.y, y)


def test_to_json():
    obj = SomeSchema(x=list(range(10)), name="whatever")
    jobj = obj.to_json()

    loaded = json.loads(jobj)
    assert_equal(loaded['x'], obj.x)
    assert loaded['name'] == obj.name


@pytest.mark.parametrize('fromfile', [True, False])
def test_from_json(fromfile, tmpdir):
    data = {
        "x": list(range(10)),
        "name": "whatever"
    }

    if not fromfile:
        obj = SomeSchema.from_json(json.dumps(data))
    else:
        filename = str(tmpdir.join('test.json'))
        with open(filename, 'w') as f:
            json.dump(data, f)

        with open(filename, 'r') as f:
            obj = SomeSchema.from_json(f)

    assert_equal(obj.x, data['x'])
    assert obj.name == data['name']
