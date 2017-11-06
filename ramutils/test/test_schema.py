import pytest
import numpy as np
from numpy.testing import assert_equal
import h5py

from traits.api import Array

from ramutils.schema import Schema


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
