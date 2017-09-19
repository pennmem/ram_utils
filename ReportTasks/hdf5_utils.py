import tables
import numpy as np


def save_arrays_as_hdf5(fname, array_dict, title='HDF OUTPUT'):
    """
    Convenience fcn to save a dictionary of ndaarrays in the HDF5 format
    :param fname: hdf5 file name - absolute path
    :param array_dict: dictionary of ndarrays
    :param title: frame title (optional)
    :return: None
    """

    h5file = tables.open_file(fname, mode='w', title=title)
    root = h5file.root

    for array_name, array in array_dict.iteritems():
        h5file.create_array(root, array_name, array)

    h5file.close()


def read_array_from_hdf5(fname, array_name):
    """
    reads specified array from hdf file
    :param fname: name of the HDF5 file
    :param array_name: name of the array
    :return: ndarray
    """

    h5file = tables.open_file(fname, mode='r')
    try:
        array_hdf5_obj = getattr(h5file.root,array_name)
    except tables.exceptions.NoSuchNodeError:
        raise AttributeError('Could not find array: %s in the %s file'%(fname,array_name))

    return array_hdf5_obj.read()



    pass


if __name__ == '__main__':

    a_array = np.arange(20)
    b_array = np.arange(40)
    c_array = np.arange(100)

    array_dict = {'a_array': a_array, 'b_array': b_array, 'c_array': c_array}

    output_file_name = 'D:/PAL5/array_test.h5'
    save_arrays_as_hdf5(fname=output_file_name, array_dict=array_dict)


    a_array_test = read_array_from_hdf5(output_file_name,'a_array')
    b_array_test = read_array_from_hdf5(output_file_name,'b_array')
    c_array_test = read_array_from_hdf5(output_file_name,'c_array')

    np.testing.assert_array_equal(a_array_test, a_array)
    np.testing.assert_array_equal(b_array_test, b_array)
    np.testing.assert_array_equal(c_array_test, c_array)