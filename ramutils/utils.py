from collections import OrderedDict
from contextlib import contextmanager
import json
import os
from timeit import default_timer

import h5py

from ramutils.log import get_logger


def reindent_json(json_file, indent=2):
    """Re-indent JSON data preserving order. Useful for removing extraneous
    whitespace leftover from templates generation.

    Parameters
    ----------
    json_file : str or file-like
        Path to JSON file or a file-like object containing JSON data.
    indent : int
        Indent width

    Returns
    -------
    reindented : str
        Re-indented JSON string

    """
    # Path to JSON file
    if isinstance(json_file, str):
        with open(json_file, 'r') as f:
            data = json.loads(f.read(), object_pairs_hook=OrderedDict)

    # File-like object
    else:
        data = json.load(json_file, object_pairs_hook=OrderedDict)

    return json.dumps(data, indent=indent)


def safe_divide(a, b):
    """ Attempts to perform a/b and catches zero division errors to prevent crashing

    Parameters:
    -----------
    a: float
        Numerator
    b: float
        Denominator

    Returns
    -------
    result: float
        0 if denominator is 0, else a/b

    """
    try:
        result = a / b
    except ZeroDivisionError:
        result = 0

    return result


def touch(path):
    """Mimics the unix ``touch`` command."""
    with open(path, 'a'):
        os.utime(path, None)


def save_array_to_hdf5(output, data_name, data, overwrite=False):
    """Save an array of data to hdf5

    Parameters:
    -----------
    output: (str) Path to hdf5 output file
    data_name: (str) Name of the dataset
    data: (np.ndarray) Data array

    Notes:
    ------
    Primarily useful for debugging purposes. Could be used to save underlying
    data for report plots.

    """
    mode = 'a'
    if overwrite:
        mode = 'w'
    hdf = h5py.File(output, mode)
    hdf.create_dataset(data_name, data=data)
    hdf.close()
    return


@contextmanager
def timer(message="Elapsed time: %.3f s", logger=None):
    """Context manager to log the elapsed time of a code block.

    Parameters
    ----------
    message : str
        Percent-formatted string to display the elapsed time
    logger : str
        Name of logger to use

    """
    ti = default_timer()
    yield
    tf = default_timer()
    log = get_logger(logger)
    log.info(message, tf - ti)


if __name__ == "__main__":
    import time

    with timer("Slept for a total of %f s"):
        time.sleep(1)
