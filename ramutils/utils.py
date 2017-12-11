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

    Parameters
    ----------
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

    Parameters
    ----------
    output: (str) Path to hdf5 output file
    data_name: (str) Name of the dataset
    data: (np.ndarray) Data array

    Notes
    -----
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


def combine_tag_names(tag_name_list):
    """Generate sensible output from a list of tuples containing anode and
    cathode contact names.

    """
    targets = [join_tag_tuple(target) for target in tag_name_list]
    return targets


def join_tag_tuple(tag_tuple):
    # Check if there is single-target stimulation
    if tag_tuple[0].find(",") == -1:
        return "-".join(tag_tuple)

    # First element of the tag tuple will be anodes, the second cathodes
    anodes = [el for el in tag_tuple[0].split(",")]
    cathodes = [el for el in tag_tuple[1].split(",")]

    pairs = ["-".join((anodes[i], cathodes[i])) for i in range(len(anodes))]
    joined = ":".join(pairs)

    return joined


def sanitize_comma_sep_list(input_list):
    """Clean up a string with comma-separated values to remove 0 elements."""
    tokens = input_list.split(",")
    tokens = [token for token in tokens if token != "0"]
    output = ",".join(tokens)
    return output


def extract_subject_montage(subject_id):
    """ Extract the subject ID and montage number from the subject ID

    Parameters
    ----------
    subject_id: str
        Subject identifier

    Returns
    -------
    str:
        Subject identifier with montage information removed
    int:
        Montage number

    """
    tokens = subject_id.split('_')
    montage = 0 if len(tokens) == 1 else int(tokens[1])
    subject = tokens[0]
    return subject, montage


def extract_experiment_series(experiment):
    experiment = str(experiment)
    if experiment == 'PS2.1':
        return '2.1'

    # Assume series is the last value
    return experiment[-1]
