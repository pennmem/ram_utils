from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
import json
import os
from timeit import default_timer
import numpy as np
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


def safe_divide(a, b=None):
    """Attempts to perform a/b and catches zero division errors to prevent
    crashing.

    Parameters
    ----------
    a : float or callable
        Numerator or the function to wrap when used as a decorator.
    b : float or None
        Denominator or None when used as a decorator.

    Returns
    -------
    result: float
        0 if denominator is 0, else a/b

    """
    if callable(a):
        @wraps(a)
        def wrapper(*args, **kwargs):
            try:
                return a(*args, **kwargs)
            except ZeroDivisionError:
                get_logger().warning("ZeroDivisionError; returning 0 instead")
                return 0
        return wrapper
    else:
        try:
            result = a / b
        except ZeroDivisionError:
            get_logger().warning("ZeroDivisionError; returning 0 instead")
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
    """ Extract the experiment series number from experiment name

    Parameters
    ----------
    experiment: str
        Name of the experiment

    Returns
    -------
    str
        Series number in string format (to accommodate PS2.1)

    """
    experiment = str(experiment)
    if experiment == 'PS2.1':
        return '2.1'

    # Assume series is the last value
    return experiment[-1]


def is_stim_experiment(experiment):
    """ Returns whether or not the given experiment is a stim experiment

    Parameters
    ----------
    experiment: str
        Name of the experiment

    Returns
    -------
    bool
        Indicator for if the given experiment is a stimulation experiment

    """
    experiment_series = extract_experiment_series(experiment)
    if experiment_series != '1':
        return True
    return False


def mkdir_p(dirs, mode=0o0777):
    """Mimic the shell command ``mkdir -p``.

    Parameters
    ----------
    dirs : str
    mode : int

    """
    try:
        os.makedirs(dirs, mode)
    except:
        pass


def bytes_to_str(istring, encoding='utf-8'):
    """Converts bytes to str in a version-neutral way.

    Parameters
    ----------
    istring : bytes or str
        Input string
    encoding : str
        Encoding to use when decoding (default: 'utf-8').

    Returns
    -------
    string : str

    """
    if hasattr(istring, 'decode'):
        return istring.decode(encoding=encoding)
    else:
        return istring


def load_event_test_data(datapath, rootdir):
    """
        Modify the path stored in the eegfile field to work with any given
        root directory at runtime. Used in test data suite to allow running
        tests rom an arbitrary location that has access to RHINO
    """
    events = np.rec.array(np.load(datapath))
    events['eegfile'] = [path.replace('/Volumes/RHINO/', rootdir) for path in
                         events['eegfile']]

    return events


def get_session_str(session_list):
    """
        Combine session numbers into '_' separated string. Used for saving
        underlying data
    """
    return "_".join([str(s) for s in session_list])
