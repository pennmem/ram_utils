from collections import OrderedDict
import json
import os


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
