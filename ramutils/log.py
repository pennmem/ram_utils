import logging
from logging.handlers import RotatingFileHandler
from os.path import expanduser

try:  # pragma: nocover
    from typing import Dict
except ImportError:
    pass

_loggers = {}  # type: Dict[logging.Logger]


def get_logger(name='ramutils'):
    """Returns a configured logger to be used throughout the ramutils package.

    :param str name: Name for the logger (default: ``'ramutils'``)

    """
    if name not in _loggers:
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter(fmt='[%(levelname)1.1s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
        stream_handler.setFormatter(stream_formatter)

        file_handler = RotatingFileHandler(expanduser("~/.ramutils.log"),
                                           maxBytes=10e6, backupCount=4)
        file_formatter = logging.Formatter(fmt='[%(levelname)1.1s %(asctime)s %(pathname)s:%(lineno)d]\n    %(message)s')
        file_handler.setFormatter(file_formatter)

        _loggers[name] = logging.getLogger(name)
        _loggers[name].addHandler(stream_handler)
        _loggers[name].addHandler(file_handler)
        _loggers[name].setLevel(logging.INFO)

    return _loggers[name]
