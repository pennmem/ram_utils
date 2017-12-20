import logging
from logging.handlers import RotatingFileHandler
from os.path import expanduser
from threading import Lock

try:  # pragma: nocover
    from typing import Dict
except ImportError:
    pass

_loggers = {}  # type: Dict[logging.Logger]


class WarningAccumulator(logging.Handler):
    """A special log handler to accumulate all logged warnings to prominently
    display them at the end.

    """
    def __init__(self):
        super(WarningAccumulator, self).__init__(level=logging.WARNING)
        self._warnings = []
        self._lock = Lock()

        formatter = logging.Formatter('[%(pathname)s:%(lineno)d] %(message)s')
        self.setFormatter(formatter)

    def emit(self, record):
        with self._lock:
            if record.levelno == logging.WARNING:
                self._warnings.append(record)

    def format_all(self, flush=True):
        """Formats all accumulated warnings.

        Parameters
        ----------
        flush : bool
            When True (the default), remove all accumulated warnings after
            formatting.

        Returns
        -------
        str or None
            A string of all formatted warnings or None if no warnings were
            accumulated.

        """
        lines = []
        with self._lock:
            for record in self._warnings:
                lines.append(self.formatter.format(record))

            if flush:
                self._warnings = []

        if len(lines):
            return "ACCUMULATED WARNINGS\n--------------------" + '\n'.join(lines)
        else:
            return None


def get_warning_accumulator():
    """Creates a new :class:`WarningAccumulator` if necessary and adds it to
    the root logger.

    Returns
    -------
    handler : WarningAccumulator

    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, WarningAccumulator):
            return handler

    handler = WarningAccumulator()
    root_logger.addHandler(handler)
    return handler


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
