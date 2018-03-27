import logging
from logging.handlers import RotatingFileHandler
from os.path import expanduser
from threading import Lock

__all__ = [
    'RamutilsStreamHandler',
    'RamutilsFileHandler',
    'get_warning_accumulator',
    'get_logger',
]


class RamutilsStreamHandler(logging.StreamHandler):
    """Custom stream handler used by ramutils loggers."""
    _FORMAT = '[%(levelname)1.1s %(asctime)s %(filename)s:%(lineno)d] %(message)s'

    def __init__(self, *args, **kwargs):
        logging.StreamHandler.__init__(self, *args, **kwargs)

        formatter = logging.Formatter(fmt=self._FORMAT)
        self.setFormatter(formatter)


class RamutilsFileHandler(RotatingFileHandler):
    """Custom rotating file handler used by ramutils loggers.

    Parameters
    ----------
    filename : str
        Base filename for logs.

    """
    _FORMAT = '[%(levelname)1.1s %(asctime)s %(pathname)s:%(lineno)d]\n    %(message)s'

    def __init__(self, filename):
        RotatingFileHandler.__init__(self, filename, 'a', maxBytes=10e6,
                                     backupCount=4)

        formatter = logging.Formatter(fmt=self._FORMAT)
        self.setFormatter(formatter)


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
            return "ACCUMULATED WARNINGS\n--------------------\n" + '\n'.join(lines)
        else:
            return ''


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
    logger = logging.getLogger(name)

    for handler in logger.handlers:
        if isinstance(handler, RamutilsStreamHandler):
            # Logging has already been configured
            return logger

    stream_handler = RamutilsStreamHandler()
    file_handler = RamutilsFileHandler(expanduser("~/.ramutils.log"))
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger
