import logging
from logging.handlers import RotatingFileHandler
from os.path import expanduser

__all__ = [
    'RamutilsStreamHandler',
    'RamutilsFileHandler',
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
