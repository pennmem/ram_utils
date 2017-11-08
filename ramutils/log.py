import logging

_loggers = {}


def get_logger(name='ramutils'):
    """Returns a configured logger to be used throughout the ramutils package.

    :param str name: Name for the logger (default: ``'ramutils'``)

    """
    if name not in _loggers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt='[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s')
        handler.setFormatter(formatter)

        _loggers[name] = logging.getLogger(name)
        _loggers[name].addHandler(handler)
        _loggers[name].setLevel(logging.INFO)

    return _loggers[name]
