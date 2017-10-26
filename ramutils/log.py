import logging

_logger = None


def get_logger():
    """Returns a configured logger to be used throughout the pipeline package.

    """
    global _logger

    if _logger is None:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt='[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d]%(end_color)s %(message)s')
        handler.setFormatter(formatter)

        _logger = logging.getLogger('pipeline')
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)

    return _logger
