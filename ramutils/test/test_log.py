import logging

import pytest

from ramutils.log import *
from ramutils.log import WarningAccumulator


@pytest.fixture
def logger():
    logger = logging.getLogger()
    yield logger
    logger.handlers = []


@pytest.mark.parametrize('flush', [True, False])
def test_warning_accumulator(flush, logger):
    handler = WarningAccumulator()
    logger.addHandler(handler)

    lines = handler.format_all(flush)
    assert lines is ''

    for n in range(10):
        logger.warning("warning %d", n)

    lines = handler.format_all(flush)
    assert isinstance(lines, str)
    for i, line in enumerate(lines.split('\n')[2:]):
        assert "warning {}".format(int(i)) in line

    if flush:
        assert len(handler._warnings) == 0
    else:
        assert len(handler._warnings) == 10


def test_get_warning_accumulator(logger):
    assert len(logger.handlers) == 0

    handler = get_warning_accumulator()
    assert isinstance(handler, WarningAccumulator)
    assert len(logger.handlers) == 1

    assert get_warning_accumulator() == handler


def test_get_logger():
    logger = get_logger('test')
    has_stream_handler = False
    has_file_handler = False

    for handler in logger.handlers:
        if isinstance(handler, RamutilsStreamHandler):
            has_stream_handler = True
        if isinstance(handler, RamutilsFileHandler):
            has_file_handler = True

    assert has_stream_handler
    assert has_file_handler

    # Getting the logger again shouldn't add new handlers
    logger = get_logger('test')
    stream_count = 0
    file_count = 0

    for handler in logger.handlers:
        if isinstance(handler, RamutilsStreamHandler):
            stream_count += 1
        if isinstance(handler, RamutilsFileHandler):
            file_count += 1

    assert stream_count == 1
    assert file_count == 1
