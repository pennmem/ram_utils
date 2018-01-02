import logging

import pytest

from ramutils.log import WarningAccumulator, get_logger, get_warning_accumulator


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
