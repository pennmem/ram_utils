import logging

import pytest

from ramutils.log import WarningAccumulator, get_logger


@pytest.mark.parametrize('flush', [True, False])
def test_warning_accumulator(flush):
    root = logging.getLogger()
    handler = WarningAccumulator()
    root.addHandler(handler)

    lines = handler.format_all(flush)
    assert lines is None

    for n in range(10):
        root.warning("warning %d", n)

    lines = handler.format_all(flush)
    assert isinstance(lines, str)
    for i, line in enumerate(lines.split('\n')[1:]):
        assert "warning {}".format(int(i)) in line

    if flush:
        assert len(handler._warnings) == 0
    else:
        assert len(handler._warnings) == 10
