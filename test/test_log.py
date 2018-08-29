from ramutils.log import *


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
