from collections import namedtuple
import logging

__version__ = "2.1.10"

version_info = namedtuple('VersionInfo', 'major,minor,patch')(*__version__.split('.'))

# So we don't end up getting things logged multiple times
_root_logger = logging.getLogger()
_root_logger.handlers = []
_root_logger.addHandler(logging.NullHandler())
