from collections import namedtuple
import logging
import warnings


__version__ = "2.3.0"

version_info = namedtuple('VersionInfo', 'major,minor,patch')(
    *__version__.split('.'))

# So we don't end up getting things logged multiple times
_root_logger = logging.getLogger()
_root_logger.handlers = []
_root_logger.addHandler(logging.NullHandler())

# disable FutureWarnings originating in PTSA
warnings.filterwarnings("ignore", category=FutureWarning, module="ptsa*",
                        append=True)
