from collections import namedtuple

__version__ = "3.0.0"

version_info = namedtuple('VersionInfo', 'major,minor,patch')(*__version__.split('.'))
