from collections import namedtuple

__version__ = "2.1.1"

version_info = namedtuple('VersionInfo', 'major,minor,patch')(*__version__.split('.'))
