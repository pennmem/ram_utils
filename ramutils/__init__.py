from collections import namedtuple

__version__ = "2.1.6"

version_info = namedtuple('VersionInfo', 'major,minor,patch')(*__version__.split('.'))
