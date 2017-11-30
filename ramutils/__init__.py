from collections import namedtuple

__version__ = "2.0.dev1"

version_info = namedtuple('VersionInfo', 'major,minor,patch')(*__version__.split('.'))
