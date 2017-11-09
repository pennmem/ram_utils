from collections import namedtuple

__version__ = "2.0.dev0"

version_info = namedtuple('VersionInfo', 'major,minor,patch')(*__version__.split('.'))
