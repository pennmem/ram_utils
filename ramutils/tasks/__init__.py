"""Package for defining (functional) tasks. Tasks should be decorated with
:func:`task`. To avoid caching results, decorate with the keyword argument
``cache=False``.

"""

from ._wrapper import memory, task, make_task
from .classifier import *
from .events import *
from .misc import *
from .montage import *
from .odin import *
from .powers import *
from .summary import *
