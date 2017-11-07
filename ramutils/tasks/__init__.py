"""Package for defining (functional) tasks. Tasks should be decorated with
:func:`task`. To avoid caching results, decorate with the keyword argument
``cache=False``.

"""

import os.path
from tempfile import gettempdir
import functools

from dask import delayed
from sklearn.externals import joblib

from ptsa.data.readers.IndexReader import JsonIndexReader

from ramutils.log import get_logger

memory = joblib.Memory(cachedir=gettempdir(), verbose=0)
logger = get_logger()


def _log_call(func, with_args=True):
    """Logs calling of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if with_args:
            logger.info("calling %s with args=%r, kwargs=%r", func.__name__, args, kwargs)
        else:
            logger.info("calling %s", func.__name__)
        return func(*args, **kwargs)
    return wrapper


def task(cache=True, log_args=False):
    """Decorator to define a task.

    Keyword arguments
    -----------------
    cache : bool
        Cache the task result (default: True)
    log_args : bool
        Log arguments the task is called with (default: False)

    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wrapped = _log_call(func, log_args)
            if cache:
                wrapped = delayed(memory.cache(wrapped))(*args, **kwargs)
            else:
                wrapped = delayed(wrapped)(*args, **kwargs)
            return wrapped
        return wrapper
    return decorator


def make_task(func, *args, **kwargs):
    """Wrap a function in a task.

    Parameters
    ----------
    func : callable
        Function to wrap
    args
        Arguments for the function
    kwargs
        Keyword arugments for the function plus ``cache`` and ``log_args`` for
        use by the :func:`task` decorator.

    """
    try:
        cache = kwargs.pop('cache')
    except KeyError:
        cache = True

    try:
        log_args = kwargs.pop('log_args')
    except KeyError:
        log_args = False

    @task(cache, log_args)
    @functools.wraps(func)
    def wrapped(*a, **k):
        return func(*a, **k)

    return wrapped(*args, **kwargs)


@task()
def read_index(mount_point='/'):
    """Reads the JSON index reader.

    :param str mount_point: Root directory to search for.
    :returns: JsonIndexReader

    """
    path = os.path.join(mount_point, 'protocols', 'r1.json')
    return JsonIndexReader(path)


if __name__ == "__main__":
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        @task()
        def test():
            print('hi!')

        test().compute()
