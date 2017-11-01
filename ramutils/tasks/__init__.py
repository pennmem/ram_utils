"""Package for defining (functional) tasks. Tasks should be decorated with
``@memory.cache`` in order to cache results (see ``joblib`` documentation for
 details). By default, all cached results will be stored in the directory that
 is returned by :func:`tempfile.gettempdir`; this can be changed by setting
 ``memory.cachedir`` prior to executing a pipeline.

"""

import os.path
from tempfile import gettempdir
import functools

from dask import delayed
from sklearn.externals import joblib

from ptsa.data.readers.IndexReader import JsonIndexReader

memory = joblib.Memory(cachedir=gettempdir(), verbose=0)


def task(cache=True):
    """Decorator to define a task.

    Keyword arguments
    -----------------
    cache : bool
        Cache the task result (default: True)

    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*fargs, **fkwargs):
            if cache:
                wrapped = delayed(memory.cache(func))(*fargs, **fkwargs)
            else:
                wrapped = delayed(func)(*fargs, **fkwargs)
            return wrapped
        return wrapper
    return decorator


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

        print(test)
