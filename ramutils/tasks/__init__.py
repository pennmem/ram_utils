"""Package for defining (functional) tasks. Tasks should be decorated with
``@memory.cache`` in order to cache results (see ``joblib`` documentation for
 details). By default, all cached results will be stored in the directory that
 is returned by :func:`tempfile.gettempdir`; this can be changed by setting
 ``memory.cachedir`` prior to executing a pipeline.

"""

import os.path
from tempfile import gettempdir

from dask import delayed
from sklearn.externals import joblib

from ptsa.data.readers.IndexReader import JsonIndexReader

memory = joblib.Memory(cachedir=gettempdir())


@delayed
def read_index(mount_point='/'):
    """Reads the JSON index reader.

    :param str mount_point: Root directory to search for.
    :returns: JsonIndexReader

    """
    path = os.path.join(mount_point, 'protocols', 'r1.json')
    return JsonIndexReader(path)
