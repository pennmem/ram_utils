import json
import logging
from logging.handlers import DatagramHandler
from uuid import uuid4

from dask.callbacks import Callback


class PipelineCallback(Callback):
    """Hooks for updating progress in a dask DAG. This uses a UDP socket to
    publish progress messages.

    Messages are JSON encoded and always have the keys ``pipeline`` which
    specifies the pipeline ID and ``type`` which specifies which hook is
    executed (with additional data depending on this). Note that ``type`` is
    specified by the dask callback naming convention.

    Usage::

        with PipelineCallback('my-totally-unique-pipeline-name'):
            delayed_stuff.compute()

    Parameters
    ----------
    pipeline_id : str
        Unique identifier for the running pipeline (generated with ``uuid4`` if
        not given).
    host : str
        UDP host address (default: ``'127.0.0.1'``)
    port : int
        UDP host port (default: ``50001``)

    """
    def __init__(self, pipeline_id=None, host='127.0.0.1', port=50001):
        super(PipelineCallback, self).__init__()
        self._pipeline_id = pipeline_id if pipeline_id is not None else uuid4().hex
        self.logger = logging.getLogger('pipeline.' + pipeline_id)

        handler = DatagramHandler(host, port)
        formatter = logging.Formatter(fmt="%(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)

        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _start(self, dsk):
        data = json.dumps({
            'pipeline': self._pipeline_id,
            'type': 'start',
        })
        self.logger.info(data)

    def _posttask(self, key, result, dsk, state, id):
        data = json.dumps({
            'pipeline': self._pipeline_id,
            'type': 'posttask',
            'progress': {
                'complete': len(state['finished']),
                'total': len(state['dependencies']),
            },
            'last_task': key,
        })
        self.logger.info(data)

    def _finish(self, dsk, state, errored):
        data = json.dumps({
            'pipeline': self._pipeline_id,
            'type': 'finish',
            'errored': errored,
        })
        self.logger.info(data)
