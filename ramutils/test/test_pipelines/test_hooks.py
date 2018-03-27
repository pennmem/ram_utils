from logging.handlers import DatagramHandler
import json
import math
import random

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

from dask import delayed

from ramutils.pipelines.hooks import PipelineCallback


@delayed
def generate_data(N):
    return [random.random() for _ in range(N)]


@delayed
def sqrt(data):
    return [math.sqrt(x) for x in data]


@delayed
def total(data):
    return sum(data)


def test_hooks():
    name = 'my-totally-unique-pipeline-name'

    with patch.object(DatagramHandler, 'emit') as emit:
        with PipelineCallback(name):
            total(sqrt(generate_data(10))).compute()

        assert emit.call_count == 3

        for args in emit.call_args_list:
            data = json.loads(args[0][0].message)
            assert data['pipeline'] == name
