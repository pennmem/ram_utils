import logging
from logging.handlers import DatagramHandler
import json
import math
import random
import time
from unittest.mock import patch

import pytest

from dask import delayed

from ramutils.pipelines.hooks import PipelineCallback, PipelineStatusListener


@delayed
def generate_data(N):
    return [random.random() for _ in range(N)]


@delayed
def sqrt(data):
    return [math.sqrt(x) for x in data]


@delayed
def total(data):
    return sum(data)


@pytest.mark.skip(reason="not important")
def test_hooks():
    name = 'my-totally-unique-pipeline-name'

    with patch.object(DatagramHandler, 'emit') as emit:
        with PipelineCallback(name):
            total(sqrt(generate_data(10))).compute()

        for i, args in enumerate(emit.call_args_list):
            data = json.loads(args[0][0].msg)
            assert data['pipeline'] == name

            if i == 0:
                assert data['type'] == 'start'
            elif i == 7:
                assert data['type'] == 'finish'
            elif i % 2 == 0:
                assert data['type'] == 'posttask'
            else:
                assert data['type'] == 'pretask'


@pytest.mark.skip(reason="not important")
@pytest.mark.parametrize('pipeline_id', [None, 'mypipeline'])
@pytest.mark.parametrize('port', [50001])
def test_listener(pipeline_id, port):
    messages = []

    def callback(msg):
        messages.append(msg)

    with PipelineStatusListener(callback, pipeline_id):
        logger = logging.getLogger(pipeline_id or __name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(DatagramHandler('127.0.0.1', port))

        logger.info('{"level":"info"}')
        logger.warning('{"level":"warning"}')
        time.sleep(0.001)

    assert len(messages) == 2
