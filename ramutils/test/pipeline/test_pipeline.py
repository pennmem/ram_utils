from unittest import TestCase
from ramutils.pipeline import RamPipeline


class PipelineTestCase(TestCase):
    def setUp(self):
        self.pipeline = RamPipeline()

    def test_set_workspace_dir(self):
        self.pipeline.set_workspace_dir('/tmp')
