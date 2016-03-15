__author__ = 'm'

import os
import os.path
import numpy as np

from DataMonitor import RamPopulator

from RamPipeline import *
from JSONUtils import JSONNode

class JSONStubPreparation(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

        # self.add_dependent_resource(resource_name='FR1_events',
        #                             json_node_access_list = ['experiments','FR1','events'])
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name='FR1_events',
                                        access_path = ['experiments','fr1','events'])

    #     json_stub_path = join(self.pipeline.workspace_dir,'_status','index.json')
    #     if not isfile(json_stub_path):
    #         rp = RamPopulator()
    #         j_stub = rp.create_subject_JSON_stub(subject_code=self.pipeline.subject)
    #         print j_stub.output()
    #
    #         j_stub.write(json_stub_path)
    #     else:
    #
    #         # j_stub = JSONNode.read(filename=json_stub_path)
    #         j_stub = self.pipeline.get_status()
    #         print 'READ JSON = ',j_stub.output()

    def run(self):
        print 'RUNNING JSONStubPreparation'
        # json_stub_path = join(self.pipeline.workspace_dir,'_status','index.json')
        # if not isfile(json_stub_path):
        #     rp = RamPopulator()
        #     j_stub = rp.create_subject_JSON_stub(subject_code=self.pipeline.subject)
        #     print j_stub.output()
        #
        #     j_stub.write(json_stub_path)
        # else:
        #
        #     # j_stub = JSONNode.read(filename=json_stub_path)
        #     j_stub = self.pipeline.get_status()
        #     print 'READ JSON = ',j_stub.output()