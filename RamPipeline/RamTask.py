

class RamTask(object):

    outputs = []
    pipeline = None

    def __init__(self):
        pass

    def get_pipeline(self):
        return self.pipeline

    def get_workspace_dir(self):
        return self.pipeline.workspace_dir

    def set_pipeline(self,pipeline):
        self.pipeline = pipeline
    #     self.__name='Task'
    #
    # def name(self):
    #     return self.__name

    # def __set_name(self, name):
    #     self.__name = name


    def run(self):
        pass

