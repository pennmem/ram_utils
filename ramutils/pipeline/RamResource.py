from JSONUtils import JSONNode

class RamResource(object):
    def __init__(self,name='',task='',status=''):
        self.name = name
        self.task = task
        self.status = status

    def to_json(self):
        jn = JSONNode()
        jn['name']=self.name
        jn['task']=self.task
        jn['status']=self.status

        return jn
    def __str__(self):
        return self.name + ' ' + self.status

    def __repr__(self):
        return self.__str__()