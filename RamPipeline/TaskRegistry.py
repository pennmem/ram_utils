from collections import OrderedDict

class TaskRegistry(object):
    def __init__(self):
        self.task_dict=OrderedDict()

    def register_task(self,task):
        # print 'task=',dir(task)
        # print 'type(x).__name__=',type(task).__name__
        self.task_dict[type(task).__name__] = task

        # self.task_dict[]=task

    def run_tasks(self):
        for task_name, task in self.task_dict.items():
            print 'RUNNIGN TASK:', task_name,' obj=',task
            task.run()
        
