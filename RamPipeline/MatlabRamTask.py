__author__ = 'm'

from RamTask import *

class MatlabRamTask(RamTask):
    def __init__(self,mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

        from MatlabUtils import matlab_engine as eng
        self.eng = eng


