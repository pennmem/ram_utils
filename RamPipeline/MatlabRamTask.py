__author__ = 'm'

from RamTask import *

class MatlabRamTask(RamTask):
    def __init__(self):
        RamTask.__init__(self)

        from MatlabUtils import matlab_engine as eng
        self.eng = eng


