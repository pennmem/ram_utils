__author__ = 'm'

from RamTask import RamTask

class MatlabRamTask(RamTask):

    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

        # from MatlabUtils import matlab_engine as eng
        # self.eng = eng

    def set_matlab_engine(self, eng):
        '''
        Sets matlab engine - this is done by the pipeline to avoid starting of the matlab engine when class
        inheriting from MatlabRamTask gets created. Matlab engines starts are very slow
        :param eng: instance of matlab engine
        :return:None
        '''
        self.eng = eng
