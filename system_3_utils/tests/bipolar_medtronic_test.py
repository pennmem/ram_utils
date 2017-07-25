import json
import unittest
from os.path import *
from ReportTasks.hdf5_utils import read_array_from_hdf5

import numpy as np
import tables
from numpy.testing import *
import time


class TestBipolarMedtronic(unittest.TestCase):
    def setUp(self):
        self.monopolar_trans_matrix_file = r'd:\experiment_configs1\monopolar_trans_matrixR1232N.h5'

        self.monopolar_trans_matrix = np.matrix(read_array_from_hdf5(self.monopolar_trans_matrix_file, 'monopolar_trans_matrix'))


    def test_bipol(self):
        num_channels = self.monopolar_trans_matrix.shape[0]

        rand_array = np.random.randn(num_channels, 5000)
        t0 = time.time()
        rand_signal = np.matrix(rand_array)
        print 'took %f ms to convert'%((time.time()-t0)/1000.0)

        trans_res = self.monopolar_trans_matrix*rand_signal
        trans_res_array = np.array(trans_res)

        print self.monopolar_trans_matrix
        assert_array_equal(trans_res[0,:], rand_signal[0,:])
        assert_array_equal(trans_res[4,:], -rand_signal[0,:]+rand_signal[4,:])
        assert_array_equal(trans_res[33,:], rand_signal[33,:])




