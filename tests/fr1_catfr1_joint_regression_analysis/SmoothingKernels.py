import numpy as np
from fractions import Fraction


class SmoothingKernel(object):
    def __init__(self, radius):
        self.radius = radius
        self.matrices = dict()

    def construct_smoothing_matrix(self, size):
        c = np.ones(shape=(size,size), dtype=int)
        for i in xrange(1,size):
            for j in xrange(i):
                if abs(i-j) > self.radius:
                    c[i,j] = 0
                    c[j,i] = 0
        d = np.empty(shape=(size,size), dtype=Fraction)
        for i in xrange(size):
            s = np.sum(c[i,:])
            for j in xrange(size):
                d[i,j] = Fraction(c[i,j], s)
        return d

    def coefficient(self, i, j, size):
        if size not in self.matrices:
            self.matrices[size] = self.construct_smoothing_matrix(size)
        return self.matrices[size][i,j]
