from math import sin,cos
import numpy as np
from numpy.random import randn
from morlet import MorletWaveletTransform
import pycircstat.descriptive
import circular_stat

wavelet_transform = MorletWaveletTransform()
wavelet_transform.init(5, 45.0, 95.0, 11, 1000.0, 5800)

#sin_wave = np.array([sin(2*45*np.pi*x/1000.0) for x in xrange(5800)], dtype=np.float)
#cos_wave = np.array([cos(2*45*np.pi*x/1000.0) for x in xrange(5800)], dtype=np.float)

sin_wave = randn(5800)
cos_wave = randn(5800)

sin_wavelets = np.empty(shape=11*5800, dtype=np.complex)
cos_wavelets = np.empty(shape=11*5800, dtype=np.complex)
wavelet_transform.multiphasevec_complex(sin_wave, sin_wavelets)
wavelet_transform.multiphasevec_complex(cos_wave, cos_wavelets)

sin_wavelets = sin_wavelets.reshape((11,5800))
sin_wavelets = sin_wavelets[:,1000:-1000]

cos_wavelets = cos_wavelets.reshape((11,5800))
cos_wavelets = cos_wavelets[:,1000:-1000]

cdiff = np.empty(shape=(11,3800), dtype=np.complex)
cdiff_means = np.empty(shape=(11,19), dtype=np.float)

for i in xrange(11):
    circular_stat.circ_diff(sin_wavelets[i], cos_wavelets[i], cdiff[i])
    print "C++: circ_mean[%d] = %f" % (i,circular_stat.circ_mean(cdiff[i]))

    circular_stat.circ_diff_time_bins(sin_wavelets[i], cos_wavelets[i], cdiff[i], cdiff_means[i])
    print cdiff_means[i]
    print "C++: circ_mean[%d] = %f" % (i,circular_stat.circ_mean(cdiff[i]))

print "Now let's do it the old way"

sin_powers = np.empty(shape=11*5800, dtype=np.float)
sin_phases = np.empty(shape=11*5800, dtype=np.float)
cos_powers = np.empty(shape=11*5800, dtype=np.float)
cos_phases = np.empty(shape=11*5800, dtype=np.float)

wavelet_transform.multiphasevec(sin_wave, sin_powers, sin_phases)
wavelet_transform.multiphasevec(cos_wave, cos_powers, cos_phases)

sin_phases = sin_phases.reshape((11,5800))
sin_phases = sin_phases[:,1000:-1000]

cos_phases = cos_phases.reshape((11,5800))
cos_phases = cos_phases[:,1000:-1000]

phase_diff = sin_phases - cos_phases

cdiff_means_old = np.empty(shape=(11,19), dtype=np.float)

bin_len = 3800 // 19

for i in xrange(19):
    cdiff_means_old[:,i] = pycircstat.descriptive.mean(phase_diff[:,i*bin_len:(i+1)*bin_len], axis=1)

cdiff_means = np.where(cdiff_means>=0.0, cdiff_means, cdiff_means+2*np.pi)

print cdiff_means - cdiff_means_old
