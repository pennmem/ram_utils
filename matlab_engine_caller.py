# import matlab.engine
import scipy.io as sio
from os.path import *
import sys

# print os.path.abspath('~/matlab_extern')
#
# print os.path.abspath(os.path.expanduser('~/matlab_extern'))

# sys.exit()
#
# print "***************** Starting MATLAB Engine *************************"
# eng = matlab.engine.start_matlab()
# print "***************** MATLAB Engine WORKING *************************"
#
# matlab_script_paths = ['~/matlab_extern','~/matlab_extern_1']
#
#
# # a = eng.matlab_path_set(abspath(expanduser('~/matlab_extern')))
#
# for path in matlab_script_paths:
#
#     a = eng.matlab_path_set(abspath(expanduser(path)))
#     print a


import MatlabUtils

from MatlabUtils import matlab_engine as eng

MatlabUtils.add_matlab_search_paths('~/matlab_extern','~/matlab_extern_1')

area = eng.triarea_new(10, 20)

print "area=", area


area1 = eng.triarea_new_1(10, 20)
print "area1=", area1

# sys.exit()


import numpy as np

from MatlabIO import MatlabIO

params_loaded = MatlabIO()
params_loaded.deserialize('params_serialized.mat')

eng.CreateParamsDemo('params_serialized.mat')

params_serialized_from_matlab = MatlabIO()
params_serialized_from_matlab.deserialize('params_serialized_from_matlab.mat')


print 'params_serialized_from_matlab.eeg.durationMS=',params_serialized_from_matlab.eeg.durationMS
print 'params_serialized_from_matlab.pow.freqBins=', params_serialized_from_matlab.pow.freqBins






# # for i in xrange(20):
#
# #     for j in xrange(20):
#
# #         ret = eng.triarea(i*1.0,j*5.0)
# #         print "i=",i," j=",j," area=",ret
#
# mat_return=eng.matrix_return(10,10)
#
# print "mat_return=",mat_return
#
# print "type(mat_return)=",type(mat_return)
#
# print 'mat_return[3:7]=',mat_return[3:7]
#
#
# # print 'mat_return[3:7][1]=',mat_return[3:7][1]
#
# # print mat_return[1][0:5]
#
# mat_contents = sio.loadmat('matrix_return_out.mat')
#
# print 'mat_contents=',mat_contents['a']
# print 'type(mat_contents)=',type(mat_contents['a'])