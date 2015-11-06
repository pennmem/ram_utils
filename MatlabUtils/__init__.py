__author__ = 'm'

import sys
from os.path import *

try:

    import matlab.engine

except ImportError,e:
    print 'Seems like support for calling Matlab from Python scripts is not enabled'
    sys.exit()

print "***************** Starting MATLAB Engine *************************"
matlab_engine = matlab.engine.start_matlab()
print "***************** MATLAB Engine WORKING *************************"


def add_matlab_search_paths(*path_strings):
    for path_str in path_strings:
        matlab_engine.addpath(matlab_engine.genpath(abspath(expanduser(path_str))))

        # matlab_engine.matlab_path_set(abspath(expanduser(path_str)))
        # matlab_engine.matlab_path_set(abspath(expanduser('~/matlab_extern1')))