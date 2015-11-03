# import matlab.engine
import scipy.io as sio



# eng = matlab.engine.start_matlab()


# # for i in xrange(20):
    
# #     for j in xrange(20):
        
# #         ret = eng.triarea(i*1.0,j*5.0)
# #         print "i=",i," j=",j," area=",ret

# mat_return=eng.matrix_return(10,10)

# print "mat_return=",mat_return

# print "type(mat_return)=",type(mat_return)

# print 'mat_return[3:7]=',mat_return[3:7]


# print 'mat_return[3:7][1]=',mat_return[3:7][1]

# print mat_return[1][0:5]

mat_contents = sio.loadmat('params_return_out.mat')

print 'mat_contents=', type(mat_contents)
print 'mat_contents.keys()=', mat_contents.keys()

print 'mat_contents[params].shape=', mat_contents['params'].shape

print 'params=', mat_contents['params']

# print mat_contents['params']


# mat_contents = sio.loadmat('params.mat')

# print 'mat_contents=', type(mat_contents)
# print 'mat_contents.keys()=',mat_contents.keys()

# print 'mat_contents[params].shape=',mat_contents['params'].shape

# print mat_contents['params']

# eeg = mat_contents['params']['eeg']

# print eeg
# print type(eeg)
# print eeg[0,0]

# print 'type(mat_contents)=',type(mat_contents['a'])