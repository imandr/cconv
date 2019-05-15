from cconv import convolve_3d
from scipy.signal import fftconvolve
import numpy as np
import time

def print_array(a):
    for i in range(len(a.flat)):
        print np.unravel_index(i, a.shape), a.flat[i]

x = np.random.random((10,50,50,50,1))
#x = np.zeros((1,5,5,5,1))
#x[0,1,2,3,0] = 1.0
f = np.random.random((3,5,5,5,1))
#f = np.ones((1,2,2,2,1))
#f = np.zeros((1,2,2,2,1))
#f[0,1,1,1,0] = 1.0

t = time.time()
y = convolve_3d(x, f, 1, 1)
t1 = time.time() - t

print "y my:", t1
print y.shape
#print_array(y)

# use fft

y1 = np.empty_like(y)
ff = f[:,::-1,::-1,::-1,:]
t = time.time()
for ib in range(x.shape[0]):
    for oc in range(ff.shape[0]):
        y1[ib,...,oc] = fftconvolve(x[ib,...], ff[oc,...], mode="valid")[...,0]
t2 = time.time() - t

print "yfft:", t2
#print_array(y1)

print np.allclose(y, y1)