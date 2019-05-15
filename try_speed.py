import numpy as np
import time

def optimize(a):
    strides = a.__array_interface__["strides"]
    if strides is None:
        return a

    if strides == sorted(strides, reverse=True) and strides[-1] == a.itemsize:
        return a

    return a.copy()
    
def smart_mult(a,b,n):
    # assume 2-dimensional arrays
    t = time.time()
    for _ in xrange(n):
        c = np.einsum("ik,kj->ij", a, b)
    return time.time()-t, c


def mult(a,b,n):
    t = time.time()
    for _ in xrange(n):
        c = a.dot(b)
    return time.time() - t, c
    
    
a = np.random.random((100,100))
b = np.random.random((100,100))

t1, m1 = mult(a,b, 1000)
print t1
t2, m2 = smart_mult(a,b, 1000)
print t2

print np.allclose(m1, m2)
