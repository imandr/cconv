import cconv
import numpy as np

a = np.random.random((3, 5, 7, 3))
print a.shape, a
print "pool..."
a_pool, index = cconv.pool(a, 1, a.shape[2])
print a_pool.shape, a_pool
print index.shape, index
g = -a_pool*100
print "back..."
print g
x = cconv.pool_back(g, index, 2, 2, a.shape[1], a.shape[2])
print "done"

print "x:", x.shape, x.dtype, x


