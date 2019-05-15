from cconv import pool_3d
import numpy as np

a = np.empty((3,4,4,4,2))
b = np.arange(a.size)
a[...] = b.reshape(a.shape)

pool0, inx = pool_3d(a, 2, 2, 2)

print "a"
print a.shape, a

print "pool"
print pool0.shape
inx_reshape = inx.reshape(-1, 3)
for i in range(pool0.size):
    j = int(pool0.flat[i])
    print np.unravel_index(j, a.shape), "->", np.unravel_index(i, pool0.shape), pool0.flat[i], inx_reshape[i]

#print "index"
#print inx

if True:

    # print mapping
    a_flat = a.flat
    p0_flat = pool0.flat
    delta = 0.01

    for i in xrange(len(a_flat)):
        a_save = a_flat[i]
        a_flat[i] = a_save + delta
        multi_index = np.unravel_index(i, a.shape)
        pool1, _ = pool_3d(a, 2, 2, 2)
        a_flat[i] = a_save
        p1_flat = pool1.flat
        dp = pool1 - pool0
        #print np.max(dp)
        di = np.argmax(dp)
        #print "di =", di
        j_multi = np.unravel_index(di, pool1.shape)
    
        dy = dp.flat[di]
        if dy:
            print "%s -> %s, delta:%f" % (multi_index, j_multi, dp.flat[di])
    
    