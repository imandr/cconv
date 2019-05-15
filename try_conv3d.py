import numpy as np
from cconv import convolve_3d

def convolve_xw(inp, w, mode):
    # inp: (nb, nx, ny, nc_in)
    # w: (nx, ny, nc_in, nc_out)
    # returns (nb, x, y, nc_out)
    
    mode = 0 if mode == 'valid' else 1
    
    inp = inp.transpose((0,4,1,2,3))    # -> [mb, ic, x, y, z]
    w = w.transpose((4,3,0,1,2))        # -> [oc, ic, x, y, z]
    #print "convolve_xw: convolve_3d..."
    return convolve_3d(inp, w, mode).transpose((0,2,3,4,1))        # -> [mb, x,y,z, oc]
    
def convolve_xy(x, y):
    # x: (nb, nx, ny, nc_in)
    # y: (nb, mx, my, nc_out)       (mx,my) < (nx,ny)
    # returns (fx, fy, nc_in, nc_out)
    
    x = x.transpose((4,0,1,2,3))        # -> [ic, mb, x, y, z]
    y = y.transpose((4,0,1,2,3))        # -> [oc, mb, x, y, z]
    #print "convolve_xy: convolve_3d..."
    return convolve_3d(x, y, 0).transpose((2,3,4,0,1))    # ->  [x,y,z,ic,oc]


image = np.ones((1,4,3,2,1))
filter = np.ones((2,2,2,1,1))
#filter[0,0,0,:,:] = 1
filtered = convolve_xw(image, filter, 'valid')
print filtered