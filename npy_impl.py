import numpy as np
import math, time
from mynnet import Layer

#
# X shape:   (mb, x, y, z, ci)
# F shape:   (co, x, y, z, ci)
# Y shape:   (mb, x, y, z, co)
#

def reshape_not_needed(a, i):
    a_strides = a.__array_interface__["strides"]

    return (a_strides is None and i == len(a.shape)-1) or \
        (a_strides is not None and a_strides[i] == a.dtype.itemsize)
    

def reshape_if_needed(a, i):
    if reshape_not_needed(a, i):    return a
        
    inx = range(len(a.shape))
    tmp = inx[i]
    inx[i] = inx[-1]
    inx[-1] = tmp
    
    a = a.transpose(inx).copy().transpose(inx)
    
    assert reshape_not_needed(a, i)
    
    return a


def convolve_fx(f, x, s):
    f = reshape_if_needed(f, 4)
    x = reshape_if_needed(x, 4)
    x = x[:,
    
    
    
    

class Conv3D(Layer):
    def __init__(self, inp, filter_x, filter_y, filter_z, out_channels, 
                stride = 1,
                name=None, 
                applier = None, weight_decay=1.0e-5):
        # filter_xy_shape is (fh, fw) 
        Layer.__init__(self, [inp], name)
        #print self.InShape
        assert len(inp.shape) == 4
        #
        #  input
        #
        self.filter_xyz_shape = (filter_x, filter_y, filter_z)
        self.out_channels = out_channels
        self.in_channels = inp.shape[3]
        self.filter_shape = self.filter_xyz_shape + (self.in_channels, self.out_channels)   
             # (rows, columns, channels_in, channels_out)
        self.weight_decay = weight_decay
        self.Stride = stride
        ox, oy, oz = inp.shape[0]-filter_x+1, inp.shape[1]-filter_y+1, inp.shape[2]-filter_z+1
        ox, oy, oz = (
            (ox+self.Strides[0]-1)/self.Strides[0], 
            (oy+self.Strides[1]-1)/self.Strides[1], 
            (oz+self.Strides[2]-1)/self.Strides[2]
        ) 
        
        self.OutShape = (ox, oy, oz, out_channels)
