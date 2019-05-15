import cconv
import numpy as np
from PIL import Image
import sys

def convolve_xw(inp, w, mode):
    # inp: (nb, nx, ny, nc_in)
    # w: (nx, ny, nc_in, nc_out)
    # returns (nb, x, y, nc_out)
    
    mode = 0 if mode == 'valid' else 1
    
    inp = inp.transpose((0,3,1,2))
    w = w.transpose((3,2,0,1))
    return cconv.convolve(inp, w, mode).transpose((0,2,3,1))
    
def normalize(ar, vmax):
    amin = np.min(ar)
    amax = np.max(ar)
    return vmax*(ar-amin)/(amax-amin)
    
#bw+smooth
f1 = np.ones((3,3,3))
f1[1,1,:] += 1.0
f1 /= 30.0

#bw+sharp
f3 = -np.ones((3,3,3))/8.0/2.0            #    p = p + (p-mean)/2 = p*1.5 - mean/2
f3[1,1,:] = 1.5
f3/=3.0

    
# (2*red+green+blue)/4
f2 = np.zeros((3,3,3))
f2[1,1,0] = 2.0
f2[1,1,1] = 1.0
f2[1,1,2] = 1.0
f2 /= 4.0
        
w = np.empty((3,3,3,3))
w[:,:,:,0] = f1
w[:,:,:,1] = f2
w[:,:,:,2] = f3
        
    
fn = sys.argv[1]
fn0, typ = fn.rsplit('.',1)
fn1 = fn0+"_smooth"+".png"
fn2 = fn0+"_rrgb"+".png"
fn3 = fn0+"_sharp"+".png"

img = Image.open(fn)
nx, ny = img.size

imgarray = np.array(list(img.getdata()))
print imgarray.shape

imgarray = imgarray.reshape((1, nx, ny, 3))

filtered = np.uint8(normalize(convolve_xw(imgarray, w, 1), 256))
filtered1 = Image.fromarray(filtered[0,:,:,0])
filtered2 = Image.fromarray(filtered[0,:,:,1])
filtered3 = Image.fromarray(filtered[0,:,:,2])

filtered1.save(fn1, "png")
filtered2.save(fn2, "png")
filtered3.save(fn3, "png")




    
    