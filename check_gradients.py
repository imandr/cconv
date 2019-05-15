import numpy as np
from mynnet import Conv, Pool, Conv3D, Pool3D, Linear, L2Regression, Model, InputLayer, Flatten
from mynnet import Tanh, Sigmoid, ReLU, Reshape, LogRegression, AdaDeltaApplier

def pool(nout):

	inp = InputLayer((9,8,5))
	pool = Pool(inp, (3,4))
	return Model(inp, pool, L2Regression(pool))

def cnn2d(nout):
    inp = InputLayer((10,10,3))
    c = Conv(inp, 3, 3, 3)
    p = Pool(c, (2,2))
    
    f = Flatten(p)
    loss = L2Regression(f)
    nn = Model(inp, f, loss)
    #nparams = 0
    #for l in nn.layers:
    #    if isinstance(l, ParamMixin):
    #        for p in l.params():
    #            nparams += p.size
    #print nparams
    return nn
    
def cnn3d(nout):
    inp = InputLayer((20,20,20, 1))
    c1 = Conv3D(inp, 3, 3, 3, 5)
    p1 = Pool3D(c1, (2,2,2))
    c2 = Conv3D(p1, 3, 3, 3, 10)
    p2 = Pool3D(c2, (2,2,2))
    
    f = Flatten(p2)
    loss = L2Regression(f)
    nn = Model(inp, f, loss)
    #nparams = 0
    #for l in nn.layers:
    #    if isinstance(l, ParamMixin):
    #        for p in l.params():
    #            nparams += p.size
    #print nparams
    return nn

def model(img_size):
    
        inp = InputLayer(img_size)
    
        r = Reshape(inp, inp.shape + (1,))
    
        c1 = ReLU(
            Pool3D(
                Conv3D(r, 5, 5, 5, 20),
                (2, 2, 2)
            )
        )
        print c1.shape
        c2 = ReLU(
            Pool3D(
                Conv3D(c1, 3, 3, 3, 30),
                (2, 2, 2)
            )
        )
        print c2.shape
        f = Flatten(c2)
        print f.shape
    
        l = Tanh(Linear(f, 20))
        out = Linear(l, 2)
        return Model([inp], out, LogRegression(out), applier_class=AdaDeltaApplier)
   
def simple(img_size):
    inp = InputLayer(img_size)
    r = Reshape(inp, inp.shape + (1,))
    c = Conv3D(r, 3, 3, 3, 2)
    print "c.shape=",c.shape
    p = Pool3D(c, (2, 2, 2))
    print "p.shape=",p.shape
    f = Flatten(p)
    print "f.shape=", f.shape
    m = Model([inp], f, L2Regression(f))
    x = np.random.random((3,)+img_size)
    y = m(x)
    print "c.out=", c.Y.shape
    print "p.out=", p.Y.shape
    print "f.out=", f.Y.shape
    print y.shape


nn = model((10,10,10))

#x = np.random.random((1,28,28))
#y_ = nn(x)
#x = np.random.random((1,28,28))

nn.checkGradients()

