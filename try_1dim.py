import numpy as np

def conv(a, b, s1, s2, mode, nout):
    assert s1 == 1 or s2 == 1
    if mode == "valid":
        assert len(a) <= len(b)
        if s2 == 1:
            l = len(b) - len(a) + 1
            l = (l+s1-1)/s1
            c = np.empty((l,))
        else:
            l = len(b) - (len(a)-1) * s2
            assert l >= nout and l < nout + s2
            l = nout
            c = np.empty((l,))
        for i in range(l):
            v = 0.0
            for k in range(len(a)):
                v += a[k] * b[s1*i+s2*k]
            c[i] = v
    else:
        assert s1 == 1 and s2 < 0
        s2 = -s2
        n = len(a)*s2 + len(b) - s2
        assert nout >= n and nout < n + s2
        c = np.zeros((nout,))
        for i in range(n):
            v = 0
            for k in range(len(a)):
                 if k >= (i-len(b)+s2)/s2 and k <= i/s2:
                     v += a[k]*b[i-s2*k]
            c[i] = v
    return c
    
def cmparrays(a, namea, b, nameb):
    if np.allclose(a,b, atol=1.e-4, rtol=1.e-5):    print "OK"
    else:
        close = np.isclose(a, b, atol=1.e-4, rtol=1.e-5)
        for i, c in enumerate(close):
            if not c:
                print "%s[%d]=%s   %s[%d]=%s   diff=%s" % (namea, i, a[i], nameb, i, b[i], a[i]/b[i]-1.0)
            
def check_grads(NX, NF, S):
    
    NY = (NX-NF+1+S-1)/S

    x = np.random.random((NX,))
    f = np.random.random((NF,))
    y_ = np.random.random((NY,))

    def L(x, f, y_):
        y = conv(f, x, S, 1, "valid", None)
        l = np.dot(y, y_)
        g = y_
        return y, l, g
    
    delta = 0.001
    y, l, dldy = L(x, f, y_)
    assert y.shape == y_.shape

    print "NX, NY, NF, S, y.shape=", NX, NY, NF, S, y.shape
    
    dldx = conv(dldy, f, 1, -S, "full", len(x))
    assert dldx.shape == x.shape, "dldx.shape=%s   x.shape=%s" % (dldx.shape, x.shape)


    print "dL/dx"
    dldx_c = np.empty_like(x)
    for i in range(len(x)):
        xsave = x[i]
        x[i] = xsave + delta
        y1, l1, _ = L(x, f, y_)
        x[i] = xsave - delta
        y2, l2, _ = L(x, f, y_)
        x[i] = xsave
        dldx_c[i] = (l1-l2)/(delta*2)

    cmparrays(dldx, "dL/dx analytical", dldx_c, "dL/dx computed")
    
    dldf = conv(dldy, x, 1, S, "valid", len(f))
    assert dldf.shape == f.shape, "dldf.shape=%s   f.shape=%s" % (dldf.shape, f.shape)

    print "dL/df"
    dldf_c = np.empty_like(f)
    for i in range(len(f)):
        fsave = f[i]
        f[i] = fsave + delta
        y1, l1, _ = L(x, f, y_)
        f[i] = fsave - delta
        y2, l2, _ = L(x, f, y_)
        f[i] = fsave
        dldf_c[i] = (l1-l2)/(delta*2)

    cmparrays(dldf, "dL/df analytical", dldf_c, "dL/df computed")

for nx in range(3,100):
    for nf in range(min(nx, 14)):
        for s in range(1,nf):
            check_grads(nx, nf, s) 
    
            
