#include "Python.h"
#include <math.h>
#include "numpy/ndarraytypes.h"
//#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <stdio.h>
#include <stdlib.h>


#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define ABS(x)  ((x) < 0 ? -(x) : (x))

#define PyArray_GETPTR5(obj, i, j, k, l, m) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1] + \
                                            (k)*PyArray_STRIDES(obj)[2] + \
                                            (l)*PyArray_STRIDES(obj)[3] + \
                                            (m)*PyArray_STRIDES(obj)[4]))
#define PyArray_GETPTR6(obj, i, j, k, l, m, n) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1] + \
                                            (k)*PyArray_STRIDES(obj)[2] + \
                                            (l)*PyArray_STRIDES(obj)[3] + \
                                            (m)*PyArray_STRIDES(obj)[4] + \
                                            (n)*PyArray_STRIDES(obj)[5]))


