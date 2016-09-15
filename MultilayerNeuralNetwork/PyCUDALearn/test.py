#coding=utf-8
'''
Created on 2014年12月14日

@author: Wangliaofan
'''

import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.driver as drv
import numpy
import time
from pycuda.compiler import SourceModule

if __name__ == '__main__':
    mod = SourceModule("""
    __global__ void multiply_them(float *dest, float *a, float *b)
    {
      const int i = threadIdx.x;
      dest[i] = a[i] * b[i];
    }
    """)
    multiply_them = mod.get_function("multiply_them")

    a = numpy.random.randn(1000).astype(numpy.float32)
    b = numpy.random.randn(1000).astype(numpy.float32)
    
    dest = numpy.zeros_like(a)
    tstart_cuda=time.time()
    multiply_them(
            drv.Out(dest), drv.In(a), drv.In(b),
            block=(1000,1,1), grid=(1,1))
    tend_cuda=time.time()
    
    tstart_py=time.time()
    z=a*b
    tend_py=time.time()
    print "cuda time", tend_cuda - tstart_cuda
    print "numpy time", tend_py - tstart_py 
    #print z-dest
    #ts=time.time()
    a_gpu = gpuarray.to_gpu(numpy.random.randn(10000,10000).astype(numpy.float32))
    b_gpu = gpuarray.to_gpu(numpy.random.randn(10000,10000).astype(numpy.float32))
    ts=time.time()
    a_b = (b_gpu*a_gpu).get()
    te=time.time()
    print "te-ts",te-ts
    
    ts2=time.time()
    c=numpy.random.randn(10000,10000).astype(numpy.float32)
    d=numpy.random.randn(10000,10000).astype(numpy.float32)
    te2=time.time()
    c_d=c*d
    print "te2-ts2",te2-ts2
    pass