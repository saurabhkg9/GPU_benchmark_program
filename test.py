#!/usr/bin/python
# -*- coding: utf-8 -*-
from numba import jit, cuda
import numpy as np

# to measure exec time
from timeit import default_timer as timer


# normal function to run on cpu
def cpu_func(a):
    for i in range(10000000):
        a[i] += 1


# function optimized to run on gpu
@jit(target_backend='cuda')
def gpu_func(a):
    for i in range(10000000):
        a[i] += 1

if __name__ == '__main__':
    n = 10000000
    a = np.ones(n, dtype=np.float64)

    start = timer()
    cpu_func(a)
    cpu_time = timer() - start
    print ("Executing without GPU:{:.2f}".format(cpu_time))

    start = timer()
    gpu_func(a)
    gpu_time = timer() - start
    print ("Executing with GPU:{:.2f}".format(gpu_time))

    diff_time = cpu_time - gpu_time
    
    perf_index = (diff_time * 100)/cpu_time
    print ("GPU percentage performance improvement over CPU for this usecase = {:.2f}".format(perf_index))
