import sys
import itertools

import numpy as np
from mpi4py import MPI


def scatter(arr):
    if arr!=None:
        arr = np.array_split(arr, MPI.COMM_WORLD.size, axis=0)
    arr = MPI.COMM_WORLD.scatter(arr, root=0)
    return arr

def broadcast(arr):
    return MPI.COMM_WORLD.bcast(arr, root=0)

def gather(arr, dtype=np.float):
    arr = MPI.COMM_WORLD.gather(arr, root=0)
    if MPI.COMM_WORLD.Get_rank()==0:

        n = 0
        while n < len(arr):
            if len(arr[n]) > 0:
                break
            n += 1

        if arr[n].ndim==1:
            arr = itertools.chain.from_iterable(arr)
            arr = np.fromiter(arr, dtype=dtype)
        elif arr[n].ndim==2:
            vlen = arr[n].shape[-1]
            arr = itertools.chain.from_iterable(itertools.chain.from_iterable(arr))
            arr = np.fromiter(arr, dtype=dtype)
            arr = np.reshape(arr, (arr.shape[0]/vlen, vlen))
        elif arr[n].ndim==3:
            vlen = arr[n].shape[-1]
            hlen = arr[n].shape[-2]
            arr = itertools.chain.from_iterable(itertools.chain.from_iterable(itertools.chain.from_iterable(arr)))
            arr = np.fromiter(arr, dtype=dtype)
            arr = np.reshape(arr, (arr.shape[0]/(vlen*hlen), hlen, vlen))

        '''
        vlen = None
        if arr[0].ndim > 1:
            vlen = arr[0].shape[-1]
        for i in range(arr[0].ndim):
            arr = itertools.chain.from_iterable(arr)
        arr = np.fromiter(arr, dtype=np.float)
        if vlen!=None:
            arr = np.reshape(arr, (arr.shape[0]/vlen, vlen))
        '''

    return arr



def Scatter(*args):
    if len(args)==1:
        return scatter(args[0])
    else:
        return [scatter(arg) for arg in args]


def Broadcast(*args):
    if len(args)==1:
        return broadcast(args[0])
    else:
        return [broadcast(arg) for arg in args]

def Gather(*args, **kwargs):
    if len(args)==1:
        return gather(args[0], **kwargs)
    else:
        return [gather(arg, **kwargs) for arg in args]



