#!/usr/bin/env python

import numpy as np
import numpy.lib.recfunctions as recfunctions
from mpi4py import MPI
import os
import pyfits
import healpy as hp

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mpifunctions
import DBfunctions
import MCMC
import healpyfunctions




def SGRecon2Map(reconfile, version, magmin=22.5, magmax=24.5, nside=256, nest=False, colorkwargs={}):
    hdus = pyfits.open(reconfile)
    #coords = hdus[-2].data[0,:]
    
    tab = hdus[-1].data
    hpIndex = tab['hpIndex']
    norm = 1.0/tab['numTruth'] * np.power(10.0,5.0)/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(3600.0, 2.0)

    coords = hdus[-2].data
    cut = np.where( (coords > magmin) & (coords < magmax) )
    start = np.amin(cut)
    end = np.amax(cut)

    galaxy = np.sum( hdus[1].data[:, start:end], axis=-1) * norm
    star = np.sum( hdus[2].data[:, start:end], axis=-1) * norm
    MakeSGMaps(galaxy, star, hpIndex, nside, version, nest=nest, **colorkwargs)


def FractionalOffsetMap(arr, hpIndex, nside):
    npix = hp.nside2npix(nside)
    map = np.zeros(npix) + hp.UNSEEN

    avg = np.average(arr)
    map[hpIndex] = (arr - avg)/avg
    mapmin = np.percentile(map[hpIndex], 5)
    mapmax = np.percentile(map[hpIndex], 95)
    larger = np.amax( np.fabs([mapmin, mapmax]) )
    mapmin = -larger
    mapmax = larger
    return map, mapmin, mapmax


def ChooseColorScale(choice, default):
    if choice != None:
        return choice
    else:
        return default


def MakeSGMaps(galaxy, star, hpIndex, nside, version, nest=False, gmin=None, gmax=None, smin=None, smax=None):
    GalMap, galmin, galmax = FractionalOffsetMap(galaxy, hpIndex, nside)
    StarMap, starmin, starmax = FractionalOffsetMap(star, hpIndex, nside)
    galmin = ChooseColorScale(gmin, galmin)
    galmax = ChooseColorScale(gmax, galmax)
    starmin = ChooseColorScale(smin, starmin)
    starmax = ChooseColorScale(smax, starmax)
    healpyfunctions.VisualizeHealPixMap(GalMap, title='Galaxy-map-%s'%(version), nest=nest, vmin=galmin, vmax=galmax, background='gray')
    healpyfunctions.VisualizeHealPixMap(StarMap, title='Star-map-%s'%(version), nest=nest, vmin=starmin, vmax=starmax, background='gray')


def PlotRecon(ax, file, ext, hp, errext=None, kind='plot', coordsext=-2, hpext=-1, plotkwargs={}):
    hdus = pyfits.open(file)
    hps = hdus[hpext].data['hpIndex']
    norm = 1.0/hdus[hpext].data['numTruth'] * np.power(10.0,5.0)/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(3600.0, 2.0)

    line = np.arange(len(hps))[hps==hp][0]
    coords = hdus[coordsext].data
    vals = hdus[ext].data[line, :] * norm[line]


    if errext!=None:
        err = hdus[errext].data[line, :] * norm[line]
    
    if kind=='plot':
        ax.plot(coords, vals, **plotkwargs)
    elif kind=='scatter':
        ax.scatter(coords, vals, **plotkwargs)
    elif kind=='errorbar':
        ax.errorbar(coords, vals, err, **plotkwargs)

    return ax




if __name__=='__main__': 
    map = {'nside': 256,
           'nest': False,
           'version': 'v1.1',
           'summin': 22.5,
           'summax': 24.5
          }
    colorkwargs = {'gmin': -1, 'gmax': 1, 'smin':-1, 'smax':1}
    rfile = 'SG-Data-Reconstructed-v1.fits'
    SGRecon2Map(rfile, map['version'], magmin=map['summin'], magmax=map['summax'], nside=map['nside'], nest=map['nest'], colorkwargs=colorkwargs)

