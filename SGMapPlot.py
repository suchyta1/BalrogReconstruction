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


def PlotHealPixel(ax, hpIndex, tfile, ofile, dfile, rfile):
    PlotRecon(ax, tfile, 1, hpIndex, kind='plot', plotkwargs={'color':'Blue', 'label':'BT-G', 'lw':0.75})
    PlotRecon(ax, ofile, 1, hpIndex, kind='plot', plotkwargs={'color':'MediumTurquoise', 'label':'BO-G', 'lw':0.75})
    PlotRecon(ax, rfile, 1, hpIndex, errext=3, kind='errorbar', plotkwargs={'color':'Indigo', 'fmt':'o', 'label':'DR-G', 'markersize':3}) 
    PlotRecon(ax, dfile, 1, hpIndex, kind='plot', plotkwargs={'color':'Green', 'label':'DO-G', 'lw':0.75})

    PlotRecon(ax, tfile, 2, hpIndex, kind='plot', plotkwargs={'color':'red', 'label':'BT-S', 'lw':0.75})
    PlotRecon(ax, ofile, 2, hpIndex, kind='plot', plotkwargs={'color':'magenta', 'label':'BO-S', 'lw':0.75})
    PlotRecon(ax, rfile, 2, hpIndex, errext=4, kind='errorbar', plotkwargs={'color':'maroon', 'fmt':'*', 'label':'DR-S', 'markersize':5}) 
    PlotRecon(ax, dfile, 2, hpIndex, kind='plot', plotkwargs={'color':'pink', 'label':'DO-S', 'lw':0.75})

    ax.legend(loc='best', ncol=2)
    ax.set_yscale('log')


def PlotRecon(ax, file, ext, hp, errext=None, kind='plot', coordsext=-2, hpext=-1, plotkwargs={}, normed=True):
    hdus = pyfits.open(file)
    norm = 1.0/hdus[hpext].data['numTruth'] * np.power(10.0,5.0)/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(3600.0, 2.0)

    if not normed:
        norm[:] = 1.0

    if hp is not None:
        hps = hdus[hpext].data['hpIndex']
        line = np.arange(len(hps))[hps==hp][0]
    else:
        line = 0

    coords = hdus[coordsext].data
    vals = hdus[ext].data[line, :] * norm[line]

    ax.set_title(r'Raw Number = %i' %(hdus[hpext].data['numTruth'][line]))

    if errext!=None:
        err = hdus[errext].data[line, :] * norm[line]
    
    if kind=='plot':
        ax.plot(coords, vals, **plotkwargs)
    elif kind=='scatter':
        ax.scatter(coords, vals, **plotkwargs)
    elif kind=='errorbar':
        ax.errorbar(coords, vals, err, **plotkwargs)


    return ax


class ReconPlotter(object):
    def __init__(self, dir, version, hpfield='hpIndex', jfield='jacknife'):
        self.version = version
        self.dir = dir
        self.obsfile = os.path.join(self.dir, 'SG-Balrog-Observed-%s.fits' %(self.version))
        self.truthfile = os.path.join(self.dir,'SG-Balrog-Truth-%s.fits' %(self.version))
        self.datafile = os.path.join(self.dir, 'SG-Data-Observed-%s.fits' %(self.version))
        self.reconfile = os.path.join(self.dir, 'SG-Data-Reconstructed-%s.fits' %(self.version))

        self.hpfield = hpfield
        self.jfield = jfield


    def HealPixels(self):
        return pyfits.open(self.reconfile)[-1].data[self.hpfield]

    def PlotHP(self, ax, hpIndex):
        PlotHealPixel(ax, hpIndex, self.truthfile, self.obsfile, self.datafile, self.reconfile);

    def Plot(self, ax, hpIndex=None):
        PlotHealPixel(ax, hpIndex, self.truthfile, self.obsfile, self.datafile, self.reconfile);


def setkwargs(plotkwargs, kwargs):
    for key in kwargs:
        plotkwargs[key] = kwargs[key]
    return plotkwargs

def SetKwargs(plotkwargs, kwargs1, kwargs2):
    plotkwargs = setkwargs(plotkwargs, kwargs1)
    plotkwargs = setkwargs(plotkwargs, kwargs2)
    return plotkwargs


class HPJPlotter(object):
    def __init__(self, dir, version, hpfield='hpIndex', jfield='jacknife', p2a=1.0e3, pscale=0.27):
        self.version = version
        self.dir = dir
        self.obsfile = os.path.join(self.dir, 'SG-Balrog-Observed-%s.fits' %(self.version))
        self.truthfile = os.path.join(self.dir,'SG-Balrog-Truth-%s.fits' %(self.version))
        self.datafile = os.path.join(self.dir, 'SG-Data-Observed-%s.fits' %(self.version))
        self.reconfile = os.path.join(self.dir, 'SG-Data-Reconstructed-%s.fits' %(self.version))
        self.hpfield = hpfield
        self.jfield = jfield
        self.p2a = p2a
        self.pscale = pscale


    def PlotJErr(self, ax, ext, hp=0, coordsext=-2, hpext=-1, plotkwargs={}, normed=True, plus=2, jd=1, log=False):
        hdus = pyfits.open(self.reconfile)
        hpcut = (hdus[hpext].data[self.hpfield] == hp) & (hdus[hpext].data[self.jfield]!=-1)
        jcut = (hdus[hpext].data[self.hpfield] == hp) & (hdus[hpext].data[self.jfield]==-1)
        lines = np.arange(len(hpcut))[hpcut]
        jline = np.arange(len(hpcut))[jcut][0]

        #norm = 1.0/(hdus[hpext].data['numTruth']) * np.power(10.0,5.0)/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(60.0, 2.0)
        #norm = np.array( [1.0/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(60.0, 2.0)]*len(hdus[hpext].data) )

        nt = hdus[hpext].data['numTruth'][jline]
        area = nt * self.p2a * np.power(self.pscale/60.0, 2)
        norm = jd * 1.0 / area
        norm = np.array([norm]*len(hpcut))

        vals = hdus[ext].data[lines, :] * np.reshape(norm[lines], (len(norm[lines]),1))
        avg = np.average(vals, axis=0)
        std = np.std(vals, axis=0) / np.sqrt(np.sum(hpcut)-1)
        if log:
            std = std / (avg * np.log(10))
            avg = np.log10(avg)

        #print std

        #std = np.std(vals, axis=0)
        #v = hdus[ext+plus].data[jline, :] * norm[jline] / jd
        #err = np.sqrt(std*std + v*v)
        err = std

        #print std, v

        coords = hdus[coordsext].data
        #ax.errorbar(coords, avg, std, **plotkwargs)
        ax.errorbar(coords, avg, err, **plotkwargs)
        return ax



    def Plot(self, ax, curve='recon', obj='galaxy', plotkwargs={}, hp=0, jack=0, log=False):
        if obj=='galaxy':
            ext = 1
            if curve=='recon':
                errext = 3
            else:
                errext = None
        elif obj=='star':
            ext = 2
            if curve=='recon':
                errext = 4
            else:
                errext = None

        if curve=='recon':
            file = self.reconfile
            kind = 'errorbar'
        elif curve=='des':
            file = self.datafile
            kind = 'plot'
        elif curve=='obs':
            file = self.obsfile
            kind = 'plot'
        elif curve=='truth':
            file = self.truthfile
            kind = 'plot'

        return self.PlotRecon(ax, file, ext, hp, jack, errext=errext, kind=kind, plotkwargs=plotkwargs, log=log) 


    def PlotRecon(self, ax, file, ext, hp, jack, errext=None, kind='plot', coordsext=-2, hpext=-1, plotkwargs={}, normed=True, log=False):
        hdus = pyfits.open(file)
        hpcut = (hdus[hpext].data[self.hpfield] == hp)
        jcut = (hdus[hpext].data[self.jfield] == jack)
        line = np.where(hpcut & jcut)[0][0]

        #norm = 1.0/(hdus[hpext].data['numTruth']) * np.power(10.0,5.0)/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(60.0, 2.0)
        #norm = np.array( [1.0/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(60.0, 2.0)]*len(hdus[hpext].data) )

        nt = hdus[hpext].data['numTruth'][line]
        area = nt * self.p2a * np.power(self.pscale/60.0, 2)
        norm = 1.0 / area
        norm = np.array([norm]*len(hpcut))

        #print hdus[hpext].data['numTruth']

        if not normed:
            norm[:] = 1.0


        coords = hdus[coordsext].data
        vals = hdus[ext].data[line, :] * norm[line]
        if log:
            vals = np.log10(vals)

        #ax.set_title(r'Raw Number = %i' %(hdus[hpext].data['numTruth'][line]))

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
    map = {#'nside': 256,
           'nside': 64,
           'nest': False,
           #'version': 'v1.1',
           'version': 'corr',
           'summin': 22.5,
           'summax': 24.5
          }
    colorkwargs = {'gmin': -1, 'gmax': 1, 'smin':-1, 'smax':1}

    rfile = 'SG-Data-Reconstructed-v9.fits'
    #rfile = 'SG-Data-Observed-v9.fits'
    SGRecon2Map(rfile, map['version'], magmin=map['summin'], magmax=map['summax'], nside=map['nside'], nest=map['nest'], colorkwargs=colorkwargs)

