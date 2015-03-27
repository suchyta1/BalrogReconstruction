#!/usr/bin/env python

import numpy as np
import numpy.lib.recfunctions as recfunctions
from mpi4py import MPI
import os
import pyfits
import healpy as hp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas

import mpifunctions
import DBfunctions
import MCMC
import healpyfunctions


def MapsFromReconImages(pobj, magmin=22.5, magmax=24.5, cmin=-1, cmax=1):
    ReconHDUs = pyfits.open(pobj.reconfile)
    RawHDUs = pyfits.open(pobj.datafile)

    coords = ReconHDUs[-2].data
    cut = np.where( (coords > magmin) & (coords < magmax) )
    start = np.amin(cut)
    end = np.amax(cut)
    
    ReconCat = ReconHDUs[-1].data
    cut = (ReconCat[pobj.hpfield] > -1) & (ReconCat[pobj.jfield] == -1)
    hps = ReconCat[pobj.hpfield][cut]
    ra, dec = healpyfunctions.Healpix2RaDec(hps, pobj.nside, nest=pobj.nest)
    area = ReconCat['numTruth'][cut] * pobj.pixpergal * np.power(pobj.pscale/60.0, 2)
    norm = 1.0 / area

    ReconGalSample = np.sum(ReconHDUs[1].data[cut][:, start:end], axis=1) * norm
    ReconStarSample = np.sum(ReconHDUs[2].data[cut][:, start:end], axis=1) * norm
    RawGalSample = np.sum(RawHDUs[1].data[cut][:, start:end], axis=1) * norm
    RawStarSample = np.sum(RawHDUs[2].data[cut][:, start:end], axis=1) * norm
    

    fig, axarr = plt.subplots(2, 3, figsize=(16,10))
    axarr[1,2].axis('off')
    ReconGalMap = MakeMap(ReconGalSample, hps, pobj.nside, axarr[0,0], nest=pobj.nest, cmin=cmin, cmax=cmax, title='Reconstructed Galaxy Map')
    RawGalMap = MakeMap(RawGalSample, hps, pobj.nside, axarr[0,1], nest=pobj.nest, cmin=cmin, cmax=cmax, title='Raw Galaxy Map')
    ReconStarMap = MakeMap(ReconStarSample, hps, pobj.nside, axarr[1,0], nest=pobj.nest, cmin=cmin, cmax=cmax, title='Reconstructed Star Map')
    RawStarMap = MakeMap(RawStarSample, hps, pobj.nside, axarr[1,1], nest=pobj.nest, cmin=cmin, cmax=cmax, title='Raw Star Map')

    ys = [0,0, 1,1]
    xs = [0,1, 0,1]
    pp = PdfPages('maps.pdf')
    for i in range(len(hps)):
        canvas = Canvas(fig)

        for j in range(len(ys)):
            axarr[ys[j], xs[j]].plot( [ra[i]], [dec[i]], color='black', marker='x', markersize=10, linestyle='None' )

        ax = axarr[0,2]
        ax.clear()
        ax = pobj.SimplePlot(hp=hps[i], ax=ax, log=True)
        ax.set_xlim( [17.5, 25.5] )
        ax.set_ylim( [-4, 4] )
        axarr[0,2] = ax
        plt.tight_layout()
        pp.savefig(fig, bbox_inches='tight')

        for j in range(len(ys)):
            axarr[ys[j],xs[j]].lines.pop(0)

    pp.savefig(fig)
    pp.close()
    plt.close(fig)



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


def MakeMap(sample, hpIndex, nside, ax, nest=False, cmin=None, cmax=None, title=None):
    Map, mmin, mmax = FractionalOffsetMap(sample, hpIndex, nside)
    cmin = ChooseColorScale(cmin, mmin)
    cmax = ChooseColorScale(cmax, mmax)
    return MakeHealPixMap(Map, ax, nest=nest, vmin=cmin, vmax=cmax, background='gray', title=title)


def MakeHealPixMap(theMap, ax, nest=False, cmap=plt.cm.bwr, vmin=None, vmax=None, background=None, title=None):
    from matplotlib.collections import PolyCollection
    nside = hp.npix2nside(theMap.size)
    mapValue = theMap[theMap > -99]
    indices = np.arange(theMap.size)
    seenInds = indices[theMap > -99]
    
    print "Building polygons from HEALPixel map."
    vertices = np.zeros( (seenInds.size, 4, 2) )
    print "Building polygons for "+str(seenInds.size)+" HEALPixels."
    for HPixel,i in zip(seenInds,xrange(seenInds.size)):
        ns = int(nside)
        hpix = int(HPixel)
        corners = hp.vec2ang( np.transpose(hp.boundaries(ns, hpix, nest=nest) ) )
        vertices[i,:,0] = corners[1] * 180 / np.pi
        vertices[i,:,1] = 90.0 - corners[0] * 180 / np.pi

    #fig, ax = plt.subplots(figsize=(12,12))
    #ax = plt.gca()
    coll = PolyCollection(vertices, array=mapValue, cmap=cmap, edgecolors='none')
    coll.set_clim([vmin,vmax])
    ax.add_collection(coll)
    ax.autoscale_view()
    if background is not None:
        ax.set_axis_bgcolor(background)
    plt.colorbar(coll,ax=ax)
    ax.set_title(title)
    return ax



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

    def __init__(self, dir, version=None, hpfield='hpIndex', jfield='jacknife', pixpergal=1.0e3, pscale=0.27, tiletot=1e5, nside=64, nest=False):
        self.dir = dir
        if version is None:
            version = os.path.basename(dir)
        self.version = version
        self.obsfile = os.path.join(self.dir, 'SG-Balrog-Observed-%s.fits' %(self.version))
        self.truthfile = os.path.join(self.dir,'SG-Balrog-Truth-%s.fits' %(self.version))
        self.datafile = os.path.join(self.dir, 'SG-Data-Observed-%s.fits' %(self.version))
        self.reconfile = os.path.join(self.dir, 'SG-Data-Reconstructed-%s.fits' %(self.version))

        self.hpfield = hpfield
        self.jfield = jfield
        self.pixpergal = pixpergal
        self.pscale = pscale
        self.tiletot = tiletot
        self.nside = nside
        self.nest = nest

        self.healpixels = np.unique( pyfits.open(self.reconfile)[-1].data[self.hpfield] )
        self.jacks = np.unique( pyfits.open(self.reconfile)[-1].data[self.jfield] )


    def PlotJErr(self, ax, ext, hp=0, coordsext=-2, hpext=-1, plotkwargs={}, normed=True, plus=2, jd=1, log=False):
        hdus = pyfits.open(self.reconfile)
        hpcut = (hdus[hpext].data[self.hpfield] == hp) & (hdus[hpext].data[self.jfield]!=-1)
        jcut = (hdus[hpext].data[self.hpfield] == hp) & (hdus[hpext].data[self.jfield]==-1)
        lines = np.arange(len(hpcut))[hpcut]
        jline = np.arange(len(hpcut))[jcut][0]

        #norm = 1.0/(hdus[hpext].data['numTruth']) * np.power(10.0,5.0)/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(60.0, 2.0)
        #norm = np.array( [1.0/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(60.0, 2.0)]*len(hdus[hpext].data) )

        nt = hdus[hpext].data['numTruth'][jline]
        area = nt * self.pixpergal * np.power(self.pscale/60.0, 2)
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



    def Plot(self, ax, curve='recon', obj='galaxy', plotkwargs={}, hp=0, jack=0, log=False, next=None, ncut=-1):
        if obj=='galaxy':
            ext = 1
            if curve=='recon':
                errext = 3
            else:
                errext = None
            next = -4
        elif obj=='star':
            ext = 2
            if curve=='recon':
                errext = 4
            else:
                errext = None
            next = -3

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

        return self.PlotRecon(ax, file, ext, hp, jack, errext=errext, kind=kind, plotkwargs=plotkwargs, log=log, next=next) 


    def PlotRecon(self, ax, file, ext, hp, jack, errext=None, kind='plot', coordsext=-2, hpext=-1, plotkwargs={}, normed=True, log=False, tmask=None, next=None, ncut=-1):
        hdus = pyfits.open(file)
        hpcut = (hdus[hpext].data[self.hpfield] == hp)
        jcut = (hdus[hpext].data[self.jfield] == jack)
        line = np.where(hpcut & jcut)[0][0]

        #norm = 1.0/(hdus[hpext].data['numTruth']) * np.power(10.0,5.0)/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(60.0, 2.0)
        #norm = np.array( [1.0/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(60.0, 2.0)]*len(hdus[hpext].data) )

        nt = hdus[hpext].data['numTruth'][line]
        area = nt * self.pixpergal * np.power(self.pscale/60.0, 2)
        norm = 1.0 / area
        norm = np.array([norm]*len(hpcut))

        #print hdus[hpext].data['numTruth']

        if not normed:
            norm[:] = 1.0


        coords = hdus[coordsext].data
        vals = hdus[ext].data[line, :] * norm[line]
        if log:
            vals = np.log10(vals)


        if errext!=None:
            err = hdus[errext].data[line, :] * norm[line]

        if tmask is not None:
            nobs = pyfits.open(self.truthfile)[next].data[line]
            cut = (nobs > ncut)
            coords = coords[cut]
            vals = vals[cut]
            if errext is not None:
                err = err[cut]
        
        if kind=='plot':
            ax.plot(coords, vals, **plotkwargs)
        elif kind=='scatter':
            ax.scatter(coords, vals, **plotkwargs)
        elif kind=='errorbar':
            ax.errorbar(coords, vals, err, **plotkwargs)

        return ax


    def SimplePlot(self, log=False, hp=-1, jack=-1, ncut=-1, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        '''
        ax = self.PlotJErr(ax, 1, log=log, jd=jd, hp=hp, plotkwargs={'fmt':'o', 'color':'red', 'markersize':3})
        ax = self.PlotJErr(ax, 2, log=log, jd=jd, hp=hp, plotkwargs={'fmt':'o', 'color':'blue', 'markersize':3})
        '''

        ax = self.Plot(ax, log=log, curve='recon', obj='galaxy', hp=hp, jack=-1, ncut=ncut, plotkwargs={'fmt':'o', 'color':'red', 'markersize':3})
        ax = self.Plot(ax, log=log, curve='recon', obj='star', hp=hp, jack=-1, ncut=ncut, plotkwargs={'fmt':'o', 'color':'blue', 'markersize':3})
        ax = self.Plot(ax, log=log, curve='truth', obj='galaxy', hp=hp, jack=-1, ncut=ncut, plotkwargs={'color':'red', 'ls':'dashed'})
        ax = self.Plot(ax, log=log, curve='truth', obj='star', hp=hp, jack=-1, ncut=ncut, plotkwargs={'color':'blue', 'ls':'dashed'})
        ax = self.Plot(ax, log=log, curve='des', obj='galaxy', hp=hp, jack=-1, ncut=ncut, plotkwargs={'color':'red'})
        ax = self.Plot(ax, log=log, curve='des', obj='star', hp=hp, jack=-1, ncut=ncut, plotkwargs={'color':'blue'})
        ax = self.Plot(ax, log=log, curve='obs', obj='galaxy', hp=hp, jack=-1, ncut=ncut, plotkwargs={'color':'red', 'ls':'-.'})
        ax = self.Plot(ax, log=log, curve='obs', obj='star', hp=hp, jack=-1, ncut=ncut, plotkwargs={'color':'blue', 'ls':'-.'})
        return ax


def SimplePlot(version, log=False, jd=10, hp=-1):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plotter = HPJPlotter(version, version)
    ax = plotter.PlotJErr(ax, 1, log=log, jd=jd, hp=hp, plotkwargs={'fmt':'o', 'color':'red', 'markersize':3})
    ax = plotter.PlotJErr(ax, 2, log=log, jd=jd, hp=hp, plotkwargs={'fmt':'o', 'color':'blue', 'markersize':3})
    ax = plotter.Plot(ax, log=log, curve='truth', obj='galaxy', hp=hp, jack=-1, plotkwargs={'color':'red', 'ls':'dashed'})
    ax = plotter.Plot(ax, log=log, curve='truth', obj='star', hp=hp, jack=-1, plotkwargs={'color':'blue', 'ls':'dashed'})
    ax = plotter.Plot(ax, log=log, curve='des', obj='galaxy', hp=hp, jack=-1, plotkwargs={'color':'red'})
    ax = plotter.Plot(ax, log=log, curve='des', obj='star', hp=hp, jack=-1, plotkwargs={'color':'blue'})
    return ax

def SimplerPlot(version, log=False, jd=10, hp=-1):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plotter = HPJPlotter(version, version)
    ax = plotter.PlotJErr(ax, 1, log=log, jd=jd, hp=hp, plotkwargs={'fmt':'o', 'color':'red', 'markersize':3})
    ax = plotter.Plot(ax, log=log, curve='truth', obj='galaxy', hp=hp, jack=-1, plotkwargs={'color':'red', 'ls':'dashed'})
    ax = plotter.Plot(ax, log=log, curve='des', obj='galaxy', hp=hp, jack=-1, plotkwargs={'color':'red'})
    return ax



'''
#recon_truth = [galaxy_recon_truth, star_recon_truth, galaxy_recon_trutherr, star_recon_trutherr, galaxy_recon_truth_center]
class FlexiblePlotter(object):

    """
    commonkwargs:
        hpfield
        jfield
    """

    def __init__(self, dir, hpfield='hpIndex', jfield='jacknife', pixpergal=1.0e3, pscale=0.27):
        self.dir = dir
        self.version = os.path.basename(self.dir)

        self.obsfile = os.path.join(self.dir, 'SG-Balrog-Observed-%s.fits' %(self.version))
        self.truthfile = os.path.join(self.dir,'SG-Balrog-Truth-%s.fits' %(self.version))
        self.datafile = os.path.join(self.dir, 'SG-Data-Observed-%s.fits' %(self.version))
        self.reconfile = os.path.join(self.dir, 'SG-Data-Reconstructed-%s.fits' %(self.version))

        self.hpfield = hpfield
        self.jfield = jfield
        self.pixpergal = pixpergal
        self.pscale = pscale


    def PlotRecon(hp=-1, jack=-1, sg=None, errtype='m'):
        if sg=='galaxy':
            ext = 1
            eext = 3
        elif sg=='star':
            ext = 2
            eext = 4
        elif sg is None:
            ext = 1
            eext = 2
'''



if __name__=='__main__': 
    '''
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
    '''

    plotter = HPJPlotter('sva1v2-i-mbins=0.5-sg=True-Lcut=0')
    MapsFromReconImages(plotter, magmin=22.5, magmax=24.5, cmin=-1, cmax=1)
