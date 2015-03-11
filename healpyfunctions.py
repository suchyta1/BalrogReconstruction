import healpy as hp
import numpy as np
import numpy.lib.recfunctions as recfunctions

from mpi4py import MPI
import mpifunctions

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def RaDec2Healpix(ra, dec, nside, nest=False):
    phi = np.radians(ra)
    theta = np.radians(90.0 - dec)
    hpInd = hp.ang2pix(nside, theta, phi, nest=nest)
    return hpInd


def SortOnHP(arr, rcoord, dcoord, nside, nest, field='hpIndex'):
    hpInd = RaDec2Healpix(arr[rcoord], arr[dcoord], nside, nest=nest)
    arr = recfunctions.append_fields(arr, field, hpInd)
    arr = np.sort(arr, order=field)
    return arr

def Split(arr, uinds, field='hpIndex'):
    arr = arr[ np.in1d(arr[field], uinds) ]
    splitat = np.searchsorted(arr['hpIndex'], uinds[1:])
    return np.split(arr, splitat)

def SplitByHealpix(truth, sim, nosim, des, balrogcoords, descoords, band, nside, nest, field='hpIndex'):
    balrogra = '%s_%s' %(balrogcoords[0], band)
    balrogdec = '%s_%s' %(balrogcoords[1], band)
    desra = '%s_%s' %(descoords[0], band)
    desdec = '%s_%s' %(descoords[1], band)

    truth = SortOnHP(truth, balrogra, balrogdec, nside, nest, field=field)
    sim = SortOnHP(sim, balrogra, balrogdec, nside, nest, field=field)
    nosim = SortOnHP(nosim, balrogra, balrogdec, nside, nest, field=field)
    des = SortOnHP(des, desra, desdec, nside, nest, field=field)

    uInds = np.unique(truth[field])
    uInds = np.intersect1d(uInds, sim[field])
    uInds = np.intersect1d(uInds, nosim[field])
    uInds = np.intersect1d(uInds, des[field])

    truth = Split(truth, uInds, field=field)
    sim = Split(sim, uInds, field=field)
    nosim = Split(nosim, uInds, field=field)
    des = Split(des, uInds, field=field)
   
    return truth, sim, nosim, des 


def ScatterByHealPixel(truth, sim, nosim, des, band, balrogcoords=['ra','dec'], descoords=['alphawin_j2000','deltawin_j2000'], nside=256, nest=False):
    if MPI.COMM_WORLD.Get_rank()==0:
        truth, sim, nosim, des = SplitByHealpix(truth, sim, nosim, des, balrogcoords, descoords, band, nside, nest, field='hpIndex')
    truth, sim, nosim, des = mpifunctions.Scatter(truth, sim, nosim, des)
    return truth, sim, nosim, des

def MakeMaps(Recon, nside, version, nest=False):
    npix = hp.nside2npix(nside)
    GalMap = np.zeros(npix) + hp.UNSEEN
    StarMap = np.zeros(npix) + hp.UNSEEN

    galavg = np.average(Recon['gal'])
    GalMap[Recon['hpIndex']] = (Recon['gal'] - galavg)/galavg
    galmin = np.percentile(GalMap[Recon['hpIndex']], 5)
    galmax = np.percentile(GalMap[Recon['hpIndex']], 95)

    staravg = np.average(Recon['star'])
    StarMap[Recon['hpIndex']] = (Recon['star'] - staravg)/staravg
    starmin = np.percentile(StarMap[Recon['hpIndex']], 5)
    starmax = np.percentile(StarMap[Recon['hpIndex']], 95)
   
    VisualizeHealPixMap(GalMap, title='Galaxy-map-%s'%(version), nest=nest, vmin=galmin, vmax=galmax, background='gray')
    VisualizeHealPixMap(StarMap, title='Star-map-%s'%(version), nest=nest, vmin=starmin, vmax=starmax, background='gray')


def VisualizeHealPixMap(theMap, nest=False, title="map", cmap=plt.cm.bwr, vmin=None, vmax=None, background=None):
    
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

    fig, ax = plt.subplots(figsize=(12,12))
    coll = PolyCollection(vertices, array = mapValue, cmap = cmap, edgecolors='none')
    coll.set_clim([vmin,vmax])
    ax.add_collection(coll)
    ax.set_title(title)
    ax.autoscale_view()
    if background is not None:
        ax.set_axis_bgcolor(background)
    fig.colorbar(coll,ax=ax)
    print "Writing to file: "+title+".png"
    fig.savefig(title+".png",format="png")


