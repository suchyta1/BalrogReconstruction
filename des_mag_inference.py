#!/usr/bin/env python

import desdb
import numpy as np
import esutil
import pyfits
import sys
import argparse
import healpy as hp
import os
import functions2
import slr_zeropoint_shiftmap as slr
import numpy.lib.recfunctions as rf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def doInference(sim_truth_catalog, sim_truth_matched_catalog, sim_obs_catalog, real_obs_catalog, lambda_reg = .01, tag='data', doplot = False, obs_bins = None, truth_bins = None):
    # --------------------------------------------------
    # Settle on a binning scheme for the data.
    # Use the Freedman-Diaconis rule, which suggests:
    #    dx = 2 * IQR(x)/n^(1/3)
    #    IQR is the interQuartile range, and n is the number of data points.
    if obs_bins is None:
        obs_binsize = 0.5#*(np.percentile(sim_obs_catalog[tag],75) - np.percentile(sim_obs_catalog[tag],25))/(len(sim_obs_catalog[tag]))**(1/3.)
        obs_nbins = int(np.ceil( (np.max( sim_obs_catalog[tag]) - np.min( sim_obs_catalog[tag] ) ) / obs_binsize) + 1) 
        obs_bins = np.concatenate( (np.array([np.min( sim_obs_catalog[tag] )])-0.001*obs_binsize,np.array([np.min( sim_obs_catalog[tag] )]) + obs_binsize*np.arange(obs_nbins)) )
    else:
        obs_nbins = obs_bins.size-1
        obs_binsize = obs_bins[1]-obs_bins[0]
    obs_bins_centers = (obs_bins[0:-1] + obs_bins[1:])/2.

    # --------------------------------------------------
    # Next, create the data histogram arrays.
    N_sim_obs, _   = np.histogram(sim_obs_catalog[tag], bins = obs_bins)
    N_real_obs, _  = np.histogram(real_obs_catalog[tag], bins = obs_bins)
    #--------------------------------------------------
    # Important: we can only reconstruct the input histogram in bins from which objects are actually detected.
    if truth_bins is None:
        truth_binsize = 0.5#2* (np.percentile(sim_truth_matched_catalog[tag],75) - np.percentile(sim_truth_matched_catalog[tag],25))/(len(sim_truth_matched_catalog[tag]))**(1/3.)
        truth_nbins = int(np.ceil( (np.max(sim_truth_matched_catalog[tag]) - np.min(sim_truth_matched_catalog[tag] ) ) / truth_binsize) + 1) 
        truth_bins = np.concatenate( (np.array([np.min( sim_truth_matched_catalog[tag] )])-0.001*truth_binsize,
                                    np.array([np.min( sim_truth_matched_catalog[tag] )]) + truth_binsize*np.arange(truth_nbins)) )
    else:
        truth_nbins = truth_bins.size-1
        truth_binsize = truth_bins[1]-truth_bins[0]
    truth_bins_centers = (truth_bins[0:-1] + truth_bins[1:])/2.

    # The binning scheme is decided, so now we make histograms.
    N_sim_truth, _ = np.histogram(sim_truth_catalog[tag], bins = truth_bins)
    N_sim_truth_sorted, _  = np.histogram(sim_truth_matched_catalog[tag], bins=truth_bins)

    obs_bin_index = np.digitize(sim_obs_catalog[tag],obs_bins)-1
    truth_bin_index = np.digitize(sim_truth_matched_catalog[tag],truth_bins)-1
    indicator = np.zeros( (truth_nbins, obs_nbins) )
    A = np.zeros( (obs_nbins, truth_nbins) )
    Acount = np.zeros( (obs_nbins, truth_nbins) )
    # Finally, compute the response matrix.
    for i in xrange(obs_bin_index.size):
        if N_sim_truth_sorted[truth_bin_index[i]] > 0:
            A[obs_bin_index[i],truth_bin_index[i]] = ( A[obs_bin_index[i],truth_bin_index[i]]+
                                                       1./N_sim_truth_sorted[truth_bin_index[i]] )
            Acount[obs_bin_index[i], truth_bin_index[i]] = Acount[obs_bin_index[i], truth_bin_index[i]] +1


    lambda_reg = 0.001
    lambda_reg_cov = 1e-12

    # Try a power-law structure to the errors?
    N_prior = N_sim_truth[1] * (truth_bins_centers / truth_bins_centers[1])**(20.)
    N_prior_obs = np.dot(A, N_prior)
    C_prelim = np.diag(N_prior + lambda_reg_cov)
    Cinv_prelim = np.linalg.inv(C_prelim + np.diag(np.zeros(truth_nbins)+lambda_reg_cov))
    C_data = np.dot( np.dot( A, C_prelim), np.transpose(A) )
    Cinv_data = np.linalg.inv(C_data + np.diag(np.zeros(obs_nbins)+lambda_reg_cov))
    #Ainv_reg = np.dot( np.linalg.pinv(np.dot( np.dot( np.transpose(A), Cinv_data ), A) + lambda_reg *Cinv_prelim ), np.dot( np.transpose( A ), Cinv_data) )
    Ainv_reg = np.dot( np.linalg.pinv(np.dot( np.transpose(A), A) + lambda_reg * np.identity( N_sim_truth.size )  ), np.dot( np.transpose( A ), Cinv_data) )

        
    # Everything.
    window = np.dot( Ainv_reg, N_sim_obs) / ( N_sim_truth + lambda_reg_cov)
    detFrac = N_sim_truth_sorted* 1.0 / (N_sim_truth + lambda_reg_cov)
    N_real_truth_nocorr = np.dot( Ainv_reg, N_real_obs)
    N_real_truth = N_real_truth_nocorr / window
    N_real_truth[~np.isfinite(N_real_truth)] = 0.
    
    Covar_orig = np.diag(N_real_truth + lambda_reg_cov)
    Amod = np.dot( Ainv_reg, A )
    Covar= np.dot(  np.dot(Amod, Covar_orig), np.transpose(Amod) )

    leakage = np.dot( np.transpose(Amod - np.diag(Amod)) , N_real_truth)
    Covar_orig = Covar_orig
    errors = np.sqrt(np.diag(Covar) + leakage**2)
    #errors = np.sqrt(np.diag(Covar))
    truth_bins_centers = (truth_bins[0:-1] + truth_bins[1:])/2.
    obs_bins_centers = (obs_bins[0:-1] + obs_bins[1:])/2.

    if doplot:
        from matplotlib.colors import LogNorm
        plt.hist2d(sim_truth_matched_catalog['mag'],sim_obs_catalog['mag'],bins = (truth_bins,obs_bins),norm=LogNorm(),normed=True)
        plt.colorbar()
        plt.xlabel("truth magnitude")
        plt.ylabel("obs magnitude")
        plt.show(block=True)
    return N_real_truth, truth_bins_centers, truth_bins, obs_bins, errors




def NoSimFields(band='i'):
    q = """
    SELECT
        balrog_index,
        mag_auto as mag,
        flags
    FROM
        SUCHYTA1.balrog_sva1v2_nosim_%s
    """ %(band)
    return q



def SimFields(band='i',table='sva1v2'):
    q = """
    SELECT
        t.tilename as tilename,
        m.xwin_image as xwin_image,
        m.ywin_image as ywin_image,
        m.xmin_image as xmin_image,
        m.ymin_image as ymin_image,
        m.xmax_image as xmax_image,
        m.ymax_image as ymax_image,
        m.balrog_index as balrog_index,
        m.alphawin_j2000 as ra,
        m.deltawin_j2000 as dec,
        m.mag_auto as mag,
        t.mag as truth_mag,
        m.flags as flags
    FROM
        SUCHYTA1.balrog_%s_sim_%s m
        JOIN SUCHYTA1.balrog_%s_truth_%s t ON t.balrog_index = m.balrog_index
    """ %(table, band, table, band)
    return q



def DESFields(tilestuff, band='i'):
    q = """
        SELECT
           tilename,
           coadd_objects_id,
           mag_auto_%s as mag,
           alphawin_j2000_%s as ra,
           deltawin_j2000_%s as dec,
           flags_%s as flags
        FROM
           sva1_coadd_objects
        WHERE
           tilename in %s
        """ % (band,band,band, band, str(tuple(np.unique(tilestuff['tilename']))))
    return q


def TruthFields(band='i', table = 'sva1v2'):
    q = """
    SELECT
        balrog_index,
        tilename,
        ra,
        dec,
        mag
    FROM
        SUCHYTA1.balrog_%s_truth_%s        
    """%(table,band)
    return q
    

def GetDESCat( depthmap, nside, tilestuff, tileinfo, band='i',depth = 50.0):
    cur = desdb.connect()
    q = DESFields(tileinfo, band=band)
    detcat = cur.quick(q, array=True)
    detcat = functions2.ValidDepth(depthmap, nside, detcat, rakey='ra', deckey='dec',depth = depth)
    detcat = functions2.RemoveTileOverlap(tilestuff, detcat, col='tilename', rakey='ra', deckey='dec')
    return detcat



def getTileInfo(catalog, HealConfig=None):
    if HealConfig is None:
        HealConfig = getHealConfig()
        
    tiles = np.unique(catalog['tilename'])
    cur = desdb.connect()
    q = "SELECT tilename, udecll, udecur, urall, uraur FROM coaddtile"
    tileinfo = cur.quick(q, array=True)
    tilestuff = {}
    for i in range(len(tileinfo)):
        tilestuff[ tileinfo[i]['tilename'] ] = tileinfo[i]
    max = np.power(map_nside/float(HealConfig['out_nside']), 2.0)
    depthmap, nside = functions2.GetDepthMap(HealConfig['depthfile'])
    return depthmap, nside
    


def cleanCatalog(catalog, tag='data'):
    # We should get rid of obviously wrong things.
    keep = np.where( (catalog[tag] > 15. ) & (catalog[tag] < 30.) & (catalog['flags'] < 2) )
    return catalog[keep]


def removeBadTilesFromTruthCatalog(truth, tag='data', goodfrac = 0.8):
    tileList = np.unique(truth['tilename'])
    number = np.zeros(tileList.size)
    for tile, i in zip(tileList,xrange(number.size)):
        number[i] = np.sum(truth['tilename'] == tile)
    tileList = tileList[number > goodfrac*np.max(number)]
    keep = np.in1d( truth['tilename'], tileList )
    return truth[keep]
    

def mergeCatalogsUsingPandas(sim=None, truth=None, key='balrog_index'):
    import pandas as pd
    simData = pd.DataFrame(sim)
    truthData = pd.DataFrame(truth)
    matched = pd.merge(simData, truthData, on=key, suffixes = ['_sim',''])
    matched_arr = matched.to_records()
    return matched_arr
    

def GetFromDB( band='i', depth = 50.0, tables = ('sva1v2','sva1v3')):
    depthfile = '../sva1_gold_1.0.2-4_nside4096_nest_i_auto_weights.fits'
    slrfile = '../slr_zeropoint_shiftmap_v6_splice_cosmos_griz_EQUATORIAL_NSIDE_256_RING.fits'

    cur = desdb.connect()
    q = "SELECT tilename, udecll, udecur, urall, uraur FROM coaddtile"
    tileinfo = cur.quick(q, array=True)
    tilestuff = {}
    for i in range(len(tileinfo)):
        tilestuff[ tileinfo[i]['tilename'] ] = tileinfo[i]
    depthmap, nside = functions2.GetDepthMap(depthfile)
    slr_map = slr.SLRZeropointShiftmap(slrfile, band)
    truths = []
    sims = []
    truthMatcheds = []
    
    for tableName in tables:
        q = TruthFields(band=band,table=tableName)
        truth = cur.quick(q, array=True)

        truth = removeBadTilesFromTruthCatalog(truth)
        truth = functions2.ValidDepth(depthmap, nside, truth, depth = depth)
        truth = functions2.RemoveTileOverlap(tilestuff, truth)
        slr_mag, slr_quality = slr_map.addZeropoint(band, truth['ra'], truth['dec'], truth['mag'], interpolate=True)

        q = SimFields(band=band, table=tableName)
        sim = cur.quick(q, array=True)
        sim = cleanCatalog(sim,tag='mag')
        unique_binds, unique_inds = np.unique(sim['balrog_index'],return_index=True)
        sim = sim[unique_inds]
        
        truthMatched = mergeCatalogsUsingPandas(sim=sim,truth=truth)
        
        sim = sim[np.in1d(sim['balrog_index'],truthMatched['balrog_index'])]
        sim.sort(order='balrog_index')
        truthMatched.sort(order='balrog_index')

        
        truthMatcheds.append(truthMatched)
        truths.append(truth)
        sims.append(sim)

    sim = np.hstack(sims)
    truth = np.hstack(truths)
    truthMatched = np.hstack(truthMatcheds)
    
    des = GetDESCat(depthmap, nside, tilestuff, sim, band=band,depth = depth)
    #slr_mag, slr_quality = slr_map.addZeropoint(band, des['ra'], des['dec'], des['mag'], interpolate=True)
    #des['mag'] = slr_mag
    des = cleanCatalog(des, tag='mag')
    
    return des, sim, truthMatched, truth, tileinfo
    

def getCatalogs(reload=False,band='i'):

    # Check to see whether the catalog files exist.  If they do, then
    # use the files. If at least one does not, then get what we need
    # from the database

    fileNames = ['desCatalogFile-'+band+'.fits','BalrogObsFile-'+band+'.fits',
                 'BalrogTruthFile-'+band+'.fits', 'BalrogTruthMatchedFile-'+band+'.fits',
                 'BalrogTileInfo.fits']
    exists = True
    for thisFile in fileNames:
        print "Checking for existence of: "+thisFile
        if not os.path.isfile(thisFile): exists = False
    if exists and not reload:
        desCat = esutil.io.read(fileNames[0])
        BalrogObs = esutil.io.read(fileNames[1])
        BalrogTruth = esutil.io.read(fileNames[2])
        BalrogTruthMatched = esutil.io.read(fileNames[3])
        BalrogTileInfo = esutil.io.read(fileNames[4])
    else:
        print "Cannot find files, or have been asked to reload. Getting data from DESDB."
        desCat, BalrogObs, BalrogTruthMatched, BalrogTruth, BalrogTileInfo = GetFromDB(band=band)
        esutil.io.write( fileNames[0], desCat , clobber=True)
        esutil.io.write( fileNames[1], BalrogObs , clobber=True)
        esutil.io.write( fileNames[2], BalrogTruth , clobber=True)
        esutil.io.write( fileNames[3], BalrogTruthMatched , clobber=True)
        esutil.io.write( fileNames[4], BalrogTileInfo, clobber=True)
        
    return desCat, BalrogObs, BalrogTruthMatched, BalrogTruth, BalrogTileInfo


def makeHistogramPlots(hist_est, bin_centers, errors, catalog_real_obs, catalog_sim_obs, catalog_sim_truth,
                       catalog_sim_truth_matched,
                       bin_edges = None, tag='data'):

    hist_sim, _ = np.histogram(catalog_sim_truth[tag], bins = bin_edges)
    hist_sim_obs, _  = np.histogram(catalog_sim_obs[tag], bins = bin_edges)
    hist_obs, obs_bin_edges  = np.histogram(catalog_real_obs[tag],bins = bin_edges)
    obs_bin_centers = (obs_bin_edges[0:-1] + obs_bin_edges[1:])/2.

    peak_location = np.where(hist_obs == np.max(hist_obs) )[0]
    if len(peak_location) > 1:
        peak_location = peak_location[0]
    norm_factor = np.sum(hist_obs[(peak_location-2):(peak_location+2)])*1. / np.sum(hist_sim_obs[(peak_location-2):(peak_location+2)])
    hist_sim_renorm = hist_sim*norm_factor
    hist_sim_obs_renorm = hist_sim_obs*norm_factor
    print "Number of objects detected: ",catalog_real_obs.size
    print "Number of objects recovered: ", np.sum(hist_est)
    fig = plt.figure(1, figsize=(14,7))
    ax = fig.add_subplot(1,2,1)
    ax.semilogy(bin_centers, hist_est,'.', c='blue', label='inferred')
    ax.errorbar(bin_centers, hist_est,np.clip(errors,1.,1e9), c='blue',linestyle=".")
    ax.plot(obs_bin_centers, hist_obs, c='black',label='observed')
    ax.plot(bin_centers, hist_sim_renorm,c='green', label='simulated')
    ax.plot(bin_centers, hist_sim_obs_renorm, c = 'orange', label = 'sim. observed')
    ax.legend(loc='best')
    ax.set_ylim([1000,1e7])
    ax.set_xlim([15,30])
    ax.set_ylabel('Number')
    ax.set_xlabel('magnitude')
    
    ax = fig.add_subplot(1,2,2)
    ax.axhspan(-.1,.1,facecolor='red',alpha=0.2)
    ax.axhline(y=0.0,color='grey',linestyle='--')
    ax.plot(bin_centers, (hist_est/(hist_sim_renorm+1e-12)-1),'.',color='blue')
    ax.errorbar(bin_centers, (hist_est/(hist_sim_renorm+1e-12)-1), errors/(hist_sim_renorm+1e-6), linestyle=".",c='blue')
    ax.set_xlabel('magnitude')
    ax.set_ylabel('normalized reconstruction residuals')
    ax.set_ylim([-1,1])
    ax.set_xlim([15,30])
    plt.show(block=True)
    


def HealPixifyCatalogs(catalog, HealConfig):
    
    HealInds = functions2.GetPix(HealConfig['out_nside'], catalog['ra'], catalog['dec'], nest=True)
    healCat = rf.append_fields(catalog,'HealInd',HealInds,dtypes=HealInds.dtype)

    return healCat



def getHealConfig(map_nside = 4096, out_nside = 128, depthfile = '../sva1_gold_1.0.2-4_nside4096_nest_i_auto_weights.fits'):
    HealConfig = {}
    HealConfig['map_nside'] = map_nside
    HealConfig['out_nside'] = out_nside
    HealConfig['finer_nside'] = map_nside
    HealConfig['depthfile'] = depthfile
    return HealConfig

          
def getEffectiveArea( catalog, areaMap, depthMap, HealConfig ,depth=0., rakey = 'ra',deckey = 'dec'):
    # create a list of tile indices from the catalog.
    catalogIndex = functions2.GetPix(HealConfig['out_nside'],catalog[rakey], catalog[deckey],nest=True)
    tileIndices = np.unique(catalogIndex)
    # Make a much finer grid.
    finer_index = np.arange(hp.nside2npix(HealConfig['finer_nside']))
    finer_theta, finer_phi = hp.pix2ang(HealConfig['finer_nside'], finer_index,nest=True)
    # Reject those tileIndices that aren't allowed by the depthmap(?)
    cut =  (areaMap > 0.0) & (depthMap > 0.0)
    areaMap = areaMap[cut]
    finer_pix = hp.ang2pix(HealConfig['out_nside'], finer_theta[cut], finer_phi[cut],nest=True)
    # Count the number of finer gridpoints inside each tileIndex.
    area = np.zeros(tileIndices.size)
    maxGridsInside =  np.power(HealConfig['map_nside']*1./HealConfig['out_nside'], 2.0)
    for tileIndex,i in zip(tileIndices,np.arange(tileIndices.size) ):
        area[i] = np.mean(areaMap[finer_pix == tileIndex] ) * hp.nside2pixarea(HealConfig['out_nside'],
                                                                               degrees=True) / maxGridsInside
    return area, tileIndices


def removeNeighbors(thing1, thing2, radius= 2./3600):
    # Returns the elements of thing 1 that are outside of the matching radius from thing 2
    
    depth=10
    h = esutil.htm.HTM(depth)
    m1, m2, d12 = h.match(thing1['ra'],thing1['dec'],thing2['ra'],thing2['dec'],radius,maxmatch=0)

    keep = ~np.in1d(thing1['balrog_index'],thing1['balrog_index'][m1])
    return keep


def hpHEALPixelToRaDec(pixel, nside=4096, nest=True):
    theta, phi = hp.pix2ang(nside, pixel, nest=nest)
    ra, dec = convertThetaPhiToRaDec(theta, phi)
    return ra, dec

def hpRaDecToHEALPixel(ra, dec, nside=  4096, nest= True):
    phi = ra * np.pi / 180.0
    theta = (90.0 - dec) * np.pi / 180.0
    hpInd = hp.ang2pix(nside, theta, phi, nest= nest)
    return hpInd

def convertThetaPhiToRaDec(theta, phi):
    ra = phi*180.0/np.pi
    dec = 90.0 - theta*180.0/np.pi
    return ra,dec

def convertRaDecToThetaPhi(ra, dec):
    theta = (90.0 - dec) * np.pi / 180.0
    phi =  ra * np.pi / 180.0
    return theta, phi

def buildBadRegionMap(sim, truth, nside=4096, nest = True, magThresh=1., HPIndices=None):
    '''
    Note that here, "truth" really means "truthMatched".
    '''
    npix = hp.nside2npix(nside)
    pixInd = np.arange(npix)
    simInd = hpRaDecToHEALPixel( sim['ra'],sim['dec'], nside=nside, nest= nest)
    magErr = np.abs(truth['mag'] - sim['mag'])
    badObj = magErr > magThresh
    binct_bad = np.bincount(simInd[badObj],minlength=npix)
    binct_tot = np.bincount(simInd, minlength = npix)
    regionMap = binct_bad * 1. / binct_tot
    regionMap[binct_tot == 0] = hp.UNSEEN
    return regionMap

def visualizeHealPixMap(theMap, nest=True, title="map"):
    
    from matplotlib.collections import PolyCollection

    nside = hp.npix2nside(theMap.size)
    mapValue = theMap[theMap != hp.UNSEEN]
    indices = np.arange(theMap.size)
    seenInds = indices[theMap != hp.UNSEEN]

    print "Building polygons from HEALPixel map."
    vertices = np.zeros( (seenInds.size, 4, 2) )
    print "Building polygons for "+str(seenInds.size)+" HEALPixels."
    for HPixel,i in zip(seenInds,xrange(seenInds.size)):
        corners = hp.vec2ang( np.transpose(hp.boundaries(nside,HPixel,nest=True) ) )
        # HEALPix insists on using theta/phi; we in astronomy like to use ra/dec.
        vertices[i,:,0] = corners[1] * np.pi / 180.0
        vertices[i,:,1] = 90.0 - corners[0]*180.0/np.pi

    fig, ax = plt.subplots(figsize=(12,12))
    coll = PolyCollection(vertices, array = mapValue, cmap = plt.cm.gray, edgecolors='none')
    ax.add_collection(coll)
    ax.set_title(title)
    ax.autoscale_view()
    fig.colorbar(coll,ax=ax)
    print "Writing to file: "+title+".png"
    fig.savefig(title+".png",format="png")

    

def makeTheMap(des=None, truth=None, truthMatched=None, sim=None, tileinfo = None,maglimits = [22.5, 24.5],band='i'):
    # Get the unique tile list.
    from matplotlib.collections import PolyCollection
    tiles = np.unique(truth['tilename'])
    reconBins = np.array([15.0, maglimits[0], maglimits[1],99])
    theMap = np.zeros(tiles.size) - 999
    mapErr = np.zeros(tiles.size)
    vertices = np.zeros((len(tiles), 4, 2))
    for i,tile in zip(xrange(len(tiles)),tiles):
        # find all galaxies in this tile.
        thisDES = des[des['tilename'] == tile]
        theseBalrog = truthMatched['tilename'] == tile
        thisTruthMatched = truthMatched[theseBalrog]
        thisSim = sim[theseBalrog]
        thisTruth = truth[truth['tilename'] == tile]
        # reconstruct the total number in the desired interval
        this_N_est, _, _, _, errors = doInference(thisTruth, thisTruthMatched, thisSim, thisDES,
                                                  truth_bins = reconBins,tag='mag',doplot=False)
        norm = np.sum( ( truth['mag'] > np.min(maglimits) ) & (truth['mag'] < np.max(maglimits) ))
        theMap[i] = this_N_est[1] * 1. / norm
        mapErr[i] = errors[1] * 1. / norm
        thisInfo = tileinfo[np.core.defchararray.equal(tileinfo['tilename'], tile)]
        ra_ll, ra_lr, ra_ur, ra_ul = thisInfo['urall'][0], thisInfo['uraur'][0], thisInfo['uraur'][0], thisInfo['urall'][0]
        dec_ll, dec_lr, dec_ur, dec_ul = thisInfo['udecll'][0], thisInfo['udecll'][0], thisInfo['udecur'][0], thisInfo['udecur'][0]
        vertices[i,:,0] = np.array((ra_ll,  ra_lr,  ra_ur,  ra_ul))
        vertices[i,:,1] = np.array((dec_ll, dec_lr, dec_ur, dec_ul))
        
    # Normalize the map to relative fluctuations.
    normedMap = theMap / np.median(theMap) - 1
    good = theMap > 0.01
    bad = ~good
    
    fig, ax = plt.subplots()
    coll = PolyCollection(vertices[good,:,:], array=normedMap[good], cmap = plt.cm.gray, edgecolors='none')
    badcoll = PolyCollection(vertices[bad,:,:],facecolors='red',edgecolors='none')
    ax.add_collection(coll)
    ax.add_collection(badcoll)
    ax.autoscale_view()
    ax.set_xlabel('ra')
    ax.set_ylabel('dec')
    ax.set_title('number density fluctuations, in range: ['+str(maglimits[0])+'< '+band+' <'+str(maglimits[1])+']')
    fig.colorbar(coll,ax=ax)
    fig.savefig("normalized_number_map")

    
    errMap = (theMap - np.median(theMap) ) / mapErr 
    fig, ax = plt.subplots()
    coll = PolyCollection(vertices[good,:,:], array=errMap[good], cmap = plt.cm.gray, edgecolors='none')
    badcoll = PolyCollection(vertices[bad,:,:],facecolors='red',edgecolors='none')
    ax.add_collection(coll)
    ax.add_collection(badcoll)
    ax.autoscale_view()
    ax.set_xlabel('ra')
    ax.set_ylabel('dec')
    ax.set_title('chi fluctuations, in range: ['+str(maglimits[0])+'< '+band+' <'+str(maglimits[1])+']')
    fig.colorbar(coll,ax=ax)
    fig.savefig("error_map")

    stop


def getGoodRegionIndices(catalog=None, badHPInds=None, nside=4096):
    hpInd = hpRaDecToHEALPixel(catalog['ra'], catalog['dec'], nside=nside, nest= True)
    keep = ~np.in1d(hpInd, badHPInds)
    return keep
    
    
def getPostageStamps(catalog, badHPindices, nside=4096,band=None):
    print "Writing postage stamp catalogs."
    # Write the catalog of bad things to file.
    # First, assign HEALPixels to each catalog object:
    hpInd = hpRaDecToHEALPixel(catalog['ra'], catalog['dec'], nside=nside,nest= True)
    badObj = catalog[np.in1d(hpInd, badHPindices)]
    badHPInds = hpRaDecToHEALPixel(badObj['ra'], badObj['dec'], nside=nside, nest= True)
    
    # Organize 'objects' by tile.
    uInds = np.unique(badHPInds)
    tiles = np.zeros(uInds.size, dtype = [('tilename', 'S12'),('xmin_image', '>i8'),('ymin_image', '>i8'),('xmax_image', '>i8'),
                                          ('ymax_image', '>i8'),('HEALPixel', '>f8')])
    tiles['HEALPixel'] = uInds

    for index, i in zip(uInds,xrange(uInds.size)):
        if (i % 10) == 0:
            print i,uInds.size
        theseBadObjs = badObj[badHPInds == index]
        tiles[i]['xmin_image'] = np.min(theseBadObjs['xmin_image'])
        tiles[i]['xmax_image'] = np.max(theseBadObjs['xmax_image'])
        tiles[i]['ymin_image'] = np.min(theseBadObjs['ymin_image'])
        tiles[i]['ymax_image'] = np.max(theseBadObjs['ymax_image'])
        tiles[i]['tilename'] = theseBadObjs['tilename'][0]
        
    esutil.io.write('badRegionCutoutCoords-'+band+'.fits',tiles,clobber=True)

    # Now write a file with the bad objects.
    badObjArray = rf.append_fields( badObj,'HEALPixel', data=badHPInds)
    esutil.io.write('badObjectCoords-'+band+'.fits',badObjArray,clobber=True)
    
    return tiles

def main(argv):

    parser = argparse.ArgumentParser(description = 'Perform magnitude distribution inference on DES data.')
    parser.add_argument('filter',help='filter name',choices=['g','r','i','z','Y'])
    parser.add_argument("-r","--reload",help='reload catalogs from DESDB', action="store_true")
    args = parser.parse_args(argv[1:])
    band = args.filter
    print band
    # Get catalogs.
    print "performing inference in band: "+args.filter
    print "Reloading from DESDM:", args.reload
    des, sim, truthMatched, truth, tileInfo = getCatalogs(reload = args.reload, band = args.filter)
    print sim.size
    # Do the inference for things that aren't badly blended to start with.
    # Yes, this is cheating. Yes, we'll fix it later.
    # --------------------------------------------------
    pure_inds =  ( removeNeighbors(sim, des) & ( truthMatched['mag']>0 ) &
                   (np.sqrt( (truthMatched['ra'] - sim['ra'])**2 + (truthMatched['dec']-sim['dec'])**2 )*3600 < 0.1) )
    #pure_inds = ( truthMatched['mag'] > 0. ) & ( np.abs(truthMatched['mag'] - sim['mag']) < 2. )
    truthMatched = truthMatched[pure_inds]
    sim = sim[pure_inds]
    print sim.size
    # --------------------------------------------------
    
    eliMap = hp.read_map("sva1_gold_1.0.4_goodregions_04_equ_nest_4096.fits", nest=True)
    nside = hp.npix2nside(eliMap.size)

    # First, determine which HPixels have objects in them.
    FullRegionMap = buildBadRegionMap(sim, truthMatched, nside=nside )
    allIndices = np.arange(eliMap.size)
    useIndices = allIndices[FullRegionMap != hp.UNSEEN]
    eliMap[FullRegionMap == hp.UNSEEN] = hp.UNSEEN
    masked = (eliMap == 1)
    unmasked = (eliMap==0)
    eliMap[masked] = 0
    eliMap[unmasked] = 1.


    
    # Then build a bad Region Map
    regionMap = buildBadRegionMap(sim, truthMatched, nside=nside, HPIndices = useIndices, magThresh = 2.0, nest=True)
    regionMapSelected = regionMap*0. + hp.UNSEEN
    regionMapSelected[( regionMap >= 0.) | (eliMap == 0.0)] = 1.0
    visualizeHealPixMap(regionMap,title='badFractionMap-'+band)
    visualizeHealPixMap(eliMap,title='sva1_gold_1.0.4_goodregions_04')
    visualizeHealPixMap(regionMapSelected,title='BalrogMask-'+band+'_sva1v2')
    
    badIndices = allIndices[( regionMap > 0.1 ) | (eliMap == 0.0)]
    interestingIndices = allIndices[( regionMap > 0.1 ) & (eliMap == 1.0)]

    #getPostageStamps( sim, interestingIndices, nside=nside, band=band)
    
    obsKeepIndices = getGoodRegionIndices(catalog=sim, badHPInds=badIndices, nside=nside)
    truthKeepIndices = getGoodRegionIndices(catalog=truth, badHPInds=badIndices, nside=nside)
    desKeepIndices = getGoodRegionIndices(catalog=des, badHPInds=badIndices, nside=nside)

    sim = sim[obsKeepIndices]
    truthMatched = truthMatched[obsKeepIndices]
    truth = truth[truthKeepIndices]
    des = des[desKeepIndices]
    print sim.size
    # Remove everything in a bad region.
    
    
    # Infer underlying magnitude distribution for whole catalog.
    print "Starting regularized inference procedure."

    N_real_est, truth_bins_centers, truth_bins, obs_bins, errors = doInference(truth, truthMatched, sim, des,
                                                                               lambda_reg = .01, tag='mag', doplot = True)
    N_obs,_ = np.histogram(des['mag'],bins=truth_bins)
    N_obs = N_obs
    N_real_est = N_real_est
    makeHistogramPlots(N_real_est, truth_bins_centers, errors,
                       des, sim, truth, truthMatched, 
                       bin_edges = truth_bins, tag='mag')
    
    # --------------------------------------------------
    y = makeTheMap(des=des, truth=truth, truthMatched = truthMatched, sim=sim, tileinfo = tileInfo,band=band )



    
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
