#!/usr/bin/env python

import numpy as np
from numpy import recarray
from scipy import linalg as slin

import sys
import esutil
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import numpy.lib.recfunctions as recfunctions


def fluxToMag(mag, zeropoint = 25.):
    return 10.**((zeropoint - mag)/2.5)

def magToFlux(flux, zeropoint = 25.):
    mag = zeropoint - 2.5*np.log10(flux)
    return mag

def addMagnitudes(mag1, mag2, zeropoint = 25.):
    flux1 = fluxToMag(mag1, zeropoint = zeropoint)
    flux2 = fluxToMag(mag2, zeropoint = zeropoint)
    newFlux = flux1 + flux2
    newMag = zeropoint - 2.5*np.log10(newFlux)
    return newMag
    

def generateTruthCatalog(n_obj = 10000, slope = 1.0, downsample = False):
    # Draw catalog entries from some simple distribution.
    # (A catalog should be a Numpy recarray, or similar, so we can get arrays indexed by keyword.)
    # (the catalog should have a 'data' and an 'index' field)
    mag = []
    if downsample is False:
        mag =  15 + 15*np.random.power(1+slope, size=n_obj)
    else:
        while len(mag) < n_obj:
            thismag =  15 + 15*np.random.power(1+slope)
            if np.random.rand() > np.exp((thismag - 26)/3.):
                mag.append(thismag)
        mag = np.array(mag)
    log_size =  (-0.287* mag + 4.98) + (0.0542* mag - 0.83 )* np.random.randn(n_obj)
    flux = fluxToMag(mag)
    size = np.exp(log_size) # Fortunately, this is basically in arcsec
    surfaceBrightness = flux / size / size / np.pi
    # A typical sky brightness is 22.5 mag / arcsec^2
    sky_sb =  np.repeat(3.* 10.**((25 - 22.5) / 2.5 ), n_obj)
    sky_flux = np.pi * size * size * sky_sb

    # The flux calibration will be total photon counts per unit of flux, integrated over the exposure time.
    # It's really just there to give us a noise model.
    calibration = np.repeat(100.,n_obj)
    error = np.sqrt( (sky_flux + flux) * calibration ) / calibration
    
    index = np.arange(int(n_obj))
    catalog = recarray((n_obj),dtype=[('data',mag.dtype),('error',error.dtype),('index',index.dtype),
                                      ('calibration',calibration.dtype),('size',size.dtype), ('SB',surfaceBrightness.dtype),
                                      ('sky_flux',sky_flux.dtype),('sky_SB',sky_sb.dtype),('flux',flux.dtype),
                                      ('blended',type(True))])

    catalog['data'] = mag
    catalog['flux'] = mag
    catalog['error'] = error
    catalog['index'] = index
    catalog['calibration'] = calibration
    catalog['size'] = size
    catalog['SB'] = surfaceBrightness
    catalog['sky_SB'] = sky_sb
    catalog['blended'] = False

    catalog = recfunctions.append_fields(catalog, 'data_truth', catalog['data'])
    return catalog
    

def blend(catalog, blend_fraction=0.1):
    # Choose two random subsets of the galaxies.
    # Add the flux of the first subset to that of the second, then remove the first.
    subset1 = np.random.choice(catalog,np.round(blend_fraction*catalog.size),replace=False)
    subset2 = np.random.choice(catalog,np.round(blend_fraction*catalog.size),replace=False)
    for entry1, entry2 in zip(subset1, subset2):
        newMag = addMagnitudes(entry1['data'], entry2['data'])
        ii = (catalog['index'] == entry1['index'])
        catalog[ii]['data'] = newMag
        catalog[ii]['flux'] = magToFlux(newMag)
        catalog[ii]['blended'] = True
        catalog[ii]['size'] = np.max( (entry1['size'], entry2['size']) )
        catalog[ii]['error'] = np.min( (entry1['error'], entry2['error']) )
        catalog[ii]['SB'] = catalog['flux'][ii] / ( catalog['size'][ii] **2) / np.pi
    keep = np.in1d(catalog['index'], subset2['index'], assume_unique=True, invert=True)
    if np.sum(keep) == 0:
        stop
    catalog = catalog[keep]
    return catalog
        

def applyTransferFunction(catalog,SN_cut = 5., mag_cut = 26., cbias = 0.0, mbias = 0.0, blend_fraction = 0.10):
    # This will add noise to the catalog entries, and apply cuts to remove some unobservable ones.
    # The noise is not trivial.
    obs_catalog = catalog.copy()

    # Blend. This happens before anything else.
    if blend_fraction > 0.0:
        obs_catalog = blend(obs_catalog, blend_fraction = blend_fraction)
    
    # Generate a noise vector based on the errors.
    noise = obs_catalog['error']*np.random.randn(len(obs_catalog))
    newFlux = 10.**((25. - obs_catalog['data'])/2.5) + noise
    newMag = 25. - 2.5*np.log10(newFlux)
    obs_catalog['data'] = newMag
    # Now recalculate the surface brightness.
    SB_new = obs_catalog['SB'] + noise / (obs_catalog['size'])**2 / np.pi
    
    # Apply a selection based on the new, noisy surface brightness.
    obs_catalog = obs_catalog[(SB_new >  obs_catalog['sky_SB']) & (obs_catalog['size'] > 0.05) & (newFlux > 0)]

    return obs_catalog
    


def doInference(sim_truth_catalog, sim_obs_catalog, real_obs_catalog,lambda_reg = 1.0):
    
    # --------------------------------------------------
    # First, settle on a binning scheme for the data.
    # Use the Freedman-Diaconis rule, which suggests:
    #    dx = 2 * IQR(x)/n^(1/3)
    #    IQR is the interQuartile range, and n is the number of data points.
    obs_binsize =0.5#2* (np.percentile(sim_obs_catalog['data'],75) - np.percentile(sim_obs_catalog['data'],25))/(len(sim_obs_catalog['data']))**(1/3.)
    obs_nbins = int(np.ceil( (np.max( sim_obs_catalog['data']) - np.min( sim_obs_catalog['data'] ) ) / obs_binsize) + 1)
    obs_bins = np.concatenate( (np.array([np.min( sim_obs_catalog['data'] )])-0.001*obs_binsize,
                                np.array([np.min( sim_obs_catalog['data'] )])-0.001*obs_binsize + obs_binsize*np.arange(obs_nbins)) )
    # --------------------------------------------------
    # Next, create the data histogram arrays.
    N_sim_obs, _   = np.histogram(sim_obs_catalog['data'], bins = obs_bins)
    N_real_obs, _  = np.histogram(real_obs_catalog['data'], bins = obs_bins)
    #--------------------------------------------------
    # To infer the response matrix, first match the sim_obs and sim_truth arrays.
    sorted_sim_truth_catalog = sim_truth_catalog[sim_obs_catalog['index']]
    # Important: we can only reconstruct the input histogram in bins from which objects are actually detected.
    truth_binsize = 0.5#2*(np.percentile(sorted_sim_truth_catalog['data'],75) - np.percentile(sorted_sim_truth_catalog['data'],25))/(len(sorted_sim_truth_catalog['data']))**(1/3.)
    truth_nbins = int(np.ceil( (np.max(sorted_sim_truth_catalog['data']) - np.min(sorted_sim_truth_catalog['data'] ) ) / truth_binsize)+1)
    truth_bins = np.concatenate( (np.array([np.min( sorted_sim_truth_catalog['data'] )])-0.001*truth_binsize,
                                  np.array([np.min( sorted_sim_truth_catalog['data'] )])-0.001*truth_binsize + truth_binsize*np.arange(truth_nbins)) )    
    N_sim_truth, _ = np.histogram(sim_truth_catalog['data'], bins = truth_bins)
    N_sim_truth_sorted, _  = np.histogram(sorted_sim_truth_catalog['data'], bins=truth_bins)
    # Now the entries in sorted_sim_truth should correspond
    # element-by-element to the entries in sim_obs_catalog, so we can
    # construct the arrays necessary to build the sum over the index
    # function.
    obs_bin_index = np.digitize(sim_obs_catalog['data'],obs_bins)-1
    truth_bin_index = np.digitize(sorted_sim_truth_catalog['data'],truth_bins)-1
    indicator = np.zeros( (truth_nbins, obs_nbins) )
    A = np.zeros( (obs_nbins, truth_nbins) )

    # Finally, compute the response matrix.
    for i in range(obs_nbins):
        for j in range(truth_nbins):
            these = np.where( ( obs_bin_index == i) & (truth_bin_index == j) )
            # This gives us the likelihood function.
            if ( N_sim_truth_sorted[j] > 0. ) & ( len(these[0]) > 1) :
                A[i,j] = float(len(these[0])) / N_sim_truth_sorted[j]

    # For this next part, we will perform a Tikhonov-regularized
    # inversion of the likelihood function, using the input Balrog
    # simulation as a prior.
    # To do this, we need a covariance matrix for the
    # data. We'll assume that the input distribution bin values have
    # only Poisson scatter, and construct the data covariance matrix
    # using the measured Balrog likelihood.
    truth_bins_centers = (truth_bins[0:-1] + truth_bins[1:])/2.
    lambda_reg = 1e-2
    lambda_reg_cov = 1e-6
    Cinv_data = np.diag(1./(N_real_obs+ lambda_reg_cov))
    #Ainv_reg = np.dot( (np.dot( np.transpose(A), A) + np.diag( np.repeat(lambda_reg**2 , truth_nbins) ) ), np.transpose( A ) )
    Ainv_reg = np.dot( np.linalg.inv(np.dot( np.dot( np.transpose(A), Cinv_data ), A) + np.diag( np.repeat(lambda_reg**2 , truth_nbins) ) ), np.dot( np.transpose( A ), Cinv_data) )
    #Ainv_reg = np.dot( np.linalg.inv(np.dot( np.dot( np.transpose(A), Cinv_data ), A) + lambda_reg**2 * np.diag(1./N_sim_truth) ), np.dot( np.transpose( A ), Cinv_data) )
    window = np.dot( Ainv_reg, N_sim_obs) / (N_sim_truth + lambda_reg_cov)
    #window = N_sim_truth_sorted*1.0/ (N_sim_truth + lambda_reg_cov)
    N_real_truth_nocorr = np.dot( Ainv_reg, N_real_obs)
    N_real_truth = N_real_truth_nocorr / window
    N_real_truth[~np.isfinite(N_real_truth)] = 0.
    
    Covar_orig = np.diag(N_real_truth)
    Amod = np.dot(Ainv_reg, A)
    Covar= np.dot( np.dot(Amod, Covar_orig), np.transpose(Amod) )

    leakage = np.dot( (Amod - np.diag(np.diag(Amod))) , N_real_truth)
    Covar_orig = Covar_orig
    return N_real_truth, truth_bins_centers, truth_bins, Covar_orig,leakage

def main(argv):
    # Generate a simulated simulated truth catalog.
    catalog_sim_truth = generateTruthCatalog(n_obj  = 10000, slope = 2.500, downsample = True)

    # Apply some complicated transfer function, get back a simulated
    # simulated observed catalog.
    catalog_sim_obs = applyTransferFunction(catalog_sim_truth)

    # Generate a simulated `real' truth catalog.
    catalog_real_truth = generateTruthCatalog(n_obj = 10000, slope = 2.50, downsample = False)
    
    # Generate a simulated `real' observed catalog.
    catalog_real_obs = applyTransferFunction(catalog_real_truth)

    # Perform inference to recover the `real' truth catalog.
    hist_est, bin_centers, bin_edges, Covar, leakage = doInference(catalog_sim_truth, catalog_sim_obs, catalog_real_obs)
    errs = np.sqrt(np.diag(Covar)) + leakage
    # Make a histogram of the 'real' input values, and compare with the inferred ones.
    hist_real, _ = np.histogram(catalog_real_truth['data'], bins = bin_edges)
    hist_sim, _ = np.histogram(catalog_sim_truth['data'], bins = bin_edges)
    hist_sim_obs, _  = np.histogram(catalog_sim_obs['data'], bins = bin_edges)
    hist_obs, obs_bin_edges  = np.histogram(catalog_real_obs['data'],bins = bin_edges)
    obs_bin_centers = (obs_bin_edges[0:-1] + obs_bin_edges[1:])/2.
    hist_obs_renorm = hist_obs.copy() * np.sum(hist_est)/np.sum(hist_obs)
    print "Number of input objects: ",len(catalog_real_truth)
    print "Number of objects detected: ",len(catalog_real_obs)
    print "Number of objects recovered: ", np.sum(hist_est)
    
    # Plots!
    fig = plt.figure(1, figsize=(26,7))
    ax = fig.add_subplot(1,2,1)
    ax.plot(bin_centers, hist_est,'.', c='blue', label='inferred')
    ax.errorbar(bin_centers, hist_est,errs, c='blue',linestyle="None")
    ax.plot(bin_centers, hist_real, c='red', label='true')
    ax.plot(obs_bin_centers, hist_obs, c='black',label='observed')
    ax.plot(bin_centers, hist_sim,c='green', label='simulated')
    ax.plot(bin_centers, hist_sim_obs, c = 'orange', label = 'sim. observed')
    ax.legend(loc='best')
    #ax.set_ylim([0,np.max(hist_real)*1.25])
    ax.set_ylabel('Number')
    ax.set_xlabel('magnitude')
    #ax.set_yscale('log')

    #ax = fig.add_subplot(1,3,2)
    #ax.errorbar(bin_centers,(hist_est-hist_real), errs/hist_real,c='blue',label='inferred - true',linestyle="None")
    #ax.plot(bin_centers, (hist_est - hist_real),'.',c='blue')
    #ax.plot(bin_centers,(hist_obs - hist_real*1.0)/hist_real, c='black', label = "obs - true")
    #ax.axhline(0.0)
    #ax.set_xlabel('magnitude')
    #ax.set_ylabel('residual')
    #ax.set_ylim([-0.25,.25])
    #ax.legend(loc = "best")
    
    
    ax = fig.add_subplot(1,2,2)
    ax.axhspan(-1,1,facecolor='red',alpha=0.2)
    ax.errorbar(bin_centers, (hist_est-hist_real), errs*0.+1.,c='blue')
    ax.plot(bin_centers, (hist_sim - hist_real),c='black',label = "sim'd - true")
#    ax.plot(bin_centers, (hist_obs_renorm - hist_real)/hist_real, c='red', label = "obs - true")
    ax.set_xlabel('Number')
    ax.set_ylabel('normalized reconstruction residuals')
    #ax.set_ylim([-10,10])
    ax.set_xlim([15,np.max(bin_edges)])
    ax.legend(loc='best')
    plt.show(block=True)

if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
