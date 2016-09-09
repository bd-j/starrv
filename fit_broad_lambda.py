import sys, time

from itertools import product
from copy import deepcopy

import numpy as np
import h5py

from sedpy.observate import vac2air

from prospect.sources import BigStarBasis
from prospect.models import priors, SedModel
from prospect.likelihood import lnlike_spec, write_log
from prospect.fitting import run_emcee_sampler
from prospect.io import write_results as bwrite

lsun_cgs = 3.846e33
pc2cm = 3.085677581467192e18  # in cm

# The parameters of interest
# resolution based on MILES Befiore et al.
res = {'name': 'sigma_smooth', 'N': 1,
       'init': 2.54 / 2.35, 'init_disp': 1.0 / 2.35,
       'disp_floor': 0.5 / 2.35,
       'isfree': True,
       'prior_function': priors.tophat,
       'prior_args': {'mini':1.0 / 2.35, 'maxi': 4.0 / 2.35},
       'units': '$\sigma_\lambda (\AA)$'}
vel = {'name': 'zred', 'N': 1,
       'init': 0.0, 'init_disp': 1e-5,
       'disp_floor': 1e-5,
       'isfree': True,
       'prior_function': priors.tophat,
       'prior_args': {'mini':-1e-4, 'maxi':1e-4}}
#set up some nuisance parameters
norm = {'name': 'spec_norm', 'N': 1,
       'init': 0.1, 'init_disp': 0.1,
       #'disp_floor': 0.05,
       'isfree': True,
       'prior_function': priors.tophat,
       'prior_args': {'mini':-1, 'maxi': 1}}
poly = {'name': 'poly_coeffs', 'N': 1,
       'init': 0.0, 'init_disp': [1.0],
       'isfree': True,
       'prior_function': priors.tophat,
       'prior_args': {'mini':-10, 'maxi': 10.0}}

model_params = [res, vel, norm, poly]

# set up some fixed scalar parameters
fixed = [('logt', 3.74), ('logg', 4.4), ('feh', 0.0),
         ('logl', 0.0), ('lumdist', 1e-5),
         ('smoothtype', 'lambda'), ('fftsmooth', True)]
for p, v in fixed:
    pdict = {'name':p, 'N': 1, 'init': v, 'isfree': False, 'prior_function': priors.tophat}
    model_params.append(pdict)


def lnprobfn(theta, model=None, sps=None, obs=None):
    """Simple likelihood
    """
    lnp_prior = model.prior_product(theta)
    if np.isfinite(lnp_prior):
        spec, phot, _ = model.mean_model(theta, sps=sps, obs=obs)
        lnp_spec = lnlike_spec(spec, obs=obs)
        write_log(theta, lnp_prior, lnp_spec, 0.0, 0, 0)
        return lnp_spec + lnp_prior

    else:
        return -np.inf


def load_obs(starid=0, starlib='', wmin=0, wmax=np.inf,
             noise_dilation=1, sps=None, **extras):
    """Read MILES/IRTF library
    """
    #conversion from L_sun/Hz/Lbol to maggies at 10pc
    conv = np.log(lsun_cgs) - np.log(4 * np.pi) - 2 * np.log(10 * pc2cm)
    conv += np.log(1e23) - np.log(3631)
    conv = np.exp(conv)
    
    with h5py.File(starlib, 'r') as f:
        wave = f['wavelengths'][:]
        spec = f['spectra'][starid, :]
        unc = f['unc'][starid, :]
        params = f['parameters'][starid]
        anc = f['ancillary'][starid]
    
    obs = {'starid':starid}
    for n in anc.dtype.names:
        obs[n] = anc[n]
    for n in params.dtype.names:
        obs[n] = params[n]

    # rectify labels
    obs['logt'] = np.log10(obs['teff'])
    obs['luminosity'] = 10**obs['logl']
    if sps is not None:
        for n in sps.stellar_pars:
            mi, ma = sps._libparams[n].min(), sps._libparams[n].max()
            obs[n] = np.clip(obs[n], mi + 0.01*np.abs(mi), ma - 0.01*np.abs(ma))
    
    # Convert to maggies
    obs['spectrum'] = spec / obs['luminosity'] * conv
    obs['unc'] = unc / obs['luminosity'] * conv
    obs['unc'] *= noise_dilation
    # Fill dictionary
    obs['wavelength'] = vac2air(wave * 1e4)
    obs['logl'] = 0.0
    obs['filters'] = None
    obs['maggies'] = None
    obs['maggies_unc'] = None
    mask = (obs['wavelength'] > wmin*1e4) & (obs['wavelength'] < wmax*1e4)
    obs['mask'] = mask

    return obs


def load_model(stardat, npoly=5, wmin=0, wmax=np.inf,
               fit_starpars=False, sps=None, **extras):

    fwhm_irtf = 10.0
    fwhm_miles = 2.54

    pnames = [p['name'] for p in model_params]
    # setup stellar parameters
    for (p, d) in [('logt', 0.01), ('feh', 0.2), ('logg', 0.1)]:
        model_params[pnames.index(p)]['init'] = stardat[p]
        if fit_starpars:
            mil, mal = sps._libparams[p].min(), sps._libparams[p].max()
            mi = np.clip(stardat[p] - d, mil, mal)
            ma = np.clip(stardat[p] + d, mil, mal)
            model_params[pnames.index(p)]['isfree'] = True
            model_params[pnames.index(p)]['prior_args'] = {'mini': mi, 'maxi': ma}

    # Choose MILES or IRTF
    if wmin > 0.72:
        # For IRTF we increase the initial guess for the resolution to ~10AA
        ind = pnames.index('sigma_smooth')
        for q in ['init', 'init_disp', 'disp_floor']:
            model_params[ind][q] *= fwhm_irtf / fwhm_miles
        for q in ['mini', 'maxi']:
            model_params[ind]['prior_args'][q] *= fwhm_irtf / fwhm_miles

    # set up polynomial
    pid = pnames.index('poly_coeffs')
    model_params[pid]['init'] = np.zeros(npoly)
    model_params[pid]['N'] = npoly
    polymax = 1.0 / (np.arange(npoly) + 1)
    model_params[pid]['prior_args']['maxi'] = polymax
    model_params[pid]['prior_args']['mini'] = 0 - polymax
    model_params[pid]['init_disp'] = polymax / 2.0    
        
    return SedModel(model_params)


def run_segment(run_params):
    # Setup
    #run_params.update(kwargs)
    outname = run_params.get('outname', None)
    if outname is None:
        outname = ("broad_results/siglamfit{version}_star{starid}_"
                   "wlo={wmin:3.2f}_whi={wmax:3.2f}")
    outroot = outname.format(**run_params)
    # Load
    sps = BigStarBasis(use_params=['logt', 'logg', 'feh'], log_interp=True,
                       n_neighbors=1, **run_params)
    obs = load_obs(**run_params)
    #Test
    try:
        out = sps.get_star_spectrum(**obs)
    except(ValueError):
        print("Can't build star {starid} at  logt={logt}, logg={logg}, feh={feh}".format(**obs))
        return None
    model = load_model(obs, sps=sps, **run_params)
    # Fit
    tstart = time.time()    
    postkwargs = {'model': model, 'sps': sps, 'obs': obs}
    esampler, _, _ = run_emcee_sampler(lnprobfn, model.initial_theta, model,
                                       postkwargs=postkwargs, **run_params)
    tsample = time.time() - tstart
    print('took {:.1f}s'.format(tsample))
    # Write
    bwrite.write_pickles(run_params, model, obs, esampler, None,
                         tsample=tsample, outroot=outroot)


if __name__ == "__main__":

    run_params = {'libname':'data/ckc_R10K.h5',
                  'starlib': 'data/culled_libv2_w_mdwarfs_w_unc.h5',
                  'version': 'v1',
                  # object setup
                  'outname': "results/siglamfit{version}_star{starid}_wlo={wmin:3.2f}_whi={wmax:3.2f}",
                  'starid': 0,
                  'wmin': 1.0,
                  'wmax': 1.1,
                  # fit setup
                  'npoly': 4,
                  'fit_starpars': False,
                  # emcee params
                  'nburn': [16, 16, 32, 64, 128],
                  'niter': 512,
                  'nwalkers': 32,
                  'noise_dilation': 1.0
                  }

    #ntry = 100
    #stars = np.round(np.linspace(0, 190, ntry)).astype(int)
    stars = np.arange(25, 100).astype(int)
    
    optical_wedges = ([0.40, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64],) #MILES optical
    #wedges = ([0.8, 0.9, 1.0, 1.1, 1.2, 1.3], #J
    #          [1.5, 1.6, 1.7, 1.8],  #H
    #          [2.0, 2.2, 2.45]  #K
    #          )

    wedges = optical_wedges
    # super-obfuscated code to get a nregions X 2 array with wlo, whi
    wlims = np.hstack([np.array([wedge[:-1], wedge[1:]]) for wedge in wedges]).T

    pardictlist = []
    for star, (wlo, whi) in product(stars, wlims):
        print(star, wlo, whi)
        pdict = deepcopy(run_params)
        pdict.update({'starid': star, 'wmin': wlo, 'wmax': whi})
        pardictlist.append(pdict)

    from multiprocessing import Pool
    pool = Pool(6)
    M = pool.map
    M(run_segment, pardictlist)

