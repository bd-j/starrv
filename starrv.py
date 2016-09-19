import time
import numpy as np
import h5py

from prospect.likelihood import lnlike_spec, write_log
from sedpy.observate import vac2air


lsun_cgs = 3.846e33
pc2cm = 3.085677581467192e18  # in cm


def lnprobfn(theta, model=None, sps=None, obs=None, verbose=False):
    """Simple likelihood
    """
    lnp_prior = model.prior_product(theta)
    if np.isfinite(lnp_prior):
        try:
            ts = time.time()
            spec, phot, _ = model.mean_model(theta, sps=sps, obs=obs)
            dt = time.time() - ts
        except(ValueError):
            # couldn't build model
            return -np.inf
        lnp_spec = lnlike_spec(spec, obs=obs)
        if verbose:
            write_log(theta, lnp_prior, lnp_spec, 0.0, dt, 0)
        return lnp_spec + lnp_prior

    else:
        return -np.inf


def load_obs(starid=0, starlib='', wmin=0, wmax=np.inf,
             noise_dilation=1, sps=None, mask_width=6.0, **extras):
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

    # Build mask including s-process lines
    mask = (obs['wavelength'] > wmin*1e4) & (obs['wavelength'] < wmax*1e4)
    for l in [4077, 4213, 4554, 4934, 4862]:
        mask = mask & ((obs['wavelength'] > (l + mask_width)) |
                       (obs['wavelength'] < (l - mask_width)))
    obs['mask'] = mask

    return obs

