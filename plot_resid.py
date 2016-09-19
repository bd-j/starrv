import sys, glob, os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as pl

from plot_broad import process, get_stardat, parse_filenames, plot_one_spec


def get_dat(files):
    dat = [plot_one_spec(f, return_residuals=True) for f in np.array(files)]

    w = [d[0] for d in dat]
    chi = [d[2] / d[3] for d in dat]
    obs = [d[1] for d in dat]
    model = [d[1] - d[2] for d in dat]
      
    return w, obs, model, chi


def plot_chi_line(linewave, w, chi, logt, logg, feh):
    chilabel = '$\chi$ (obs-mod) @ {:4.1f}'.format(linewave)
    parnames = ['logT', '[Fe/H]', 'logg']
    fig, axes = pl.subplots(3, 2)

    ind = np.argmin(np.abs(w - linewave)) + 1
    chiline = chi[:, ind]

    for i, par in enumerate([logt, feh, logg]):
        ax = axes.flat[i]
        ax.plot(par, chiline, 'o')
        ax.set_ylabel(chilabel)
        ax.set_xlabel(parnames[i])

    ax = axes.flat[-1]
    ax.hist(chiline, bins=10, histtype='stepfilled', alpha=0.5)
    ax.set_xlabel(chilabel)
    return fig, axes


def delta_par(files):
    aps = ['logt', 'feh', 'logg']
    apresults = np.array([process(f, parnames=aps) for f in files])
    delta = np.zeros_like(apresults)
    for i, p in enumerate(aps):
        delta[:,:,i] = apresults[:,:,i] - np.array([d[p] for d in stardata])[:, None]


def stack_byparam(w, chi, param,  axes, return_vectors=False,
                  parname='Par', **kwargs):
    nsplit = len(axes.flat)
    op = np.argsort(param)
    npb = len(chi) / nsplit
    inds = range(0, len(chi), len(chi) / nsplit)
    inds[-1] = len(chi)-1
    print(inds, len(inds), len(chi))
    mus = []
    sigmas = []
    for i, ax in enumerate(axes.flat):
        sub = op[inds[i] : inds[i+1]]
        lims = param[sub].min(), param[sub].max()
        av = np.median(chi[sub, :], axis=0)
        sd = chi[sub, :].std(axis=0)
        ax.plot(w, av, label='mean')
        ax.plot(w, sd, label='dispersion')
        ax.set_title('{:3.2f}<{}<{:3.2f}'.format(lims[0], parname, lims[1]))
        mus.append(av)
        sigmas.append(sd)

    if return_vectors:
        return mus, sigmas
    else:
        return axes


def plot_delta():
    pass
    
if __name__ == "__main__":
    labeltxt = "#{starid:0.0f}:{name}"
    resdir = 'results_v3_odyssey'
    version = 'v3'
    show_res = False
    outroot = ("{}/siglamfit{}_star*_"
               "wlo=*_whi=*_mcmc.h5").format(resdir, version)
    parnames=['sigma_smooth', 'zred', 'spec_norm']
    parlabel=['R (FWHM)', 'v (km/s)', 'ln C']

    args = sys.argv
    if len(args) > 1:
        show_res = bool(args[1])
    
    files = glob.glob(outroot)
    #files = files[:40]

    # Get info for each segment/star
    results = np.array([process(f, parnames=parnames) for f in files])
    stardata = [get_stardat(f) for f in files]
    starid, wmin, wmax = parse_filenames(files, outroot=outroot)
    logt = np.array([d['logt'] for d in stardata])
    feh = np.array([d['feh'] for d in stardata])
    logg = np.array([d['logg'] for d in stardata])
    warm = (logt > 3.6) & (logt < np.log10(6300))
    
    # convert results units
    results[:,:,1] *= 2.998e5
    if 'lam' in outroot:
        parlabel[0] = "$\Delta\lambda$ (FWHM)"
        #results[:,:,0] *= 2.355
        results[:,:,0] = np.sqrt((2.355*results[:,:,0])**2 + (((wmin+wmax)/2.0 * 1e4 /1e4)**2)[:,None])
        results_R, parlabels_R = results.copy(), deepcopy(parlabel)
        results_R[:,:,0] = ((wmin+wmax)/2.0)[:, None] * 1e4 / results[:,:,0]
        parlabels_R[0] = r'R = $\bar{\lambda}/(\Delta\lambda)$ (FWHM)'
    else:
        results[:,:,0] = 1/np.sqrt( (2.355/results[:,:,0])**2 + (1.0/1e4)**2)

    wave, obs, model, chi =  get_dat(files)
    ratio = [o/m - 1 for o,m in zip(obs, model)]
        
    
    for  i, wlo in enumerate(np.unique(wmin)):
        sel = (wmin == wlo) & warm
        fig, axes = pl.subplots(3, 3)
        x = np.array(np.array(ratio)[sel].tolist())
        # x = np.array(np.array(chi)[sel].tolist())
        w = np.array(wave)[sel][0]
        for j, (par, parname) in enumerate(zip([logt, feh, logg], ['logt', 'feh', 'logg'])):
            stack_byparam(w, x, par[sel], axes[j, :], parname=parname)
        sys.exit()



