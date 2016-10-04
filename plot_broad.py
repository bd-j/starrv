import sys, os, glob
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
colors = pl.rcParams['axes.color_cycle']

import prospect.io.read_results as bread
from prospect.utils import plotting
from prospect.sources import BigStarBasis


tistring = '{filename}={name}, (logt, feh, logg)=({logt:3.2f}, {feh:3.2f}, {logg:3.1f}'


def process(filename, parnames=['sigma_smooth', 'zred']):
    """Get stats from the chains for every results file.
    """
    res, pr, mod = bread.results_from(filename)
    pn, _, best, pcts = plotting.get_stats(res, parnames, start=0.66)
    return np.vstack([best, pcts.T])


def get_stardat(filename, pars=['name', 'miles_id', 'logt', 'logg', 'feh']):
    res, pr, mod = bread.results_from(filename)
    obs = res['obs']
    return {k: obs.get(k, None) for k in pars}


def plot_chains(filenames, outname='test.pdf', check=False, start=0.5,
                showpars=['sigma_smooth', 'zred', 'spec_norm']):
    with PdfPages(outname) as pdf:
        for f in filenames:
            res, pr, mod = bread.results_from(f)
            obs = res['obs']
            obs.update({'filename': f})
            pfig = bread.param_evol(res, showpars=showpars)
            pfig.suptitle(tistring.format(**obs))
            pdf.savefig(pfig)
            pl.close(pfig)

            if check:
                # Check for convergence
                raise(NotImplementedError)

def plot_one_spec(filename, sps=None, return_residuals=False):
    res, pr, mod = bread.results_from(filename)
    obs = res['obs']
    if sps is None:
        sps = BigStarBasis(use_params=['logt', 'logg', 'feh'], log_interp=True,
                           n_neighbors=1, **res['run_params'])
    obs.update({'filename': os.path.basename(filename)})
    w, s, u, m = obs['wavelength'], obs['spectrum'], obs['unc'], obs['mask']
    pn, _, best, pcts = plotting.get_stats(res, mod.theta_labels(), start=0.66)
    best_spec, _, _ = mod.mean_model(best, sps=sps, obs=obs)
    cal = mod._speccal
    delta = s - best_spec
    chi = delta / u
    if return_residuals:
        return w[m], s[m], delta[m], u[m]
    fig, axes = pl.subplots(3, 1, sharex=True)
    axes[0].plot(w[m], s[m], label='observed')
    axes[0].plot(w[m], best_spec[m], label='model bestfit')
    axes[1].plot(w[m], cal[m], label='bestfit polynomial')
    axes[2].plot(w[m], chi[m], label='$\chi (obs-mod)$')
    [ax.legend(loc=0, prop={'size': 8}) for ax in axes]
    fig.suptitle(tistring.format(**obs))
    return fig, axes


def residual_stack(files, ax=None, **kwargs):
    chi = 0
    for f in files:
        w, s, d, u = plot_one_spec(f, return_residuals=True)
        chi += d / u

    chi /= len(files)
    if ax is not None:
        ax.plot(w, chi, **kwargs)
        return ax, w, chi
    else:
        return w, chi

    
def parse_filenames(filenames, **extras):
    """Pull out info from the filenames
    """
    linfo = [float(l.replace('star','').replace('wlo=','').replace('whi=',''))
             for f in filenames
             for l in os.path.basename(f).split('_')[1:4]
             ]
    star = linfo[::3]
    wmin = linfo[1::3]
    wmax = linfo[2::3]
    return np.array(star), np.array(wmin), np.array(wmax)


def step(xlo, xhi, y, ylo=None, yhi=None, ax=None,
         label=None, **kwargs):
    """A custom method for plotting step functions as a set of horizontal lines
    """
    clabel = label
    for i, (l, h, v) in enumerate(zip(xlo, xhi, y)):
        ax.plot([l,h],[v,v], label=clabel, **kwargs)
        if ylo is not None:
            ax.fill_between([l,h], [ylo[i], ylo[i]], [yhi[i], yhi[i]],
                            alpha=0.5, **kwargs)
        clabel = None


def plot_orders():
    orders = {'9': (0.69, 0.85),
              '8': (0.70, 0.95),
              '7': (0.8, 1.1),
              '6': (0.95, 1.25),
              '5': (1.1, 1.5),
              '4': (1.4, 1.9),
              '3': (1.95, 2.5),
              }
    fig, ax = pl.subplots()
    for o, (lo, hi) in orders.items():
        ax.plot([lo, hi], np.zeros(2) + float(o), '-o')
    return fig, ax, orders


def plot_blocks(results, stardata, starid, wmin, wmax, parnames=None):
    fig, axes = pl.subplots(results.shape[-1], 1)
    #sys.exit()
    for i, par in enumerate(parnames):
        ax = axes.flat[i]
        for j, sid in enumerate(np.unique(starid)):
            sel = starid == sid
            dat = np.array(stardata)[sel][0]
            dat['starid'] = sid
            step(wmin[sel], wmax[sel], results[sel, 0, i],
                 ylo=results[sel, 1, i], yhi=results[sel, 3, i],
                 label=labeltxt.format(**dat),
                 ax=ax, linewidth=2, color=colors[ np.mod(j, 9)])
    axes[-1].set_xlabel('$\lambda (micron)$')
    axes[1].axhline(0.0, linestyle=':', linewidth=2.0, color='k')
    #[ax.legend(loc=0, prop={'size':8}) for ax in axes.flat]
    return fig, axes


def summary_stats(results):
    summary = np.zeros([4, results.shape[-1]])
    values = np.squeeze(results[:,0,:])
    sigmas = np.squeeze(results[:,3,:] - results[:,1,:]) / 2
    summary[0, :] = np.average(values, weights=1.0 / sigmas**2, axis=0,)
    summary[1, :] = np.sqrt(1 / np.sum(1/sigmas**2, axis=0))
    summary[2, :] = np.nanmedian(values, axis=0)
    summary[3, :] = np.nanstd(values, axis=0)
    return summary


def plot_ensemble(results, wmin, wmax, simple=False, **extras):

    ind = 2 * int(simple)
    print(ind)
    fig, axes = pl.subplots(results.shape[-1], 1)
    if not np.iterable(axes):
        axes = np.array([axes])
    for j, wlo in enumerate(np.unique(wmin)):
        sel = wmin == wlo
        summary = summary_stats(results[sel])
        print(summary[ind:ind+1,0])
        for i, ax  in enumerate(axes.flat):
            step([wmin[sel][0]], [wmax[sel][0]], [summary[ind, i]],
                 ylo=[summary[ind, i] - summary[ind+1,i]],
                 yhi=[summary[ind, i] + summary[ind+1,i]],
                 ax=ax, linewidth=2, color=colors[0])
    axes[-1].set_xlabel('$\lambda (micron)$')
    return fig, axes


def write_results_h5(results, wmin, wmax, stardata, outfile='starrv_results.h5'):
    import h5py
    
    rparam = ['resolution', 'velocity', 'lnC']
    ns = len(results)
    
    pars = [('wmin', np.float64), ('wmax', np.float64), ('starid', np.float64), ('starname', 'S20')]
    params = np.zeros(ns, dtype=np.dtype(pars))
    params['wmin'] = wmin
    params['wmax'] = wmax
    params['starname'] = [np.str(s['name']) for s in stardata]
    
    cols = ['map', 'p16', 'p50', 'p84']
    pdt = np.dtype([(c, np.float64) for c in cols]) 

    with h5py.File(outfile, 'w') as f:
        par = f.create_dataset('segment_parameters', data=params)
        for i, p in enumerate(rparam):
            s = np.zeros(ns, dtype=pdt)
            for j, c in enumerate(cols):
                s[c] = results[:, j, i]
            dat = f.create_dataset(p, data=s)
        

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

    # Make summary plots
    nshow = 48 #len(results)
    bfig, baxes = plot_blocks(results[:nshow], stardata[:nshow], starid[:nshow],
                              wmin[:nshow], wmax[:nshow], parnames=parnames)
    if wmin.min() < 0.7:
        baxes[0].axhline(2.54, linestyle=':', linewidth=2.0, color='k', label='Beifiore')
    if show_res:
        baxes[0].axhline(2000, linestyle=':', linewidth=2.0, color='k', label='IRTF')
    [ax.set_ylabel(parlabel[i]) for i, ax in enumerate(baxes.flat)]
    
    # Ensemble figure
    simple = True
    efig, eaxes = plot_ensemble(results[warm], wmin[warm], wmax[warm], simple=simple)
    [ax.set_ylabel(parlabel[i]) for i, ax in enumerate(eaxes.flat)]
    [ax.axhline(0.0, linestyle=':', linewidth=2.0, color='k') for ax in eraxes[1:]]
    eaxes[0].axhline(2.54, linestyle=':', linewidth=2.0, color='k', label='Beifiore')
    [ax.set_xlim(0.39, 0.65) for ax in eaxes.flat]
    eaxes[0].set_ylim(2, 3.5)
    
    # Ensemble figure in R space
    simple = True
    erfig, eraxes = plot_ensemble(results_R[warm, :, None], wmin[warm], wmax[warm], simple=simple)
    eraxes[0].axhline(2000, linestyle=':', linewidth=2.0, color='k')
    [ax.axhline(0.0, linestyle=':', linewidth=2.0, color='k') for ax in eraxes[1:]]
    [ax.set_ylabel(p) for p, ax in zip(parlabels_R, eraxes.flat)]
    
    # Residual stack
    rfig, raxes = pl.subplots(6, 2)
    for  i, wlo in enumerate(np.unique(wmin)):
        _ = residual_stack(np.array(files)[wmin == wlo], ax=raxes.flat[i])

    # Calculate deltas for atmospheric parameters

    fit_starpars=False
    if fit_starpars:
        aps = ['logt', 'feh', 'logg']
        apresults = np.array([process(f, parnames=aps) for f in files])
        delta = np.zeros_like(apresults)
        for i, p in enumerate(aps):
            delta[:,:,i] = apresults[:,:,i] - np.array([d[p] for d in stardata])[:, None]
        
