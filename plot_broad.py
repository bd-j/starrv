import sys, os, glob

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

import prospect.io.read_results as bread
from prospect.utils import plotting


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

def plot_one_spec(filename, sps=None):
    res, pr, mod = bread.results_from(filename)
    obs = res['obs']
    obs.update({'filename': os.path.basename(filename)})
    w, s, u, m = obs['wavelength'], obs['spectrum'], obs['unc'], obs['mask']
    pn, _, best, pcts = plotting.get_stats(res, mod.theta_labels(), start=0.66)
    best_spec, _, _ = mod.mean_model(best, sps=sps, obs=obs)
    cal = mod._speccal
    fig, axes = pl.subplots(3, 1, sharex=True)
    axes[0].plot(w[m], s[m], label='observed')
    axes[0].plot(w[m], best_spec[m], label='model bestfit')
    axes[1].plot(w[m], cal[m], label='bestfit polynomial')
    axes[2].plot(w[m], (s[m] - best_spec[m] ) /u[m], label='$\chi (obs-mod)$')
    [ax.legend(loc=0, prop={'size': 8}) for ax in axes]
    fig.suptitle(tistring.format(**obs))
    return fig, axes

    
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

        
if __name__ == "__main__":
    colors = pl.rcParams['axes.color_cycle']
    labeltxt = "#{starid:0.0f}:{name}"
    resdir = 'results'
    version = 'v1'
    outroot = ("{}/siglamfit{}_star*_"
               "wlo=*_whi=*_mcmc").format(resdir, version)
    parnames=['sigma_smooth', 'zred']
    parlabel=['R (FWHM)', 'v (km/s)']

    files = glob.glob(outroot)
    #files = files[:40]

    stardata = [get_stardat(f) for f in files]
    starid, wmin, wmax = parse_filenames(files, outroot=outroot)
    results = np.array([process(f, parnames=parnames) for f in files])

    results[:,:,1] *= 2.998e5
    if 'lam' in outroot:
        parlabel[0] = "$\Delta\lambda$ (FWHM)"
        #results[:,:,0] *= 2.355
        results[:,:,0] = np.sqrt((2.355*results[:,:,0])**2 + (((wmin+wmax)/2.0 * 1e4 /1e4)**2)[:,None])
    else:
        results[:,:,0] = 1/np.sqrt( (2.355/results[:,:,0])**2 + (1.0/1e4)**2)

    fig, axes = pl.subplots(2, 1)
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
        ax.set_ylabel(parlabel[i])

    axes[1].set_xlabel('$\lambda (micron)$')
    axes[1].axhline(0.0, linestyle=':', linewidth=2.0, color='k')
    axes[0].axhline(2.54, linestyle=':', linewidth=2.0, color='k', label='Beifiore')
    [ax.legend(loc=0, prop={'size':8}) for ax in axes.flat]
    fig.show()
    #step(results
