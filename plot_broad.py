import sys, os, glob

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

import prospect.io.read_results as bread
from prospect.utils import plotting

def process(filename, parnames=['sigma_smooth', 'zred']):
    res, pr, mod = bread.results_from(filename)
    pn, _, best, pcts = plotting.get_stats(res, parnames, start=0.66)
    return np.vstack([best, pcts.T])


def get_stardat(filename, pars=['name', 'miles_id', 'logt', 'logg', 'feh']):
    res, pr, mod = bread.results_from(filename)
    obs = res['obs']
    return {k: obs.get(k, None) for k in pars}

def plot_chains(filenames, outname='tes.pdf', check=False, start=0.5,
                showpars=['sigma_smooth', 'zred', 'spec_norm']):
    with PdfPages(outname) as pdf:
        for f in filenames:
            res, pr, mod = bread.results_from(f)
            obs = res['obs']
            obs.update({'filename': f})
            pfig = bread.param_evol(res, showpars=showpars)
            pfig.suptitle('{filename}={name}'.format(**obs))
            pdf.savefig(pfig)
            pl.close(pfig)

            if check:
                # Check for convergence
                raise(NotImplementedError)
            
def parse_filenames(filenames, **extras):
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
    resdir = 'results'
    version = ''
    outroot = ("{}/sigfit{}_star*_"
               "wlo=*_whi=*_mcmc").format(resdir, version)
    parnames=['sigma_smooth', 'zred']
    parlabel=['R (FWHM)', 'v (km/s)']

    files = glob.glob(outroot)
    #files = files[:40]
    
    starid, wmin, wmax = parse_filenames(files, outroot=outroot)
    results = np.array([process(f, parnames=parnames) for f in files])

    results[:,:,1] *= 2.998e5
    if 'R' in parlabel[0]:
        results[:,:,0] = 1/np.sqrt( (2.355/results[:,:,0])**2 + (1.0/1e4)**2)
    elif 'lambda' in parlabel[0]:
        results[:,:,0] = np.sqrt((2.355*results[:,:,0])**2 + ((wmin+wmax)/2.0 * 1e4 /1e4)**2)

    fig, axes = pl.subplots(2, 1)
    #sys.exit()
    for i, par in enumerate(parnames):
        ax = axes.flat[i]
        for j, sid in enumerate(np.unique(starid)):
            sel = starid == sid
            step(wmin[sel], wmax[sel], results[sel, 0, i],
                 ylo=results[sel, 1, i], yhi=results[sel, 3, i],
                 label='Star #{:.0f}'.format(sid),
                 ax=ax, linewidth=2, color=colors[ np.mod(j, 9)])
        ax.set_ylabel(parlabel[i])
        ax.set_xlabel('$\lambda (micron)$')

    fig.show()
    #step(results
