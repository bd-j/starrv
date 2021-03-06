import numpy as np
import matplotlib.pyplot as pl
import h5py


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


def summary_stats(results):
    summary = np.zeros([4])
    values = np.squeeze(results['map'])
    sigmas = np.squeeze(results['p84'] - results['p16']) / 2
    # weighted average and SDOM, assuming gaussian approximation to posteriors
    summary[0] = np.average(values, weights=1.0 / sigmas**2, axis=0,)
    summary[1] = np.sqrt(1 / np.sum(1/sigmas**2, axis=0))
    # simple median and dispersion of MAP values
    summary[2] = np.nanmedian(values)
    summary[3] = np.nanstd(values)
    return summary


def plot_ensemble(data, wmin, wmax, simple=False, ax=None,
                  show=['resolution', 'velocity'], **extras):

    ind = 2 * int(simple)
    # loop over segments
    for j, wlo in enumerate(np.unique(wmin)):
        sel = wmin == wlo
        summary = summary_stats(data[sel])
        whi = wmax[sel][0]
        step([wlo], [whi], [summary[ind]],
             ylo=[summary[ind] - summary[ind+1]],
             yhi=[summary[ind] + summary[ind+1]],
             ax=ax, linewidth=2, **extras)
    
    return ax


if __name__ == "__main__":

    # switch to plot resolution in R instead of Delta(lambda)
    plotR = True
    # which parameters to show
    show = ['resolution', 'velocity']

    dfile = h5py.File('summary_v3.h5', 'r')
    params = dfile['segment_parameters'][:]
    wmin = params['wmin'][:]
    wmax = params['wmax'][:]

    try:
        warm = (params['logt'] > 3.6) & (params['logt'] < 3.8)
    except(ValueError):
        warm = slice(None)


    fig, axes = pl.subplots(len(show), 1)
    axes = np.atleast_1d(axes)
    for i, pname in enumerate(show):
        data = dfile[pname][:]
        if (pname == 'resolution') & plotR:
            # Convert from FWHM(lambda) to approximate R value
            for c in data.dtype.names:
                data[c] = (wmin+wmax)* 1e4 /2.0 / data[c]
        ax = axes.flat[i]
        ax = plot_ensemble(data[warm], wmin[warm], wmax[warm], simple=True,
                           ax=ax, color=pl.rcParams['axes.color_cycle'][0])
    axes[-1].set_xlabel('$\lambda (micron)$')
