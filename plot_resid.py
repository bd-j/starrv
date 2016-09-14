import numpy as np
import matplotlib.pyplot as pl

from plot_broad import plot_one_spec


def get_dat(files):
    dat = np.array([plot_one_spec(f, return_residuals=True) for f in np.array(files)])

    w = dat[0, 0, :]
    chi = dat[:, 2, :] / dat[:, 3, :]
    obs = dat[:, 1, :]
    model = obs - dat[:, 2, :]
      
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

