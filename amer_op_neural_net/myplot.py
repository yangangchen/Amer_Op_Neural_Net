'''
The plotting functions
'''

from .config import *

figsize = (8, 6)
fontsize = 16
scaling = 100


def limit_2d(x, y, c, xlim=None, ylim=None):
    assert x.ndim == 1
    subx = x.copy()
    if y.ndim == 1:
        suby = y.copy()
    else:
        suby = y.copy().reshape((-1, y.shape[-1]))
    if c is not None and c.__class__.__name__ != "str":
        if c.ndim == 1:
            subc = c.copy()
        else:
            subc = c.copy().reshape((-1, c.shape[-1]))
    else:
        subc = c
    if xlim is not None:
        index = (subx >= min(xlim[0], xlim[1])) & (subx <= max(xlim[0], xlim[1]))
        subx = subx[index]
        suby = suby[index]
        if c is not None and c.__class__.__name__ != "str":
            subc = subc[index]
    if ylim is not None:
        index = (suby >= min(ylim[0], ylim[1])) & (suby <= max(ylim[0], ylim[1]))
        subx = subx[index]
        suby = suby[index]
        if c is not None and c.__class__.__name__ != "str":
            subc = subc[index]
    return subx, suby, subc


def line_2d(fig, x, y, c=None, xlim=None, ylim=None, axis_equal=False,
            title=None, xlabel=None, ylabel=None):
    assert x.ndim == 1
    assert c is None or c.__class__.__name__ == "str"
    subx, suby, subc = limit_2d(x, y, c, xlim, ylim)

    index = subx.argsort()
    # fig = plt.figure(figsize=figsize)
    if c is None:
        plt.plot(subx[index], suby[index])
    else:
        plt.plot(subx[index], suby[index], color=c)
    plt.clim(0, 1)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if axis_equal:
        plt.axis("equal")
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    return fig


def scatter_2d(fig, x, y, c, s=10, marker=".", xlim=None, ylim=None, axis_equal=False,
               title=None, xlabel=None, ylabel=None):
    assert x.ndim == 1
    subx, suby, subc = limit_2d(x, y, c, xlim, ylim)

    if c is None:
        plt.scatter(subx, suby, cmap=cm.bwr, s=s, marker=marker)
    else:
        plt.scatter(subx, suby, c=subc, cmap=cm.bwr, s=s, marker=marker)
        plt.clim(0, 1)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if axis_equal:
        plt.axis("equal")
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    return fig


def evaluate_scatter(X, Y, Yexact, YLabel, c, title, axeslabels, eff_d, Yscaling=False):
    if eff_d == 1:
        scaling2 = scaling if Yscaling else 1
        fig = plt.figure(figsize=figsize)
        if axeslabels is None:
            axeslabels = [None, None]
        scatter_2d(fig=fig, x=X[:, 0] * scaling, y=Y[:, 0] * scaling2, c=c[:, 0], s=10, marker=".", axis_equal=False,
                   title=title, xlabel=axeslabels[0], ylabel=axeslabels[1])
        if YLabel is not None:
            scatter_2d(fig=fig, x=X[:, 0] * scaling, y=YLabel[:, 0] * scaling2, c=c[:, 0],
                       s=2, marker=".", axis_equal=False)
        if Yexact is not None:
            line_2d(fig=fig, x=X[:, 0] * scaling, y=Yexact[:, 0] * scaling2, c='k', axis_equal=False)
        if savefig_mode:
            fig.savefig(directory + title.replace(" =", "=").replace("= ", "=").replace(" ", "_").replace(".", "p") \
                        + ".pdf", bbox_inches='tight')
        plt.close(fig)


def evaluate_exercise_boundary(X, control, control_exact, eff_d, title, n=None):
    if eff_d == 1:
        fig = plt.figure(figsize=figsize)
        if control_exact is not None:
            control_error = abs(control.astype(int) - control_exact.astype(int))
            control_color = 1 / (4 + 1) * (4 * control_exact - control + 1)
            xlabel = "underlying asset (s)" if d == 1 else "geometric average of underlying assets (s')"
            index = (control_error.ravel() == 0)
            t = np.tile(np.linspace(0, T, N + 1, dtype=np_floattype), (len(X), 1))
            if index.sum() > 0:
                scatter_2d(fig=fig, x=X.ravel()[index] * scaling, y=t.ravel()[index],
                           c=control_color.ravel()[index], s=1, marker=".", ylim=[T, 0],
                           xlabel=xlabel, ylabel="time (t)")
            if index.sum() < len(index):
                scatter_2d(fig=fig, x=X.ravel()[~index] * scaling, y=t.ravel()[~index],
                           c=control_color.ravel()[~index], s=5, marker="x", ylim=[T, 0])
        else:
            xlabel = "underlying asset (s)" if d == 1 else "geometric average of underlying assets (s')"
            t = np.tile(np.linspace(0, T, N + 1), (len(X), 1))
            scatter_2d(fig=fig, x=X.ravel() * scaling, y=t.ravel(),
                       c=control.ravel(), ylim=[T, 0],
                       xlabel=xlabel, ylabel="time (t)")
        plt.title("exercise boundary, " + title, fontsize=fontsize)
        if savefig_mode:
            fig.savefig(directory + "exercise_boundary_" + title.replace(" ", "_") + ".pdf", bbox_inches='tight')
        plt.close(fig)

