"""
General purpose plotting library
================================

.. module:: lib_plotting
   :platform: Linux
   :synopsis: module to easily-handle plotting functions

.. moduleauthor:: Jose M. Esnaola-Acebes <josemesnaola@gmail.com>

This library contains functions to format, modify and create figures, axes and plot, mainly using
:mod:`matplotlib` as basic plotting library.


Formatting methods
------------------

.. autosummary::
   :toctree: generated/

   format_ax             Formats the axes properties in a rather
                          flexible way.
   custom_plot_params    Sets general parameters of the
                          :mod:`matplotlib` module.
   apply_custom_style    Decorator: applies :func:`custom_plot_params`
   set_plot_dim           Sets the dimension of a figure.
   mod_axes              Removes axis of ax or axes.
   mod_ax_lims           Modify limits, labels and format of axes of a given ax.

Figure and axes builder methods
-------------------------------

.. autosummary::
   :toctree: generated/

   set_plot_environment  Creates a figure and axes if needed.
   get_fig_ax            Searches for figure associated to an ax, and
                          vice versa.
   setup_ax              Quickest way of checking and/or creating
                          figures and/or axes when necessary.
   create_colorbar_ax    Set ups and gives format to the ax that will
                               contain a color-bar.

Plotting methods
----------------

.. autosummary::
   :toctree: generated/

   plot_error_filled     Plots line and associated confidence interval
                          as a semi-transparent filled background.
   density_scatter       Plots a scatter plot with a density-based
                          color-map.
   fig_grid              Draws an auxiliary grid to aid at designing
                               figures and their axes locations.
   rectspan              Draws a shaded rectangle over multiple axes.

Auxiliary classes
-----------------

.. autosummary::
   :toctree: generated/

   MidpointNormalize     reates a custom normalization rule.

Auxiliary methods
-----------------

.. autosummary::
   :toctree: generated/

   store                 Stores an object in binary format.
   retrieve              Retrieves a binary stored object.
   hex_to_rgb            Converts an hexadecimal rgb color to tuple.


Implementation
--------------

.. todo::

   Give a brief description about the data-frames that are use in the different functions in this library.
"""

import pickle
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from matplotlib.patches import Rectangle, ConnectionPatch
from scipy.interpolate import interpn
import functools

import logging

logging.getLogger('lib_plotting').addHandler(logging.NullHandler())

__author__ = 'Jose M. Esnaola-Acebes'
__docformat__ = 'reStructuredText'

"""
Formatting methods
------------------ 
"""


def format_ax(ax, axis='both', twidth=1.0, bwidth=1.0, tlength=2.0, xformat=r'%0.1f', yformat=None, tickfontsize=None,
              labelfontsize=None, taxis='x', nbins=False, baxis='x', xticks=None, xticklabels=None, yticks=None,
              yticklabels=None, xpad=None, xlabel=False, ylabel=False, ypad=None, yrot=None):
    """Function that formats the axes according to some custom options. Function can be called as many times as
    wanted.

    :param plt.Axes ax: ax to be formatted.
    :param str axis: where ('x', 'y', 'both') to apply the formatting of axis. (default: 'both' axis).
    :param float twidth: width fo the ticks.
    :param float bwidth: width of the axis spine line
    :param float tlength: length of ticks
    :param str taxis: where ('x' or 'y') to change the ticklabels font size (if any).
    :param float tickfontsize: font size fo ticklabels.
    :param float labelfontsize: font size of labels.
    :param str baxis: where to set the number of bins (``x``, ``y`` or ``both``)
    :param int nbins: set the number of ticks in ``baxis``.
    :param str xformat: format of the ticklabels of the x axis
    :param list of float xticks: list of ticks of the x axis.
    :param list of str xticklabels: list of tick-labels of the x axis.
    :param str xlabel: set the y label.
    :param float xpad: padding of the label of the x axis.
    :param str yformat: format of ticklabels of y axis
    :param list of float yticks: list of ticks of the y axis.
    :param list of str yticklabels: list of tick-labels of the y axis.
    :param str ylabel: set the x label.
    :param float ypad: padding of the label of the y axis.
    :param float yrot: rotation of the y label.
    """

    ax.tick_params(axis=axis, which='major', direction='in', top=True, right=True, width=twidth, length=tlength)
    ax.xaxis.set_major_formatter(FormatStrFormatter(xformat))
    if yformat:
        ax.yaxis.set_major_formatter(FormatStrFormatter(yformat))
    for axis in ['top', 'bottom', 'right', 'left']:
        ax.spines[axis].set_linewidth(bwidth)

    # Ticks and ticklabels
    if xticks is not None:
        ax.set_xticks(xticks)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=tickfontsize)
    if tickfontsize and taxis == 'x':
        plt.setp(ax.get_xticklabels(), fontsize=tickfontsize)

    if nbins:
        ax.locator_params(axis=baxis, nbins=nbins)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=tickfontsize)
    if tickfontsize and taxis == 'y':
        plt.setp(ax.get_yticklabels(), fontsize=tickfontsize)

    # Labels
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=labelfontsize, labelpad=xpad)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=labelfontsize, labelpad=ypad, rotation=yrot)


def custom_plot_params(fontsizefactor=1.0, latex=False, dpi=300, **kwargs):
    """Function that sets the main properties of the figures and plots.

    :param float  fontsizefactor: Scalilng factor applied to fontsize. (default: 1.0)
    :param bool latex: Whether to use latex to render formulas, number etc. (default: False)
    :param int dpi: Dots (pixels) per inch. Matplotlib default is 100. (default: 200)
    :param kwargs: additional keyword arguments that modify :py:dict:`matplotlib.pyplot.rcParams`.
    """

    style = kwargs.pop('style', None)
    if style is not None:
        plt.style.use(style)
    fig_width = 2.1  # width in inches
    fig_height = 1.85  # height in inches
    fig_size = [fig_width, fig_height]
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = True

    plt.rcParams['lines.linewidth'] = 1.
    plt.rcParams['lines.markeredgewidth'] = 0.3
    plt.rcParams['lines.markersize'] = 2.5
    plt.rcParams['font.size'] = 10 * fontsizefactor
    plt.rcParams['legend.fontsize'] = 8 * fontsizefactor
    plt.rcParams['axes.facecolor'] = '1'
    plt.rcParams['axes.edgecolor'] = '0'
    plt.rcParams['axes.linewidth'] = '0.7'

    plt.rcParams['axes.labelcolor'] = '0'
    plt.rcParams['axes.labelsize'] = 9.5 * fontsizefactor
    plt.rcParams['axes.titlesize'] = 9.5 * fontsizefactor
    plt.rcParams['xtick.labelsize'] = 8 * fontsizefactor
    plt.rcParams['ytick.labelsize'] = 8 * fontsizefactor
    plt.rcParams['xtick.color'] = '0'
    plt.rcParams['ytick.color'] = '0'
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2

    #
    if latex:
        rc_fonts = {
            "text.usetex": False,
            # 'mathtext.default': 'regular',
            'text.latex.preamble': [r'\usepackage{amsmath,amssymb,bm,physics,lmodern}'],
            # "font.family": "serif",
            # "font.serif": "computer modern roman",
            "font.sans-serif": ["Arial", "Helvetica"],
        }
    else:
        rc_fonts = {
            "text.usetex": False,
        }
        plt.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams.update(rc_fonts)

    # Perform any other change in rcParams
    for key in kwargs:
        try:
            plt.rcParams[key] = kwargs[key]
        except KeyError:
            logging.error("Keyword '%s' is not valid for 'plt.rcParams', skipping." % key)


# Decorator of plots
def apply_custom_style(plot_func):
    @functools.wraps(plot_func)
    def wrapper_apply_custom_style(*args, **kwargs):
        custom_kw = kwargs.pop('custom_kw', dict())
        custom_plot_params(**custom_kw)
        return plot_func(*args, **kwargs)

    return wrapper_apply_custom_style


def set_plot_dim(x, y):
    """ Function that sets the figure dimensions.

    :param float x: width of figure in inches.
    :param float y: heigth of figure in inches.
    """
    fig_width = x
    fig_height = y
    fig_size = [fig_width, fig_height]
    plt.rcParams['figure.figsize'] = fig_size


def mod_axes(axes, x='top', y='right'):
    """Function that removes the spines of the ax.

    :param list of plt.Axes or plt.Axes axes: Axes from which the spines will be removed.
                                              It admints a list or a single ax.
    :param str x: Which spine to remove from x axis ('top', 'bottom'). (default: 'top').
    :param str y: Which spine to remove from u axis ('left', 'right'). (default: 'right').
    """

    if not isinstance(axes, (list, np.ndarray)):
        axes = np.array([axes])
    elif isinstance(axes, list):
        axes = np.array(axes)

    xticks = 'bottom' if x == 'top' else 'top'
    yticks = 'left' if y == 'right' else 'right'
    for ax in axes.ravel():
        ax.spines[x].set_visible(False)
        ax.spines[y].set_visible(False)
        ax.yaxis.set_ticks_position(yticks)
        ax.xaxis.set_ticks_position(xticks)


def mod_ax_lims(ax, xlims=(0,), ylims=(0,), xbounds=(0,), ybounds=(0,), xticks=(0,), yticks=(0,), xlabel='',
                ylabel='', xformatter='', yformatter=''):
    """ Modify the axis of an ax object: limits, bounds of axis, ticks, labels, legend fontsize,
    format of ticks' numbers.

    :param plt.Axes ax: An axes object where the modifications are performed.
    :param tuple xlims: limits of the x axis.
    :param tuple ylims: limits of the y axis.
    :param tuple xbounds: actual bounds of the x axis.
    :param tuple ybounds: actual bounds of the y axis.
    :param tuple or list xticks: list of x axis ticks.
    :param tuple or list yticks: list of y axis ticks.
    :param str xlabel: label of x axis.
    :param str ylabel: label of y axis.
    :param str xformatter: numerical format (and font) of the x ticks (e.g. '%d').
    :param str yformatter: numerical format (and font) of the y ticks (e.g. '%d').
    :return: None
    """
    ax.set_xlim(*xlims) if xlims != (0,) else None
    ax.set_ylim(*ylims) if ylims != (0,) else None
    ax.set_xticks(xticks) if xticks != (0,) else None
    ax.set_yticks(yticks) if yticks != (0,) else None
    ax.set_xlabel(xlabel) if xlabel != '' else None
    ax.set_ylabel(ylabel) if ylabel != '' else None
    try:
        ax.spines['left'].set_bounds(ybounds) if ybounds != (0,) else None
        ax.spines['bottom'].set_bounds(xbounds) if xbounds != (0,) else None
    except TypeError:
        ax.spines['left'].set_bounds(*ybounds) if ybounds != (0,) else None
        ax.spines['bottom'].set_bounds(*xbounds) if xbounds != (0,) else None

    ax.yaxis.set_major_formatter(FormatStrFormatter(yformatter)) if yformatter != '' else None
    ax.xaxis.set_major_formatter(FormatStrFormatter(xformatter)) if xformatter != '' else None


def sffamily_logformat(y, pos):
    """ Format the tick labels of a logarithmic scaled axis using sans serif font.

    :param float y: number to be formatted.
    :param float pos: position of the tick.
    :return: formatted string.
    """
    exponent = int(np.log10(y))
    return r"10$^{\mathsf{%d}}$" % exponent


"""
Figure and axes builder methods
-------------------------------  
"""


def set_plot_environment(**kwargs):
    """ Function that sets a plotting environment: creates figure and axes if none are are provided in kwargs,
    sets figure size (``figsize``) in case none is provided and modifies the axes using ``mod_axes``.

    :param kwargs: keyword arguments passed first to ``get_fig_ax`` and then to ``plt.subplots``.
    :return: figure, axes, and modified keyword dictionary.
    :rtype: (plt.Figure, plt.Axes or np.ndarray of plt.Axes, dict)
    """
    # plt.rcParams['figure.dpi'] = kwargs.pop('dpi', 300)

    xfig, yfig = kwargs.pop('figsize', (3, 2))
    set_plot_dim(xfig, yfig)

    (fig, axes), kwargs = get_fig_ax(**kwargs)  # get the fig and/or axes passed through kwargs, if any.
    if (fig, axes) == (False, False):
        fig, axes = plt.subplots(**kwargs)  # Create new fig and axes if none passed, with kwargs as kw arguments.
    mod_axes(axes)  # Modify the axes in a default manner (strip-off top and right spines).
    return fig, axes, kwargs


def get_fig_ax(**kwargs):
    """ Gets figure and axe(s) passed through kwargs if any. If a figure is passed, then a single ax is created.

    :param kwargs: keyword arguments containing the figure and ax(es) if any.
    :return: (figure, axes), and modified keyword dictionary.
    :rtype: ((plt.Figure, plt.Axes), dict)
    """
    fig = kwargs.pop('fig', False)
    ax = kwargs.pop('ax', False)
    if not fig and ax:
        if type(ax) == list:
            ax0 = ax[0]
        else:
            ax0 = ax
        fig = ax0.get_figure()

    if fig and not ax:
        # TODO: allow custom grid-placement of the ax(es)
        # TODO: allow multiple axes creation?
        ax = fig.add_subplot(111)

    return (fig, ax), kwargs


def setup_ax(kwargs):
    """ Major wrapper of set_plot_environment, intended to quickly give format to axes and figure, if none provided.
    This function is thought as a way to avoid unnecessary checking in plotting functions.

    :param dict kwargs: dictionary optionally containing ax and figure passed as keyword, which may also contain
                   other keyword arguments which are passed to ``set_plot_environment`` in case an ``ax`` is not given.
    :return: axes, figure and keyword dictionary.
    :rtype: (plt.Axes, plt.Figure, dict)
    """
    fg = kwargs.pop('fig', False)
    ax = kwargs.pop('ax', False)
    if not ax:
        fg, ax, kwargs = set_plot_environment(**kwargs)
        if isinstance(ax, (list, np.ndarray)):
            ax = ax[0]
    elif not fg:
        fg = ax.get_figure()

    return ax, fg, kwargs


def create_colorbar_ax(fig, subplotspec, **kwargs):
    """ Function that formats an ax which will contain a color-bar.

    :param matplotlib.figure.Figure fig: figure where the axes will be created.
    :param matplotlib.gridspec.SubplotSpec subplotspec: SubplotSpec. Location of a subplot in a 'GridSpec'.
                                                  For example, ``gs[0, 0]``, where ``gs`` is a GridSpec object.
    :param kwargs: keyword arguments passed to plt.figure.add_subplot method .
    :return: ax containing the color-bar.
    :rtype: plt.Axes
    """

    ax = fig.add_subplot(subplotspec, **kwargs)
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ['bottom', 'right', 'top', 'left']:
        ax.spines[spine].set_visible(False)

    return ax


"""
Plotting methods
----------------
"""


def plot_error_filled(x, y, errorbar, *args, **kwargs):
    """ Plots y vs. x with a confidence interval represented by the shaded and colored area.

    :param np.ndarray of float x: independent variable.
    :param np.ndarray of float y: dependen variable.
    :param np.ndarray of float errorbar: Symmetric errorbar, that is, it will be plotted using y + errorbar
                                         and y - errorbar.
    :param kwargs: Additional arguments passed to matplotlib.pyplot.plot and matplotlib.pyplot.fill_between
    :return: ax, line plot and errorbar collection.
    :rtype: (plt.Axes, plt.Line2D, matplotlib.collections.PolyCollection)
    """
    ax = kwargs.pop('ax', None)
    if ax is None:
        custom_plot_params()
        f, ax = plt.subplots(1, 1)

    independent_axis = kwargs.pop('indp_axis', 'x')
    alpha = kwargs.pop('alpha', 0.4)
    if independent_axis == 'x':
        p, = ax.plot(x, y, *args, **kwargs)
        eb = ax.fill_between(x, y + errorbar, y - errorbar, alpha=alpha, **kwargs)
    elif independent_axis == 'y':
        p, = ax.plot(y, x, *args, **kwargs)
        eb = ax.fill_betweenx(x, y + errorbar, y - errorbar, alpha=alpha, **kwargs)
    else:
        logging.error("Independent axis ``indp_axis`` not in ('x', 'y').")
        return

    return ax, p, eb


def density_scatter(x, y, sort=True, bins=(16, 16), ax=None, **kwargs):
    """ Scatter plot colored by a 2d histogram. Plots a scatter plot where dots colors are taken from a 2 dimensional,
    interpolated histogram.

    :param np.ndarray of float x: independent (abscissa) variable. 1-d array.
    :param np.ndarray of float y: dependent (ordinate) variable.
    :param bool sort: select if densest points should be plotted last.
    :param (int, int) bins: number of bins to build the histogram.
    :param plt.Axes ax: ax where to perform the plot.
    :param kwargs: additional keyword arguments passed to the ``plt.scatter`` function.
    :return: axes containing the density scatter plot.
    :rtype: plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d",
                bounds_error=False)

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    sca = ax.scatter(x, y, c=z, **kwargs)

    def mkfunc(a, *args):
        return '%1.0fM' % (a * 1e-6) if a >= 1e6 else '%1.0fK' % (a * 1e-3) if a >= 1e3 else '%1.0f' % a

    fig.colorbar(sca, ax=ax, format=FuncFormatter(mkfunc))

    return ax


def fig_grid(fig=None, orientation='h', shift=0.99, color='black'):
    """ Plot a numbered grid-like structure into the figure to have a reference of the location of the different
    elements in a figure, such as axes, text, etc.

    :param plt.Figure fig: figure where to draw the `grid`.
    :param str orientation: orientation of the grid, either horizontal ``h`` (default) or vertical ``v``.
    :param float shift: perpendicular coordinate of the grid, range between 0 and 1.
    :param str color: color of the grid, passed to ``plt.text``.
    :return: 0 if no exception occurred, 1 if no figure was provided
    :rtype: int
    """
    grid = True
    if not fig:
        return 1

    for i in range(10):
        if orientation == 'h':
            x1, y1 = (i / 10.0, shift)
        else:
            y1, x1 = (i / 10.0, shift)
        fig.text(x1, y1, r"%d" % i, va='center', ha='center', fontsize=10, visible=grid, color=color)
        for j in range(0, 10):
            if j != 0:
                if orientation == 'h':
                    x20, y20 = ((i + 0.1 * j) / 10.0, shift)
                else:
                    y20, x20 = ((i + 0.1 * j) / 10.0, shift)
                fig.text(x20, y20, r"%d" % j, va='center', ha='center', fontsize=5, visible=grid, color=color)
            for k in range(1, 10):
                if orientation == 'h':
                    x2, y2 = ((i + 0.1 * j + 0.01 * k) / 10.0, shift)
                else:
                    y2, x2 = ((i + 0.1 * j + 0.01 * k) / 10.0, shift)
                fig.text(x2, y2, r"%d" % k, va='center', ha='center', fontsize=2,
                         visible=grid, color=color)
    return 0


def rectspan(x1, x2, axes, **kwargs):
    """Creates a shaded rectangle that spans over multiple axes.

    :param float x1: first horizontal coordinate in data coordinates.
    :param float x2: second horizontal coordinate in data coordinates.
    :param axes: list of axes where the shaded box must be plot.
    :param kwargs: optional arguments passed to the patch builders.
    """
    alpha_lines = kwargs.pop('alpha_lines', 1.0)
    patches = []
    artists = []
    for ax in axes:
        rect = Rectangle((x1, 0), width=(x2 - x1), height=1.0, transform=ax.get_xaxis_transform(), **kwargs)
        ax.add_patch(rect)
        patches.append(rect)

    for ax1, ax2 in zip(axes[0:-1], axes[1:]):
        for x in (x1, x2):
            kwargs['alpha'] = alpha_lines
            p = ConnectionPatch((x, 1), (x, 0), coordsA=ax2.get_xaxis_transform(), coordsB=ax1.get_xaxis_transform(),
                                **kwargs)
            ax1.add_artist(p)
            artists.append(p)

    return patches, artists


"""
Auxiliary classes
-----------------                    
"""


class MidpointNormalize(mpl.colors.Normalize):
    """ Class that helps renormalizing the color scale. Gives a normalized scale, used in color maps, where
    boundary and middle points can be selected at will.

    *Example*::
        if we want to create a color bar that takes color values from 0 to 1, for example using ``coolwarm``
        color map, and we want the middle point (the ambiguous color) to be located at 0.2 instead of 0.5,
        we may call the class as: ``norm = MidpointNormalize(0, 1, 0.2)``. Then we can use the color
        map with cmap(norm(value_between_0_and_1)).
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        """

        :param float vmin: minimum value.
        :param float vmax: maximum value.
        :param float midpoint: `middle` point.
        :param bool clip: if ``clip`` is True then values out of [``vmin``, ``vmax``] range will be set to 0 or 1.
                         (default: False)
        """
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """

        :param float value: float number between ``vmin`` and ``vmax``, included.
        :param bool clip: if ``clip`` is True then values out of [``vmin``, ``vmax``] range will be set to 0 or 1.
                     (default: False)
        :return: normalized value.
        :rtype: np.ma.core.MaskedArray
        """
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


"""
Auxiliary methods
-----------------                    
"""


def store(obj, name, directory='.', extension='', critical=False, **kwargs):
    """ Function to save an object in binary format (pickled).

    :param object obj: object to be saved. It must be pickleable.
    :param str name: name of the file where the object will be saved.
    :param str directory: path to the file, often a directory starting with './'.
    :param str extension: extension to be used (it may also be included in ``name``).
    :param bool critical: whether the storing process is critical for the survival of the task. (default: False)
    :param kwargs: additional keyword arguments passed to pickle.dump
    :return: 0: no errors. 1: FileNotFoundError, 2: PermissionError, 3: IOError.
    :rtype: int
    """

    log_error = logging.critical if critical else logging.error
    path, filename = os.path.split(name)
    if path == '':
        path = directory
    path = os.path.join(path, filename) + extension
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, **kwargs)
        error = 0
    except FileNotFoundError:
        log_error("Storing of object failed -> Path: '%s' not found." % (path + name))
        error = 1
    except PermissionError:
        log_error("Storing of object failed -> Permission to write to '%s' denied." % (path + name))
        error = 1
    except IOError:
        log_error('Storing of %s failed.' % (path + name))
        error = 3

    if critical and error:
        exit(-1)
    else:
        return error


def retrieve(name, directory='.', extension='.', critical=False, **kwargs):
    """ Function to retrieve an object in binary format (pickled).

    :param str name: name of the file from where the object will be retrieve.
    :param str directory: path to the file, often a directory starting with './'.
    :param str extension: extension to be used (it may also be included in ``name``).
    :param bool critical: whether the retrieving process is critical for the survival of the task. (default: False)
    :param kwargs: additional keyword arguments passed to pickle.load
    :return: object if no errors. None if errors.
    :rtype: object or None
    """

    log_error = logging.critical if critical else logging.error
    path, filename = os.path.split(name)
    if path == '':
        path = directory
    path = os.path.join(path, filename) + extension
    obj = None

    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f, **kwargs)
    except FileNotFoundError:
        log_error("Storing of object failed -> Path: '%s' not found." % (path + name))
    except PermissionError:
        log_error("Storing of object failed -> Permission to write to '%s' denied." % (path + name))
    except IOError:
        log_error('Storing of %s failed.' % (path + name))

    if critical and obj is None:
        exit(-1)
    else:
        return obj


def hex_to_rgb(color_hex, mode='255'):
    """ Converts a color in hexadecimal to a rgb tuple of the form (0-255, 0-255, 0-255).
    If `mode` is set to '1', then it returns a rgb value with each value between 0 and 1.

    :param str color_hex: color in hexadecimal representation.
    :param str mode: sets the range of the returned value.
    :return: a rgb tuple
    :rtype: tuple
    """
    rgb = np.array(tuple(int(color_hex.strip("#")[i:i + 2], 16) for i in (0, 2, 4)))
    if mode == '1':
        rgb = rgb / 255
    return tuple(rgb)
