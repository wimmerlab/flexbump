import sys
import pathlib

import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

from mayavi import mlab
from pyface.api import GUI
import scipy.ndimage
import scipy as sp
from scipy.stats import circmean, linregress
from astropy.stats import bootstrap

sys.path.insert(0, './lib/')

from lib_sconf import Parser, log_conf, check_overwrite, load_obj, save_obj
from lib_plotting import *
from lib_ring import get_phases_and_amplitudes_auto, load_ic, compute_phase, psi_evo_r_constant, circ_dist
from lib_ring import amp_eq_simulation as simulation
from lib_analysis import save_plot, compute_estimation, compute_pm_reg, log_reg, measure_delta_ang, \
    get_amp_eq_stats, running_avg, compute_circ_reg_ml, compute_ppk_slope, equalbins

logging.getLogger('lib_analysis').addHandler(logging.NullHandler())

__author__ = 'Jose M. Esnaola-Acebes'
__docformat__ = 'reStructuredText'


# #### #### #### #### ####
# Figure 1 of the paper ##
# #### #### #### #### ####
@mlab.show
def plot_fig1_3d(datafile=None, trial=100, perfect=False, fframe=-1, **kwargs):
    """3D representation of the activity of the network and stimuli. This 3d plot corresponds to panel C
    on Fig. 1. The plot must be saved from mayavi GUI then imported to GIMP where the alpha channel can be added
    and transform color (white) to alpha. Finally it is cut to the content and saved as .png to be exported to
    Inkscape, where the rest of the elements are added as needed.

    .. Note:
        pefect=True is compulsory for now.

    :param str datafile: file containing the simulation data.
    :param int trial: trial number.
    :param bool perfect: whether the perfect integration is plotted.
    :param int fframe: set the frame of the figure to show the figures in talks (3 frames: 1, 2, 3)
    :param kwargs: additional keyword arguments.
    :return: None
    """
    # Load data
    if datafile is None:
        datafile = pathlib.Path("results/paper_figs/fig1/simu_10000_nobump_sigma-0.15_i0-0.04.npy")

    data = np.load(str(datafile), allow_pickle=True)
    config = dict(data['conf'])
    i0_over = config['i0_init']
    i1 = config['i1']
    modes = config['m']
    bump = not config['no_bump']

    # Select data
    rdata, df = (data['rates'], data['data'])
    # Get phases and amplitudes
    r_aligned, phases, amps, amps_m = get_phases_and_amplitudes_auto(rdata, aligned_profiles=True)

    # Select trial
    selected_data = df.loc[df.chosen != -1]
    data_trial = selected_data.iloc[trial]

    # For debugging:
    logging.debug(data_trial)

    r = rdata[:, trial, :]
    phase = np.rad2deg(phases[:, trial])

    # Simulation variables
    (tsteps, ntrials, n) = rdata.shape
    theta = np.arange(n) / n * 180 * 2 - 180
    dts, cue, nframes = (0.01, 0.250, 8)

    # Stimulus data
    labels = ['x%d' % k for k in range(1, nframes + 1)]
    stim_phases = data_trial[labels].to_numpy(int)
    logging.debug('Orientations of the frames are: %s' % stim_phases)

    # Perfect integration
    if not perfect:
        pph = data['p_ph']
        total_stim = [[]]
    else:
        total_stim = np.zeros((1, n))
        for frame in range(nframes):
            stim = i1 * np.cos(np.deg2rad(theta - stim_phases[frame]))
            total_stim = np.concatenate((total_stim, np.repeat([stim], int(cue / dts), axis=0)))
        total_stim[0] = total_stim[1]
        cumstim = np.cumsum(total_stim, axis=0)
        pph = compute_phase(cumstim, n)

    # Load initial conditions (we will use them to create the first and last periods of time)
    rf = load_ic('./obj/r0_initial_conditions.npy', critical=('n',), n=n, i0_over=i0_over,
                 w0=modes[0], w1=modes[1], w2=modes[2])
    if bump:
        r0 = rf.copy()
    else:
        r0 = np.ones_like(theta) * np.mean(r[0])

    # Force maximum time duration of plotted data
    tpoints = np.arange(0, tsteps) * dts
    targ = int(np.argwhere(np.abs(2.0 - tpoints) <= dts)) + 1
    tpoints = tpoints[:targ]
    r, phase = r[:targ], phase[:targ]

    # Append initial and final states manually
    init_t = 0.2
    init_t_steps = int(init_t / dts)
    final_t = 0.2
    final_t_steps = int(final_t / dts)
    # roll de activity profile to the last phase
    last_phase = phase[-1]

    idx = (np.abs(theta - last_phase)).argmin()
    r_final = np.roll(rf, idx + n // 2)

    r = np.concatenate((np.repeat([r0], init_t_steps, axis=0), r, np.repeat([r_final], final_t_steps, axis=0)))
    tpoints = np.concatenate(
        (np.arange(-init_t, 0.0, dts), tpoints[:-1], np.arange(tpoints[-1], tpoints[-1] + final_t, dts)))
    if pph is not None:
        if not perfect:
            pph = pph[:targ, trial]
        else:
            pph = pph[:targ]
        last_pph = pph[-1]
        pph = np.rad2deg(np.concatenate((np.ones(init_t_steps) * 0.0, pph, np.ones(final_t_steps) * last_pph)))

    # #### #### #### #### #### ####
    # Prepare Mayavi figure    ####
    # #### #### #### #### #### ####
    if fframe == 1:  # Show only the stimulus
        opacity1 = 0.0
        opacity2 = 0.0
    elif fframe == 2:  # Show the stimulus and the activity
        opacity1 = 1.0
        opacity2 = 0.0
    else:  # Show everything
        opacity1 = 1.0
        opacity2 = 1.0
    fig_size = kwargs.pop('figsize', (1920, 1080))
    fig = mlab.figure(size=fig_size)
    try:
        engine = mlab.get_engine()
    except NameError:
        from mayavi.api import Engine
        engine = Engine()
        engine.start()
    if len(engine.scenes) == 0:
        engine.new_scene()

    scene = engine.scenes[0]
    scene.scene.background = (1.0, 1.0, 1.0)
    scene.scene.foreground = (0.0, 0.0, 0.0)

    # Smooth the network's activity and plot
    sigma = [0.02, 10.0]  # Width of the gaussian filter
    smooth_r = sp.ndimage.filters.gaussian_filter(r, sigma, mode='wrap')
    x, y = np.mgrid[tpoints[0]:tpoints[-1]:len(tpoints) * 1j, theta[0]:theta[-1]:200j]
    s = [300, 1, 4]  # Scale up the plot (x, y, z) = (time, space, firing rate)

    # The activity suddenly increases when the noise is removed (expected) but does not look good and it may
    # be confusing: apply an scaling factor to the last period of the trial.
    smooth_r[-final_t_steps:] = smooth_r[-final_t_steps:] / np.max(smooth_r[-final_t_steps:]) * np.max(
        smooth_r[-final_t_steps - 1]) * 1.05
    # Draw the surface
    surface = mlab.mesh(s[0] * x, s[1] * y, s[2] * smooth_r, scalars=smooth_r, colormap='gist_gray', opacity=opacity1)
    surface.module_manager.scalar_lut_manager.reverse_lut = True  # Reverse the colormap
    # Obtain the phase of the smoothed activity
    _, sphase, samp, _ = get_phases_and_amplitudes_auto(smooth_r)
    # Set initial neutral phase at 0
    sphase[:init_t_steps] = 0
    line = mlab.plot3d(s[0] * tpoints, s[1] * np.rad2deg(sphase), s[2] * samp, tube_radius=None,
                       color=(1, 0, 0), opacity=opacity1)

    # Plot stimuli below the network's activity
    if len(total_stim) > 0:
        # Normalize stimuli and scale to 1/4th of the activity of the network
        total_stim = np.concatenate((np.zeros((init_t_steps, n)), np.array(total_stim),
                                     np.zeros((final_t_steps, n))))
        total_stim = (total_stim - np.min(total_stim)) / (np.max(total_stim) - np.min(total_stim))
        total_stim = (total_stim * np.max(smooth_r) / 4)

        # Move it below
        total_stim = total_stim - kwargs.get('sheight', 40)
    # Get the phase of the stimulus
    _, stim_phase, stim_amp, _ = get_phases_and_amplitudes_auto(total_stim)
    stim_phase[0:init_t_steps] = 0
    stim_phase[-final_t_steps:] = 0

    # Draw stimulus surface and its phase on top of the surface
    mlab.mesh(s[0] * x, s[1] * y, s[2] * total_stim, scalars=total_stim, colormap='PuBu')
    stim_line = mlab.plot3d(s[0] * tpoints, s[1] * np.rad2deg(stim_phase), s[2] * stim_amp,
                            tube_radius=None,
                            color=(0, 0, 0))

    # Plot projection of the phase and the perfect integration above the network activity
    height = kwargs.get('pheight', 60) * np.ones_like(tpoints)  # where to draw the projection plane
    height_z = height[0] * np.ones_like(total_stim)
    # Draw a plane in which project the phase
    mlab.mesh(s[0] * x, s[1] * y, s[2] * height_z, color=(1, 1, 1), opacity=opacity2 * 0.5)
    phase_line = mlab.plot3d(s[0] * tpoints, s[1] * np.rad2deg(sphase), s[2] * height, tube_radius=None,
                             color=(1, 0, 0), opacity=opacity2)

    perfect_line = mlab.plot3d(s[0] * tpoints, s[1] * pph, s[2] * height, tube_radius=None,
                               color=(0.204, 0.541, 0.741), opacity=opacity2)

    # Add a box (outline)
    if kwargs.get('outline', False):
        mlab.outline(figure=fig, extent=[s[0] * tpoints[0], s[0] * tpoints[-1], s[1] * theta[0], s[1] * theta[-1],
                                         -s[2] * kwargs.get('sheight', 40), s[2] * height[0]], line_width=1.0)

    # Draw middle dashed line(s): mayavi does not have a "dashed" option for the lines. Produce them manually.
    mlines = []
    dash_interval = int(len(tpoints) / 30)
    for k, tdashed in enumerate(tpoints[::dash_interval]):
        trange = np.array([tdashed, tdashed + dash_interval * dts])
        if k % 2 != 0:
            mlines.append(mlab.plot3d(s[0] * trange, s[1] * np.zeros_like(trange), s[2] * height[0:2],
                                      tube_radius=None,
                                      color=(0, 0, 0), opacity=opacity2))

    # Modify width of lines. Note: exporting to png (or jpeg) with magnification scales differently the objects.
    #                              lines remain constant, thus reducing their width.
    for line_artist in [perfect_line, phase_line]:
        line_artist.actor.property.line_width = kwargs.get('linewidth', 2.0)
    for line_artist in [line, stim_line]:
        line_artist.actor.property.line_width = 2.0

    # Camera position. Obtained from the Mayavi recording tool:
    scene.scene.camera.position = [-1035.1836028162786, -216.23983830128412, 525.1663676339206]
    scene.scene.camera.focal_point = [263.66734015783317, 51.079865017818925, -20.595638645162524]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.3743515731874983, 0.06900964354810182, 0.9247153987846256]
    scene.scene.camera.clipping_range = [553.0046763231907, 2546.78408342354]
    scene.scene.camera.compute_view_plane_normal()

    scene.scene.render()

    if kwargs.get('save', False):
        bump = 'bump' if bump else 'nobump'
        i0_over = config['i0']
        fig_name = pathlib.Path(f"./results/paper_figs/fig1/3dplot/3dplot_{bump}-{i0_over:.2f}_i1-{i1:.2f}_{trial}.png")
        force = kwargs.pop('overwrite', False)
        auto = kwargs.pop('auto', True)
        logger.info(f"Saving 3d plot to '{fig_name}'...")
        fig_name = check_overwrite(fig_name, force=force, auto=auto)
        mlab.savefig(fig_name, magnification=1.0)
        mlab.close()
    return None


def plot_fig1_stats(save=False, **kwargs):
    """ Plots of estimation and psychometric curve shown in panels D, E of Fig. 1 of the paper. Additional
    tweaks are performed in Inkscape.

    :param bool save: whether to save or not the figure.
    :param kwargs: additional keyword arguments.
    :return: figure.
    :rtype: plt.Figure
    """

    # Load data
    datafile = 'results/variable_frames/simu_10000_nobump_sigma-0.15_i0-0.06_frames-8_t-3.0'
    datafile2 = 'results/variable_frames/simu_10000_nobump_sigma-0.15_i0-0.06_frames-4_t-1.0'
    dfs = []
    for dfile in [datafile, datafile2]:
        logging.info(f"Loading data from '{dfile}'...")
        df = pd.read_csv(dfile + '.csv')
        df.attrs['datafile'] = dfile
        dfs.append(df)

    # For debugging:
    logging.debug(f"Head of table:\n{df.head()}")

    # #### #### #### #### #### ####
    # Prepare figure      #### ####
    # #### #### #### #### #### ####
    custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 300

    # Set up grid
    gs1 = gridspec.GridSpec(2, 1)

    # Axes location variables
    (left, right, top, bottom) = (0.12, 0.98, 0.98, 0.1)
    hspace = 0.5
    wspace = 0.5

    gs1.update(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)

    # Create figure and axes
    figsize = kwargs.pop('figsize', (2.0, 3.0))
    fig = plt.figure(figsize=figsize)  # Size is in inches (calculated for a US Letter size in inkscape)
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(False)

    ax_es = fig.add_subplot(gs1[0, 0])
    ax_pm = fig.add_subplot(gs1[1, 0])

    axes = fig.get_axes()
    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.grid(False)

    # Delete some axes
    mod_axes(axes)

    # Actual plotting starts here
    # #### #### #### #### #### ##

    # Some styles and colors
    # color1 = "#5f5f5f"
    # color2 = "k"
    color2 = ["#058c8c", "#bee7e7"]
    color1 = ["#b1b1b1", "#d9d9d9"]
    dashed = (0, (10, 3))
    # Format of estimation ax
    el = kwargs.pop('eslim', 90)
    pmlim = kwargs.pop('pmlim', 45)
    eslims = (-el - 5, el + 5)
    esticks = [-el, -int(el / 2), 0, int(el / 2), el] if el % 2 == 0 else [-el, 0, el]
    xlabel = kwargs.pop('xlabel', r"Average direction $(^\circ)$")
    ylabel = kwargs.pop('ylabel', r"Estimation, $\psi$ $(^\circ)$")
    pmlims = (-pmlim - 5, pmlim + 5)
    pmticks = [-pmlim, -int(pmlim / 2), 0, int(pmlim / 2), pmlim] if pmlim % 2 == 0 else [-pmlim, 0, pmlim]
    mod_ax_lims(ax_es, xlims=eslims, ylims=eslims, ylabel=ylabel, xticks=esticks, yticks=esticks,
                xbounds=(-el, el), ybounds=(-el, el), xformatter='%d', yformatter='%d')
    plt.setp(ax_es.get_xticklabels(), visible=False)
    # Format of psychometric ax
    mod_ax_lims(ax_pm, xlims=pmlims, ylims=(-0.05, 1.0), xlabel=xlabel,
                ylabel=r"Probability CW", xticks=pmticks, yticks=[0, 0.5, 1],
                ybounds=(0, 1), xbounds=(-pmlim, pmlim), xformatter='%d', yformatter='%.1f')

    ax_es.plot([-90, 90], [-90, 90], ls=dashed, color='black', lw=1.0)
    for d, c, ls, mc, nframes in zip(dfs, [color2, color1], ['-', '--'], ['k', 'white'], [8, 4]):
        # Estimation plot
        binning, est, est_err = compute_estimation(d, kwargs.get('nbins', 41), lim=kwargs.get('eslim', 90))
        label = "%d frames" % nframes
        patch = ax_es.fill_between(binning, est - est_err, est + est_err, color=c[1], ls='solid')
        if nframes == 4:
            patch.set_zorder(0)
        else:
            patch.set_edgecolor('white')
        ax_es.plot(binning, est, color=c[0], label=label, ls=ls, lw=1.5)
        ax_es.legend(frameon=False, fontsize=8)

        # Psychometric curve
        x, y = compute_pm_reg(d, lim=pmlim, compute=True, save=False, nframes=nframes, bins=51)  # Fitting points
        ax_pm.plot(x, y, clip_on=False, linewidth=1.5, color=c[0], ls=ls)

        # x, y = compute_pm_data(d, lim=pmlim, bins=21)  # Data points
        # ax_pm.plot(x, y, 'o', clip_on=False, markersize=3.0, markerfacecolor=mc, markeredgewidth=1.0, color=c)
        # ax_pm.errorbar(x, y, yerr=np.array([lq, uq]), fmt='o', clip_on=False, linewidth=0.5,
        #                capsize=1.0, color=c)

    if save:
        directory = 'results/paper_figs/fig1/'
        fig.set_tight_layout(False)
        filename = directory + f"fig1_stats"
        save_plot(fig, filename, overwrite=False, auto=True, saveformat=['png', 'pdf', 'svg'])

    return fig


# #### #### #### #### ####
# Figure 2 of the paper ##
# #### #### #### #### ####
@mlab.show
def plot_fig2_3d(datafile='None', **kwargs):
    """3D representation of the activity of the network, showing the bump formation and its translation for Fig. 2
    of the paper. As in :func:`plot_fig1_3d`, the plot must be saved and modify manually for further modification
    with Inkscape.

    :param str datafile: file containing the simulation data.
    :param kwargs: additional keyword arguments.
    :return: None
    """
    # Load data
    if datafile == 'None':
        datafile = "results/simu_1_nobump_sigma-0.00_i0-0.08_01.npy'"

    data = np.load(datafile, allow_pickle=True)
    config = dict(data['conf'])
    i0_over = config['i0_init']
    i1 = 0.0
    modes = config['m']
    bump = not config['no_bump']

    # Select data
    rdata, df = (data['rates'], data['data'])
    # Get phases and amplitudes
    r_aligned, phases, amps, amps_m = get_phases_and_amplitudes_auto(rdata, aligned_profiles=True)

    # Select trial
    trial = 0  # This is hardcoded here because the default simulation data only has this trial.
    selected_data = df.loc[df.chosen != -1]
    data_trial = selected_data.iloc[trial]

    # For debugging:
    logging.debug(data_trial)

    r = rdata[:, trial, :]
    phase = np.rad2deg(phases[:, trial])

    # Simulation variables
    (tsteps, ntrials, n) = rdata.shape
    theta = np.arange(n) / n * 180 * 2 - 180
    dts, cue, nframes = (0.01, 0.250, 8)

    # Stimulus data
    labels = ['x%d' % k for k in range(1, nframes + 1)]
    stim_phases = data_trial[labels].to_numpy(int)
    logging.debug('Orientations of the frames are: %s' % stim_phases)

    # Prepare stimulus (in the end it won't be shown)
    total_stim = np.zeros((1, n))
    for frame in range(nframes):
        if frame > 3:
            i1 = 0.005
        stim = i1 * np.cos(np.deg2rad(theta - stim_phases[frame]))
        total_stim = np.concatenate((total_stim, np.repeat([stim], int(cue / dts), axis=0)))
    total_stim[0] = total_stim[1]

    # Load initial conditions (we will use them to dray the first period of the simulation)
    base_r0 = kwargs.get('base_r0', 0)
    if bump:
        rf = load_ic('./obj/r0_initial_conditions.npy', critical=('n',), n=n, i0_over=i0_over,
                     w0=modes[0], w1=modes[1], w2=modes[2])
        r0 = rf.copy()
        base_r0 = r0.mean()
    else:
        r0 = np.ones_like(theta) * np.mean(r[base_r0])
    r[0:base_r0] = np.repeat([r0], base_r0, axis=0)

    # Force maximum time duration of plotted data
    tpoints = np.arange(0, tsteps) * dts
    targ = int(np.argwhere(np.abs(2.0 - tpoints) <= dts)) + 1
    tpoints = tpoints[:targ]
    r, phase = r[:targ], phase[:targ]

    # Append initial and final states
    init_t = 0.2
    init_t_steps = int(init_t / dts)

    r = np.concatenate((np.repeat([r0], init_t_steps, axis=0), r))
    tpoints = np.concatenate((np.arange(-init_t, 0.0, dts), tpoints))

    # #### #### #### #### #### ####
    # Prepare Mayavi figure    ####
    # #### #### #### #### #### ####
    fig_size = kwargs.pop('figsize', (1920, 1080))
    fig = mlab.figure(size=fig_size)
    transitions = kwargs.get('transitions', 2)  # 1: plot only the formation of the bump, 2: also plot the translation
    try:
        engine = mlab.get_engine()
    except NameError:
        from mayavi.api import Engine
        engine = Engine()
        engine.start()
    if len(engine.scenes) == 0:
        engine.new_scene()

    scene = engine.scenes[0]
    scene.scene.background = (1.0, 1.0, 1.0)
    scene.scene.foreground = (0.0, 0.0, 0.0)

    # Smooth the network's activity and plot
    sigma = [0.02, 10.0]  # Width of the gaussian filter
    smooth_r = sp.ndimage.filters.gaussian_filter(r, sigma, mode='wrap')
    bump_step = int((len(smooth_r) - init_t_steps) / 2) + init_t_steps
    x, y = np.mgrid[tpoints[0]:tpoints[-1]:len(tpoints) * 1j, theta[0]:theta[-1]:200j]
    s = [300, 1, 4]  # Scale up the plot (x, y, z) = (time, space, firing rate)

    # Draw the surface (even if we only want to plot the first part of the surface; reason: aspect ratio of the figure)
    surface = mlab.mesh(s[0] * x, s[1] * y, s[2] * smooth_r, scalars=smooth_r, colormap='gist_gray', opacity=0.7)
    if transitions == 1:  # Draw the surface only up to the formation of the bump
        surface2 = mlab.mesh(s[0] * x[:bump_step + 1], s[1] * y[:bump_step + 1], s[2] * smooth_r[:bump_step + 1],
                             scalars=smooth_r[:bump_step + 1], colormap='gist_gray', opacity=0.7)
        surface2.module_manager.scalar_lut_manager.reverse_lut = True  # Reverse the colormap
        surface2.actor.property.backface_culling = True  # Set the opacity of the back faces to 0, for clarity.

    surface.module_manager.scalar_lut_manager.reverse_lut = True  # Reverse the colormap
    surface.actor.property.backface_culling = True  # Set the opacity of the back faces to 0, for clarity.
    # Obtain the phase of the smoothed activity
    _, sphase, samp, _ = get_phases_and_amplitudes_auto(smooth_r)
    sphase[0:init_t_steps + base_r0] = 0

    # Plot the activity profile R(ti, \theta) for different times ti
    profile_lines = []
    bump_formation_steps = np.linspace(init_t_steps, bump_step, 5, dtype=int)
    final_step = len(tpoints) - 2
    bump_translation_steps = np.linspace(bump_step, final_step, 5, dtype=int)
    steps = np.unique(np.concatenate((bump_formation_steps, bump_translation_steps)))
    c1 = (0.89, 0.29, 0.2)
    c2 = (0.008, 0.65, 0.988)
    colors = list(np.repeat([c1], 5, axis=0)) + list(np.repeat([c2], 4, axis=0))
    opacities = np.concatenate((np.arange(0.2, 1.2, 0.2), np.arange(0.4, 1.2, 0.2)))
    opacities[opacities > 1.0] = 1.0
    for k, tprofile in enumerate(steps):
        profile_lines.append(mlab.plot3d(s[0] * tpoints[tprofile] * np.ones_like(theta), s[1] * theta,
                                         s[2] * smooth_r[tprofile], tube_radius=None, color=tuple(colors[k]),
                                         opacity=opacities[k]))

    # Also draw the bump_step profile next to the last profile to be able to compare the translation of the bump
    profile_lines.append(mlab.plot3d(s[0] * tpoints[-1] * np.ones_like(theta), s[1] * theta,
                                     s[2] * smooth_r[bump_step], tube_radius=None, color=(0.5, 0.5, 0.5),
                                     opacity=1.0))

    # Plot the stimuli below the activity (although it won't be shown)
    if len(total_stim) > 0:
        # Normalize stimuli and scale to 1/4th of the activity of the network
        total_stim = np.concatenate((np.zeros((init_t_steps, n)), np.array(total_stim)))
        total_stim = (total_stim - np.min(total_stim)) / (np.max(total_stim) - np.min(total_stim))
        total_stim = (total_stim * np.max(smooth_r) / 4)

        # Move it below
        total_stim = total_stim - kwargs.get('sheight', 40)
    # Get the phase of the stimulus
    _, stim_phase, stim_amp, _ = get_phases_and_amplitudes_auto(total_stim)
    stim_phase[0:init_t_steps] = 0
    # Draw the surface and make it invisible (it is necessary to do this to preserve the figure size and perspective)
    stim_surface = mlab.mesh(s[0] * x, s[1] * y, s[2] * total_stim, scalars=total_stim, colormap='PuBu')
    stim_surface.actor.actor.visibility = False

    # Add a box (outline)
    if kwargs.get('outline', False):
        mlab.outline(figure=fig,
                     extent=[s[0] * tpoints[0], s[0] * tpoints[-1], s[1] * theta[0], s[1] * theta[-1], 0, 0],
                     line_width=1.0)

    # middle line(s)
    height = np.max(smooth_r[bump_step]) * np.ones_like(tpoints)
    mlines = []
    dash_points = tpoints[bump_step:]
    dash_interval = int(len(tpoints) / 30)
    if transitions > 1:
        for k, tdashed in enumerate(dash_points[::dash_interval]):
            trange = np.array([tdashed, tdashed + dash_interval * dts])
            if k % 2 != 0:
                mlines.append(mlab.plot3d(s[0] * trange, s[1] * np.zeros_like(trange), s[2] * height[0:2],
                                          tube_radius=None,
                                          color=(0.5, 0.5, 0.5)))

    for line_artist in profile_lines:
        line_artist.actor.property.line_width = kwargs.get('linewidth', 2.0)

    # Camera position. From the mayavi recording tool:
    scene.scene.camera.position = [-575.7457257137944, -831.424669560834, 723.8253146398912]
    scene.scene.camera.focal_point = [252.24734161691953, -0.5793361102525374, 33.607312072128394]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.38778874385592915, 0.328877934963016, 0.8610802483120277]
    scene.scene.camera.clipping_range = [574.6647323268697, 2350.041694489856]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

    # Hide the elements that are not part of this frame
    if transitions < 2:
        surface.actor.actor.visibility = False
        for line in profile_lines[-5:]:
            line.actor.actor.visibility = False

    # save the generate plot
    if kwargs.get('save', False):
        fig_name = f"./results/paper_figs/fig2_potential_transitions/fig2_transition.png"
        fig_name = kwargs.get('fig_name', fig_name)
        force = kwargs.pop('overwrite', False)
        auto = kwargs.pop('auto', False)
        logger.info(f"Saving file to '{fig_name}'...")
        fig_name = check_overwrite(fig_name, force=force, auto=auto)
        mlab.savefig(fig_name, magnification=1.0)

    return None


# #### #### #### #### ####
# Figure 3 of the paper ##
# #### #### #### #### ####
def plot_fig3_ppk(i0s=(0.02, 0.05, 0.08), save=False, **kwargs):
    """ Estimation plots, temporal integration and psychometric curves.

    :param i0s:
    :param save:
    :param kwargs:
    :return:
    """
    # Get kernels
    _, (w_ring, w_amp), _ = get_amp_eq_stats(i0s, rinit=kwargs.get('pkrinit', 0.2), **kwargs)
    # Bump case
    stats = pd.read_csv('results/integration_stats_frames-8_t-2.0_new.csv')
    bdata = stats.loc[(stats.sigma == 0.15) & (stats.bump == True) & (stats.frames == 8) & (stats.t == 2.0) &
                      (stats.biased == True)].copy()
    bdata.sort_values('i0', inplace=True)

    # Get kernels for the unbiased bump condition and biased bump condition
    df = pd.read_csv('./results/integration_stats_frames-8_t-2.0_new.csv')
    sel = df.loc[(df.bump == True) & (df.biased == True) & (df.sigma == 0.15)].copy()
    sel2 = df.loc[(df.bump == True) & (df.biased == False) & (df.sigma == 0.15)].copy()
    xlabels = ['x%d' % k for k in range(1, 9)]
    sel = sel[xlabels + ['i0']].copy()

    # #### #### #### #### #### ####
    # Prepare figure      #### ####
    # #### #### #### #### #### ####
    custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 200

    # Set up grid
    gs1 = gridspec.GridSpec(1, 3)
    gs2 = gridspec.GridSpec(2, 2)
    gs3 = gridspec.GridSpec(2, 1)

    # Axes location variables
    margins = dict(left=0.1, right=0.96, top=0.98, bottom=0.8, hspace=0.5, wspace=0.2)
    gs1.update(**margins)
    margins.update(hspace=0.6, wspace=0.7, top=(margins['bottom'] - 0.15), bottom=0.1, right=0.88)
    gs2.update(**margins)
    margins.update(left=0.89, right=0.96, wspace=0)
    gs3.update(**margins)

    # Create figure and axes
    figsize = kwargs.pop('figsize', (3.9, 2.9))
    fig = plt.figure(figsize=figsize)  # Size is in inches (calculated for a US Letter size in inkscape)
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(False)

    # kernel axes
    knax = [fig.add_subplot(gs1[0, k]) for k in range(3)]
    # Slope ax
    sax = fig.add_subplot(gs2[0, 0])
    # Amplitudes ax
    aax = fig.add_subplot(gs2[0, 1])
    # Slope vs. time ax
    tax = fig.add_subplot(gs2[1, 0], xscale='log')
    # Slope vs. frames ax
    fax = fig.add_subplot(gs2[1, 1], xscale='log')
    # R distributions
    rax = fig.add_subplot(gs3[0, 0], sharey=aax)

    axes = fig.get_axes()
    for axb in axes:
        axb.patch.set_alpha(0.0)
        axb.grid(False)

    # Delete some axis
    mod_axes(axes)
    # Delete all the axis of the distribution ax
    for spine in ['top', 'bottom', 'right', 'left']:
        rax.spines[spine].set_visible(False)

    # Some styles
    cs = ['red', 'black', 'blue']
    dashed = (0, (10, 3))
    # Actual plotting starts here
    # #### #### #### #### #### ##
    # Psychophysical kernel (estimation)
    for k, (wr, wa) in enumerate(zip(w_ring, w_amp)):
        frames = np.linspace(0, 2.0, len(wr))
        knax[k].plot(frames, wr, color=cs[k], lw=1.5)
        if kwargs.get('supp', False):
            # Select the bump kernel
            bker = sel.loc[sel.i0 == i0s[k], xlabels].to_numpy().ravel()
            # Recompute the kernels for the unbiased case taking only the 7 last frames
            if kwargs.get('unbiased_pk', False):
                xlabels2 = ['xu%d' % k for k in range(2, 9)]
                sel2 = sel2[xlabels2 + ['i0']].copy()
                bker2 = sel2.loc[sel2.i0 == i0s[k], xlabels2].to_numpy().ravel()
            else:
                sel2 = sel2[xlabels + ['i0']].copy()
                bker2 = sel2.loc[sel2.i0 == i0s[k], xlabels].to_numpy().ravel()
            frames2 = frames[len(frames) - len(bker2):]
            knax[k].plot(frames, bker, '-', alpha=0.5, lw=1.5)
            knax[k].plot(frames2, bker2, '-', lw=1.5)
        else:
            knax[k].plot(frames, wa, '--', color=(0.6, 0.6, 0.6), lw=1.5)
        mod_ax_lims(knax[k], xlims=(-0.1, frames[-1]), xticks=(0, 1, 2), ylims=(-0.01, 0.3), yticks=[0, 0.1, 0.2, 0.3],
                    xbounds=(frames[0], frames[-1]), ybounds=(0, 0.3), xformatter='%d', yformatter='%.1f',
                    xlabel='Time (s)')
        if k == 0:
            knax[k].set_ylabel(r"Stim. impact")
        else:
            plt.setp(knax[k].get_yticklabels(), visible=False)
        # Label
        knax[k].text(0.97, 0.03, r"$I_0 = %.2f$" % i0s[k], transform=knax[k].transAxes, ha='right', va='bottom',
                     fontsize=8)

    # Slope vs. i0
    xtit = r"Excitatory drive, $I_0$"
    # load data
    df = pd.read_csv('./results/integration_stats_frames-8_t-2.0_new.csv')
    # Select the data with `sigma` = 0 and nobump initial condition
    df = df.loc[(df.sigma == 0.15) & (df.bump == False)].copy()
    df = df.sort_values('i0')
    xmax = df.i0.to_numpy()[-1]
    sax.plot([0, xmax], [0, 0], linestyle=dashed, color='black', lw=0.5)
    sax.plot(df.i0, df.ppk_slope, color='black', lw=1.5)
    sax.plot(bdata.i0, bdata.ppk_slope, color='#3586cf', lw=1.5, alpha=0.385)
    for k, i0 in enumerate(i0s):
        markerops = dict(color=cs[k], markersize=3.0, markerfacecolor='white', markeredgewidth=1.0, zorder=100)
        sax.plot([i0], [df.loc[df.i0 == i0].ppk_slope.to_numpy()], 'o', **markerops)
        tax.plot([2.0], [df.loc[df.i0 == i0].ppk_slope.to_numpy()], 'o', **markerops)
        fax.plot([250], [df.loc[df.i0 == i0].ppk_slope.to_numpy()], 'o', **markerops)

    mod_ax_lims(sax, xlims=(0.0, xmax), xticks=(0.01, 0.1, 0.2), ylims=(-0.65, 0.65), yticks=[-0.6, 0, 0.6],
                xbounds=(0.01, xmax), ybounds=(-0.6, 0.6), xformatter='%.2f', yformatter='%.1f', xlabel=xtit,
                ylabel="PK slope")

    # Amp vs. time
    # load data
    grays = plt.get_cmap('gist_gray_r')
    amps = np.loadtxt('./results/amplitudes_frames-8_t-2.0.txt')
    tpoints = np.arange(len(amps[0])) * 0.01
    mean = np.mean
    k = 0
    for j, (amp, i0) in enumerate(zip(amps, df.i0.to_numpy())):
        if i0 not in i0s:
            if i0 == 0.15:
                # aax.plot(tpoints[1:], amp[1:], color=grays(0.3 + 0.7 * j / (1 + len(amps))), lw=1, alpha=0.5)
                aax.plot(tpoints[1:], amp[1:], color="#50a35c", lw=1, alpha=1)
            continue
        aax.plot(tpoints[1:], amp[1:], color=cs[k], lw=1.5, zorder=50)
        # label = r"$I_0 = %.2f$" % i0
        # aax.text(2.05, amp[-1], label, ha='left', va='center', fontsize=8)
        # Distributions of amplitudes
        ampdata0 = pd.read_csv(f"results/paper_figs/fig3_kernels/data_april-21/"
                               f"simu_10000_nobump_sigma-0.15_i0-{i0:.3f}_frames-8_t-2.0.csv")
        ampdata = ampdata0.loc[np.abs(ampdata0.average_circ) < 45].copy()
        if k == 0:
            thetas = np.zeros((int(2.0 / 0.01), len(ampdata0)))
            for m, x in enumerate(xlabels):
                t0 = int(0.250 / 0.01) * m
                t1 = int(0.250 / 0.01) * (m + 1)
                thetas[t0:t1] = np.repeat([np.deg2rad(ampdata0[x])], t1 - t0, axis=0)
            i1s = np.ones((int(2.0 / 0.01), len(ampdata0))) * 0.005 * 0.01 / 2E-4
            pvi = np.cumsum(i1s * np.exp(1j * thetas), axis=0)
            ampsf_full = np.real(np.sqrt(pvi * np.conjugate(pvi)))
            thetas = np.zeros((int(2.0 / 0.01), len(ampdata)))
            for m, x in enumerate(xlabels):
                t0 = int(0.250 / 0.01) * m
                t1 = int(0.250 / 0.01) * (m + 1)
                thetas[t0:t1] = np.repeat([np.deg2rad(ampdata[x])], t1 - t0, axis=0)
            i1s = np.ones((int(2.0 / 0.01), len(ampdata))) * 0.005 * 0.01 / 2E-4
            pvi = np.cumsum(i1s * np.exp(1j * thetas), axis=0)
            ampsf = np.real(np.sqrt(pvi * np.conjugate(pvi)))
            factor = 0.5
            sns.kdeplot(factor * ampsf[-1] + 12.5, shade=True, ax=rax, color=(0.6, 0.6, 0.6), vertical=True, lw=0.5,
                        ls='--', zorder=100)
            rax.plot(0.202, mean(factor * ampsf[-1] + 12.5), '<', ms=2, color=(0.6, 0.6, 0.6), clip_on=False,
                     zorder=100)
            aax.plot(tpoints[1:], mean(factor * ampsf_full[1:] + 12.5, axis=1), '--', color=(0.6, 0.6, 0.6),
                     lw=1.5, zorder=2)
            # aax.plot(tpoints[1:], mean(factor*ampsf[1:] + 12.5, axis=1), '-', color='white', lw=1.5, zorder=50)
        sns.kdeplot(ampdata.amp_last.to_numpy(), shade=True, ax=rax, color=cs[k], vertical=True, lw=0.5)
        rax.plot(0.2, mean(ampdata.amp_last), '<', ms=4, color=cs[k], clip_on=False)

        k += 1

    mod_ax_lims(aax, xlims=(0, 0.), xticks=(0, 1, 2), ylims=(-5, 55), yticks=[0, 25, 50],
                xbounds=(0, 2), ybounds=(0, 55), xformatter='%d', yformatter='%d', xlabel='Time (s)',
                ylabel="Firing rate (Hz)")
    mod_ax_lims(rax, xlims=(0, 0.2), xticks=[])
    rax.spines['bottom'].set_visible(False)
    rax.spines['left'].set_visible(False)
    plt.setp(rax.get_yticklabels(), visible=False)
    rax.tick_params(axis='both', which='both', length=0)
    # Slope vs. time
    # load data
    # slopes = pd.read_csv('./results/ppk_slopes_duration-0.25_frames-2to16.csv')
    slopes = pd.read_csv('/home/jesnaola/Descargas/Fig3E_PK_slope_vs_stim_duration.csv')
    print(slopes)
    tax.plot([0.05, 20.0], [0, 0], linestyle=dashed, color='black', lw=0.5)
    slopes.reset_index(inplace=True)
    for k, i0 in enumerate([2, 5, 8]):
        tax.plot(slopes['stim_duration_in_ms']/1000, slopes[f'I0_{i0}'], label=r"$I_0 = %.2f$" % (i0*0.01), color=cs[k], lw=1.5)

    # TODO: no noise
    # tax.plot(sl.t, sl.slope, label=r"$I_0 = %.2f$" % i0, color='darkgray', alpha=0.4)

    mod_ax_lims(tax, xlims=(0.06, 21.0), xticks=(0.1, 1.0, 10.0, 20.0), ylims=(-0.6, 1.1), yticks=[-0.5, 0, 0.5, 1.0],
                xbounds=(0.06, 21.0), ybounds=(-0.5, 1.0), xformatter='%.1f', yformatter='%.1f',
                xlabel=r"Stimulus duration (s)", ylabel="PK slope")
    tax.get_xaxis().set_minor_formatter(FormatStrFormatter(''))

    # Slope vs. duration of frame
    # load data
    slopes = pd.read_csv('./results/ppk_slopes_tmax-2.0_frames-2to16.csv')
    dur = slopes.loc[(slopes.i0 == 0.02)].sort_values('duration').duration.to_numpy() * 1000
    fax.plot([dur[0], dur[-1]], [0, 0], linestyle=dashed, color='black', lw=0.5)
    for k, i0 in enumerate(i0s):
        sl = slopes.loc[slopes.i0 == i0]
        sl = sl.sort_values('duration')
        fax.plot(sl.duration.to_numpy() * 1000, sl.ppk_slope, label=r"$I_0 = %.2f$" % i0, color=cs[k], lw=1.5)

    mod_ax_lims(fax, xlims=(dur[0] - 10, dur[-1]), xticks=(dur[0], 500, dur[-1]),
                ylims=(-0.6, 0.6), yticks=[-0.5, 0, 0.5], xbounds=(dur[0], dur[-1]), ybounds=(-0.5, 0.5),
                xformatter='%d', yformatter='%.1f', xlabel=r"Frame duration (ms)", ylabel="PK slope")
    fax.get_xaxis().set_minor_formatter(FormatStrFormatter(''))

    if save:
        directory = './results/paper_figs/fig3_kernels/'
        fig.set_tight_layout(False)
        filename = directory + f"fig_ring_kernels"
        save_plot(fig, filename, overwrite=False, auto=True, saveformat=['png', 'pdf', 'svg'])

    return fig


# #### #### #### #### ####
# Figure 4 of the paper ##
# #### #### #### #### ####
def plot_fig4_est(i0s=(0.02, 0.05, 0.08), save=False, **kwargs):
    """ Estimation plots, psychometric curves and variability of the estimation.

    :param i0s:
    :param save:
    :param kwargs:
    :return:
    """
    # Get estimation, kernels, and pm (data and fit)
    (ering, _), (w_ring, _), (pmring, _) = get_amp_eq_stats(i0s, rinit=kwargs.get('pkrinit', 0.2), **kwargs)
    pmring_reg, pmring_dat = pmring

    # #### #### #### #### #### ####
    # Prepare figure      #### ####
    # #### #### #### #### #### ####
    custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 200

    # Set up grid
    gs1 = gridspec.GridSpec(2, 3)
    gs2 = gridspec.GridSpec(1, 2)

    # Axes location variables
    margins = dict(left=0.12, right=0.98, top=0.98, bottom=0.48, hspace=0.4, wspace=0.15)
    gs1.update(**margins)
    margins.update(top=(margins['bottom'] - 0.15), bottom=0.1, wspace=0.5, hspace=0.0)
    gs2.update(**margins)

    # Create figure and axes
    figsize = kwargs.pop('figsize', (3.9, 2.9))
    fig = plt.figure(figsize=figsize)  # Size is in inches (calculated for a US Letter size in inkscape)
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(False)

    # estimation axes
    esax = [fig.add_subplot(gs1[0, k]) for k in range(3)]
    # kernel axes (as insets)
    knax = [inset_axes(esax[k], width="35%", height="30%", loc='upper left', borderpad=0.5) for k in range(3)]
    # psychometric axes
    pmax = [fig.add_subplot(gs1[1, k]) for k in range(3)]
    # estimation error ax
    eeax = fig.add_subplot(gs2[0, 0])
    # accuracy ax
    acax = fig.add_subplot(gs2[0, 1], sharex=eeax)

    axes = fig.get_axes()
    for axb in axes:
        axb.patch.set_alpha(0.0)
        axb.grid(False)

    # Delete some axes
    mod_axes(axes)
    cs = ['red', 'black', 'blue']

    # Actual plotting starts here
    # #### #### #### #### #### ##
    # Estimation and Psychophysical kernel (estimation)
    dashed = (0, (10, 3))
    el = kwargs.pop('eslim', 60)
    pl = kwargs.pop('pmlim', 60)
    xtit = kwargs.pop('xlabel', r"Avg. direction $(^\circ)$")
    ytit = kwargs.pop('ytit', r"Estimation $\psi$ $(^\circ)$")

    for k, (er, axt, wr, axb, pmr, pmd, axp) in \
            enumerate(zip(ering, esax, w_ring, knax, pmring_reg, pmring_dat, pmax)):

        # Plot estimation curve
        axt.plot([-90, 90], [-90, 90], ls=dashed, color='black', lw=0.5)
        axt.plot(er[0], er[1], color=cs[k])
        axt.fill_between(er[0], er[1] - er[2], er[1] + er[2], color=cs[k], alpha=0.3)
        axt.text(0.95, 0.05, r"$I_0 = %.2f$" % i0s[k], transform=axt.transAxes, ha='right', va='bottom')

        # Format ax
        ticks = [-el, -int(el / 2), 0, int(el / 2), el] if el % 2 == 0 else [-el, 0, el]
        mod_ax_lims(axt, xlims=(-el - 5, el + 5), ylims=(-el - 5, el + 5), xticks=ticks, yticks=ticks,
                    xbounds=(-el, el), ybounds=(-el, el), xformatter='%d')
        if k == 0:
            axt.set_ylabel(ytit)
            axt.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        else:
            plt.setp(axt.get_yticklabels(), visible=False)
        plt.setp(axt.get_xticklabels(), visible=False)

        # Plot ppk
        axb.plot(np.arange(1, 9), wr, color=cs[k])
        axb.spines['left'].set_visible(False)
        # axb.spines['bottom'].set_visible(False)
        mod_ax_lims(axb, xlims=(0.5, 8.5), ylims=(0, 0.3), xticks=[], yticks=[], xbounds=(1, 8), ybounds=(0, 0.3))
        plt.setp(axb.get_yticklabels(), visible=False)
        plt.setp(axb.get_xticklabels(), visible=False)
        axb.patch.set_alpha(0.3)
        axb.set_facecolor('darkgray')

        # Psychometric curve
        axp.plot(pmr[0], pmr[1], color=cs[k])
        # Psychometric points
        axp.plot(pmd[0], pmd[1], 'o', clip_on=False, markersize=3.0, markerfacecolor='white', markeredgewidth=1.0,
                 color=cs[k])
        ticks = [-pl, -int(pl / 2), 0, int(pl / 2), el] if pl % 2 == 0 else [-pl, 0, pl]
        mod_ax_lims(axp, xlims=(-(pl + 5), pl + 5), ylims=(-0.05, 1.01),
                    xticks=ticks, yticks=[0, 0.5, 1], xbounds=(-pl, pl), ybounds=(0, 1),
                    xformatter='%d', xlabel=xtit)

        if k != 0:
            plt.setp(axp.get_yticklabels(), visible=False)
        else:
            axp.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axp.set_ylabel(r"Probability CW")
            # axp.set_xlabel(xtit, position=(0, 0), ha='left')

    # Panel C: estimation error
    # data = get_sqr_errors()
    int_filename = kwargs.get('int_filename', 'results/integration_stats_frames-8_t-2.0.csv')
    data = pd.read_csv(int_filename)
    data = data.loc[(data.bump == False) & ((data.sigma == 0.00) | (data.sigma == 0.15))].copy()
    i0range = np.sort(np.unique(data.i0.to_numpy()))
    ticks = list(np.sort(np.concatenate(([0.05, 0.1], [i0range[0], i0range[-1]]))))
    colors = ['#f37f24', 'black']  # Before it was: '#348abd'
    c_pi = '#988ed5'
    for sigma, c in zip([0.00, 0.15], colors):
        correction = 2 * (0.15 - sigma) ** 2
        sel = data.loc[(data.sigma == sigma) & ((data.i0 + correction) <= 0.2)].copy()
        sel = sel.sort_values('i0')
        eeax.plot(sel.i0 + correction, np.sqrt(sel.sqr_error), '-', label=r"$\sigma_{\mathrm{OU}} = %.2f$" % sigma,
                  lw=1.5,
                  color=c)
        if sigma == 0.15:
            eeax.plot(sel.i0 + correction, np.sqrt(sel.pi_error), label=r"Weighted avg.", ls='--', lw=1.5, color=c_pi)
            sel = sel.loc[sel.i0.isin(i0s)]
            for j, (_, row) in enumerate(sel.iterrows()):
                eeax.plot(row.i0 + correction, np.sqrt(row.sqr_error), 'o', color=cs[j],
                          markersize=3.0, markerfacecolor='white', markeredgewidth=1.0)

    mod_ax_lims(eeax, xlims=(0, i0range[-1] + 0.01), xticks=ticks, ylims=(-2, 100), yticks=(0, 50, 100),
                ybounds=(0, 100), xbounds=(0.01, i0range[-1]), xformatter='%.2f', yformatter='%d',
                xlabel=r"Excitatory drive $I_0$", ylabel=r"Estimation error ($^\circ$)")
    eeax.legend(frameon=False, fontsize=8)
    # plt.setp(eeax.get_yticklabels(), rotation=45)
    # eeax.tick_params(axis='y', which='major', pad=0)

    # Panel F: accuracy
    # load data
    # data = pd.read_csv(int_filename)
    # data = data.loc[(data.sigma == 0.15) & (data.bump == False)].copy()
    # data = data.sort_values('i0')
    # i0range = data.i0.to_numpy()
    # acc = kwargs.get('acc_label', 'pmyr26')
    #
    # # acax.plot(i0range, 100 * data[acc], 'o', color=color, markersize=3.0)
    # acax.plot(i0range, 100 * data[acc], '-', color=colors[1], lw=1.5)
    # sel = data.loc[data.i0.isin(i0s)]
    # for j, (_, row) in enumerate(sel.iterrows()):
    #     acax.plot(row.i0, 100 * row[acc], 'o', color=cs[j],
    #               markersize=3.0, markerfacecolor='white', markeredgewidth=1.0)
    # ticks = list(np.sort(np.concatenate(([0.05, 0.1], [i0range[0], i0range[-1]]))))
    # mod_ax_lims(acax, xlims=(0, i0range[-1] + 0.01), ylims=(50, 100), xticks=ticks, yticks=[50, 75, 100],
    #             xbounds=(i0range[0], i0range[-1]), ybounds=(50, 100), xformatter='%.2f', yformatter='%d',
    #             xlabel=r"Excitatory drive $I_0$", ylabel=r"Accuracy (\%)")
    #
    # New panel F: estimation error vs. stimulus duration
    data = pd.read_csv("~/Descargas/Fig4C_estimation_error_vs_stim_duration.csv")
    acax.set_xscale('log')
    x = data['stim_duration_in_ms'] * 10**(-3)
    for sig in [0.0, 0.05, 0.10, 0.15]:
        y = data[f"sigma_{int(sig*100):02d}"]
        acax.plot(x, y, label=r"%.2f" % sig)
    mod_ax_lims(acax, xlims=(0.06, 21.0), ylims=(-1, 45), xticks=[0.1, 1.0, 10], yticks=[0, 20, 40],
                xbounds=(0.07, 21), ybounds=(0, 45), yformatter='%d', xformatter='%.1f',
                xlabel=r"Stimulus duration (s)", ylabel=r"Estimation error ($^\circ$)")
    acax.legend(title=r"$\sigma_{OU}$", frameon=False, fontsize=8)

    if save:
        directory = './results/paper_figs/fig4_stats/'
        fig.set_tight_layout(False)
        filename = directory + f"fig_4_stats"
        save_plot(fig, filename, overwrite=True, auto=True, saveformat=['png', 'pdf', 'svg'])

    return fig


# #### #### #### #### ####
# Figure 5 of the paper ##
# #### #### #### #### ####
def plot_fig5_bump(save=False, **kwargs):
    """ Plot that compares bump vs. no-bump dynamics: estimation distribution, bias and PK slope.

    :param save: save the plot.
    :param kwargs: additional options.
    :return: figure
    """
    biased = not kwargs.pop('supp', False)
    # Load the data for the first 3 panels
    nbdata = np.load("./results/variable_frames/simu_10000_nobump_sigma-0.15_i0-0.06_frames-8_t-3.0.npy",
                     allow_pickle=True)
    bref = ('biased_' if biased else '')
    bdata = np.load(f"./results/variable_frames/simu_10000_bump_{bref}sigma-0.15_i0-0.06_frames-8_t-2.0.npy",
                    allow_pickle=True)

    # #### #### #### #### #### ####
    # Prepare figure      #### ####
    # #### #### #### #### #### ####
    custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 200

    # Set up grid
    gs1 = gridspec.GridSpec(1, 2)  # Bump phases and distributions (as insets)
    gs2 = gridspec.GridSpec(1, 3)  # Estimation bias, std. and pk slope vs I0
    # Aux. grid for panel E (errors)
    gs3 = gridspec.GridSpec(1, 3)  # Estimation bias, std. and pk slope vs I0

    # Axes location variables
    margins = dict(left=0.07, right=0.98, top=0.98, bottom=0.6, hspace=0, wspace=0.2)
    gs1.update(**margins)
    margins.update(top=(margins['bottom'] - 0.15), left=0.07, bottom=0.12, wspace=1)
    gs2.update(**margins)
    margins.update(left=0.04, right=0.94)
    gs3.update(**margins)

    # Create figure and axes
    figsize = kwargs.pop('figsize', (3.9, 2.9))
    fig = plt.figure(figsize=figsize)  # Size is in inches (calculated for a US Letter size in inkscape)
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(False)

    # Phases axes
    phax = [fig.add_subplot(gs1[0, k]) for k in range(2)]
    # Distribution inset axes
    dbax = [inset_axes(phax[k], width="18%", height="100%", loc='center right', borderpad=0) for k in range(2)]
    # Estimation ax
    esax = fig.add_subplot(gs2[0, 0])
    # estimation bias ax
    ebax = fig.add_subplot(gs2[0, 1])
    # std ax
    stax = fig.add_subplot(gs3[0, 2], sharex=ebax)

    # Delete some axes
    mod_axes(fig.get_axes())

    # Second axis for the squared error:
    stax2 = stax.twinx()
    mod_axes(stax2, x='top', y='left')

    axes = fig.get_axes()
    for axb in axes:
        axb.patch.set_alpha(0.0)
        axb.grid(False)

    # Some style elements
    cs = [['#e6443f', '#d8944a'], ['#3586cf', '#47cfcd']]
    dashed = (0, (10, 3))

    # Actual plotting starts here
    # #### #### #### #### #### ##
    # Phase evolutions
    xtit = "Time (s)"
    ytit = r"$\psi (t)$ ($^\circ$)"

    angles = kwargs.get('angles', [0, 45])
    tol_b = kwargs.get('ang_tol_b', 2)
    tol_nb = kwargs.get('ang_tol_nb', 2)
    double = kwargs.get('double', False)
    labels = [r"\bf \sffamily No initial bump", r"\bf \sffamily Initial bump"]
    for k, (data, ax, dax, cpair, tol) in enumerate(zip([nbdata, bdata], phax, dbax, cs, [tol_nb, tol_b])):
        df = data['data']
        df_chosen = df.loc[df.chosen != -1].copy().reset_index(drop=True)  # Select trials with saved activity
        rates = data['rates']
        t = np.arange(0, 200) * 0.01
        ax.plot([t[0], t[-1]], [0, 0], ls=dashed, lw=0.5, zorder=1, color='k')  # Reference at zero degrees

        for th, c in zip(angles, cpair):  # Select the data whose average orientation ie equal to th
            # For the histogram
            hist_sel = df.loc[(np.abs(df.average_circ - th) < tol) | (np.abs(df.average_circ + th) < tol)].copy()
            # Select trials with average orientation th that have an activity representation
            sel1 = df_chosen.loc[np.abs(df_chosen.average_circ - th) < tol].copy()  # th sign(positive)
            _, ph1, _, _ = get_phases_and_amplitudes_auto(rates[:, sel1.index.to_list()])  # Obtain phase
            ph1 = np.rad2deg(ph1)
            if len(ph1) > 200:  # Take only the first 2 seconds of activity
                ph1 = ph1[:200]

            # Plot the phase evolution
            ax.plot(t, ph1, color=c, alpha=0.4)
            if th != 0:
                sel2 = df_chosen.loc[np.abs(df_chosen.average_circ + th) < tol].copy()  # th sign(negative)
                _, ph2, _, _ = get_phases_and_amplitudes_auto(rates[:, sel2.index.to_list()])
                ph2 = -np.rad2deg(ph2)
                if len(ph2) > 200:  # Take only the first 2 seconds of activity
                    ph2 = ph2[:200]
                hist_sel.loc[hist_sel.category == -th, 'estim'] = -hist_sel.loc[hist_sel.category == -th, 'estim']
                if double:
                    ax.plot(t, ph2, color=c, alpha=0.4)
            # Plot the distribution
            sns.kdeplot(hist_sel.estim.to_numpy(), shade=True, ax=dax, color=c, vertical=True, lw=0.5)
            avg = circmean(hist_sel.estim, low=-180, high=180)
            dax.plot([0.075], [avg], '<', color=c, markersize=4.0, clip_on=False)
            ax.plot([2.06], [th], marker='_', color=c, markersize=4.0, markeredgewidth=1.0)

        # Modify axes style
        mod_ax_lims(ax, xlims=(-0.05, 2.55), ylims=(-95, 95), xticks=[0, 1, 2], yticks=[-90, -45, 0, 45, 90],
                    xbounds=(0, 2), ybounds=(-90, 90), xformatter='%d', yformatter='%d', xlabel=xtit)
        ax.text(2.0, -85, labels[k], ha='right', va='bottom', fontsize=10, color=cpair[0])
        if k == 0:
            # ax.set_xlabel(xtit, position=(0, 0), ha='left')
            ax.set_ylabel(ytit)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

        mod_ax_lims(dax, ylims=(-95, 95), xlims=(-0.01, 0.08), xticks=[], yticks=[])
        dax.spines['bottom'].set_visible(False)
        dax.spines['left'].set_visible(False)
        plt.setp(dax.get_yticklabels(), visible=False)

    # Estimation plot
    el = kwargs.pop('eslim', 60)
    xtit = kwargs.pop('xlabel', r"Avg. direction $(^\circ)$")
    ytit = kwargs.pop('ytit', r"Estimation $\psi$ $(^\circ)$")

    es_nb = compute_estimation(nbdata['data'])
    es_b1 = compute_estimation(bdata['data'], ylabel='ph4')
    es_b2 = compute_estimation(bdata['data'])

    esax.plot(es_nb[0], es_nb[1], color=cs[0][0], label='No bump', lw=1.5)
    esax.plot(es_b1[0], es_b1[1], color=cs[1][0], label="Bump, $t=1$s", ls='--', lw=1.5)
    esax.plot(es_b2[0], es_b2[1], color=cs[1][0], label="Bump, $t=2$s", lw=1.5)
    esax.plot([-el, el], [-el, el], ls=dashed, color='black', lw=1.0)

    esax.legend(frameon=False, fontsize=8, loc='lower right')

    # Format ax
    ticks = [-el, -int(el / 2), 0, int(el / 2), el] if el % 2 == 0 else [-el, 0, el]
    mod_ax_lims(esax, xlims=(-el - 5, el + 5), ylims=(-el - 5, el + 5), xticks=ticks, yticks=ticks,
                xbounds=(-el, el), ybounds=(-el, el), xformatter='%d', yformatter='%d', xlabel=xtit, ylabel=ytit)

    # Load data for the rest of the plot
    stats = pd.read_csv('results/integration_stats_frames-8_t-2.0_new.csv')
    nbdata = stats.loc[(stats.sigma == 0.15) & (stats.bump == False) & (stats.frames == 8) & (stats.t == 2.0)].copy()
    nbdata.sort_values('i0', inplace=True)
    bdata = stats.loc[(stats.sigma == 0.15) & (stats.bump == True) & (stats.frames == 8) & (stats.t == 2.0) &
                      (stats.biased == biased)].copy()
    bdata.sort_values('i0', inplace=True)
    if not biased:
        bdata.ppk_slope = bdata.ppk_slope_u

    labels = ['No bump', 'Bump']
    for cpair, df, lbl in zip(cs, [nbdata, bdata], labels):
        # Estimation bias plot
        ebax.plot(df.i0, 1 - df.k_slope, color=cpair[0], lw=1.5, label=lbl)
        # ebax.plot(df.i0, df.k_slope, 'o', color=cpair[0], markersize=3.0)
        # Error plot
        stax.plot(df.i0, df.std_dev, color=cpair[0], lw=1.5)
        # Plot the estimated squared error for both?
        stax2.plot(df.i0, np.sqrt(df.sqr_error), ls=':', color=cpair[0], lw=1.5)

    # Format axes
    xtit = r"Excitatory drive $I_0$"
    mod_ax_lims(ebax, xlims=(0, 0.21), ylims=(-0.05, 0.7), xticks=[0.01, 0.1, 0.2], yticks=[0, 0.3, 0.6],
                xbounds=(0.01, 0.2), ybounds=(0, 0.6), xformatter='%.2f', yformatter='%.1f', xlabel=xtit)
    ebax.legend(frameon=False)
    ebax.set_ylabel(r"Estimation bias" + '\n' + r"($1 - k_1$)", labelpad=0)
    ebax.plot([0.01, 0.2], [1, 1], ls=dashed, color='k', lw=0.5)
    # mod_ax_lims(stax, ylims=(-50, 1100), yticks=[0, 500, 1000],
    #             xbounds=(0.01, 0.2), ybounds=(0, 1000), xformatter='%.2f', yformatter='%d', xlabel=xtit,
    #             ylabel=r"Estimation error" + '\n' + r"(deg$^2$)")
    mod_ax_lims(stax, ylims=(-1, 32), yticks=[0, 15, 30],
                xbounds=(0.01, 0.2), ybounds=(0, 30), xformatter='%.2f', yformatter='%d', xlabel=xtit,
                ylabel=r"Std. deviation ($^\circ$)")
    mod_ax_lims(stax2, ylims=(-1, 32), yticks=[0, 15, 30],
                yformatter='%d', xbounds=(0.01, 0.2),
                ylabel=r"Estimation error")
    stax2.spines['right'].set_bounds((0, 30))
    stax2.tick_params(axis='y', direction='in')
    # plt.setp(stax.get_yticklabels(), rotation=45)
    # stax.tick_params(axis='y', which='major', pad=0)

    if save:
        directory = f"./results/paper_figs/fig5_bump_vs_nobump/"
        fig.set_tight_layout(False)
        filename = directory + f"fig_5_bump" + ('' if biased else '_supp')
        save_plot(fig, filename, overwrite=False, auto=True, saveformat=['png', 'pdf', 'svg'])

    return fig


# #### #### #### #### #### #### #
# Figure 5 (supp) of the paper  #
# #### #### #### #### #### #### #
def plot_fig5_bump_supp(save=False, **kwargs):
    """ Plot that compares bump vs. no-bump dynamics: estimation distribution, bias and PK slope.

    :param save: save the plot.
    :param kwargs: additional options.
    :return: figure
    """
    supp = kwargs.pop('supp', False)
    biased = not supp
    # Load the data for the first 3 panels
    nbdata = np.load("./results/variable_frames/simu_10000_nobump_sigma-0.15_i0-0.06_frames-8_t-3.0.npy",
                     allow_pickle=True)
    bref = ('biased_' if biased else '')
    bdata = np.load(f"./results/variable_frames/simu_10000_bump_{bref}sigma-0.15_i0-0.06_frames-8_t-2.0.npy",
                    allow_pickle=True)

    # #### #### #### #### #### ####
    # Prepare figure      #### ####
    # #### #### #### #### #### ####
    custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 200

    # Set up grid
    gs1 = gridspec.GridSpec(1, 2)  # Bump phases and distributions (as insets)
    gs2 = gridspec.GridSpec(1, 1)  # Estimation vs. avg. orientation
    gs3 = gridspec.GridSpec(1, 3)  # Estimation bias, std. and pk slope vs I0
    # Aux. grid for panel E (errors)
    gs4 = gridspec.GridSpec(1, 3)  # Estimation bias, std. and pk slope vs I0

    # Axes location variables
    margins = dict(left=0.07, right=0.65, top=0.98, bottom=0.6, hspace=0, wspace=0.1)
    gs1.update(**margins)
    margins.update(left=(margins['right'] + 0.13), right=0.98)
    gs2.update(**margins)
    margins.update(top=(margins['bottom'] - 0.15), left=0.07, bottom=0.12, wspace=0.8)
    gs3.update(**margins)
    margins.update(left=0.04, right=0.96)
    gs4.update(**margins)

    # Create figure and axes
    figsize = kwargs.pop('figsize', (3.9, 2.9))
    fig = plt.figure(figsize=figsize)  # Size is in inches (calculated for a US Letter size in inkscape)
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(False)

    # Phases axes
    phax = [fig.add_subplot(gs1[0, k]) for k in range(2)]
    # Distribution inset axes
    dbax = [inset_axes(phax[k], width="18%", height="100%", loc='center right', borderpad=0) for k in range(2)]
    # Estimation ax
    esax = fig.add_subplot(gs2[0, 0])
    # estimation bias ax
    ebax = fig.add_subplot(gs3[0, 0])
    # std ax
    stax = fig.add_subplot(gs4[0, 1], sharex=ebax)
    # PK slope ax
    pkax = fig.add_subplot(gs3[0, 2], sharex=ebax)

    # Delete some axes
    mod_axes(fig.get_axes())

    # Second axis for the squared error:
    stax2 = stax.twinx()
    mod_axes(stax2, x='top', y='left')

    axes = fig.get_axes()
    for axb in axes:
        axb.patch.set_alpha(0.0)
        axb.grid(False)

    # Some style elements ((red, orange), (blue, turquoise))
    # cs = [['#e6443f', '#d8944a'], ['#3586cf', '#47cfcd']]
    # Some style elements ((red, orange), (purple, pink))
    cs = [['#e6443f', '#d8944a'], ['#9467bd', '#e377c2']]
    dashed = (0, (10, 3))

    # Actual plotting starts here
    # #### #### #### #### #### ##
    # Phase evolutions
    xtit = "Time (s)"
    ytit = r"$\psi (t)$ ($^\circ$)"

    angles = kwargs.get('angles', [0, 45])
    tol_b = kwargs.get('ang_tol_b', 2)
    tol_nb = kwargs.get('ang_tol_nb', 2)
    double = kwargs.get('double', False)
    if not supp:
        labels = [r"\bf \sffamily No initial bump", r"\bf \sffamily Initial bump"]
    else:
        labels = [r"\bf \sffamily No initial bump", r"\bf \sffamily \noindent Unbiased\\initial bump"]

    for k, (data, ax, dax, cpair, tol) in enumerate(zip([nbdata, bdata], phax, dbax, cs, [tol_nb, tol_b])):
        df = data['data']
        df_chosen = df.loc[df.chosen != -1].copy().reset_index(drop=True)  # Select trials with saved activity
        rates = data['rates']
        t = np.arange(0, 200) * 0.01
        ax.plot([t[0], t[-1]], [0, 0], ls=dashed, lw=0.5, zorder=1, color='k')  # Reference at zero degrees

        for th, c in zip(angles, cpair):  # Select the data whose average orientation ie equal to th
            # For the histogram
            hist_sel = df.loc[(np.abs(df.average_circ - th) < tol) | (np.abs(df.average_circ + th) < tol)].copy()
            # Select trials with average orientation th that have an activity representation
            sel1 = df_chosen.loc[np.abs(df_chosen.average_circ - th) < tol].copy()  # th sign(positive)
            _, ph1, _, _ = get_phases_and_amplitudes_auto(rates[:, sel1.index.to_list()])  # Obtain phase
            ph1 = np.rad2deg(ph1)
            if len(ph1) > 200:  # Take only the first 2 seconds of activity
                ph1 = ph1[:200]

            # Plot the phase evolution
            ax.plot(t, ph1, color=c, alpha=0.4)
            if th != 0:
                sel2 = df_chosen.loc[np.abs(df_chosen.average_circ + th) < tol].copy()  # th sign(negative)
                _, ph2, _, _ = get_phases_and_amplitudes_auto(rates[:, sel2.index.to_list()])
                ph2 = -np.rad2deg(ph2)
                if len(ph2) > 200:  # Take only the first 2 seconds of activity
                    ph2 = ph2[:200]
                hist_sel.loc[hist_sel.category == -th, 'estim'] = -hist_sel.loc[hist_sel.category == -th, 'estim']
                if double:
                    ax.plot(t, ph2, color=c, alpha=0.4)
            # Plot the distribution
            sns.kdeplot(hist_sel.estim.to_numpy(), shade=True, ax=dax, color=c, vertical=True, lw=0.5)
            avg = circmean(hist_sel.estim, low=-180, high=180)
            dax.plot([0.075], [avg], '<', color=c, markersize=4.0, clip_on=False)
            ax.plot([2.06], [th], marker='_', color=c, markersize=4.0, markeredgewidth=1.0)

        # Modify axes style
        mod_ax_lims(ax, xlims=(-0.05, 2.55), ylims=(-95, 95), xticks=[0, 1, 2], yticks=[-90, -45, 0, 45, 90],
                    xbounds=(0, 2), ybounds=(-90, 90), xformatter='%d', yformatter='%d', xlabel=xtit)
        ax.text(2.0, -85, labels[k], ha='right', va='bottom', fontsize=10, color=cpair[0])
        if k == 0:
            # ax.set_xlabel(xtit, position=(0, 0), ha='left')
            ax.set_ylabel(ytit)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

        mod_ax_lims(dax, ylims=(-95, 95), xlims=(-0.01, 0.08), xticks=[], yticks=[])
        dax.spines['bottom'].set_visible(False)
        dax.spines['left'].set_visible(False)
        plt.setp(dax.get_yticklabels(), visible=False)

    # Estimation plot
    el = kwargs.pop('eslim', 60)
    xtit = kwargs.pop('xlabel', r"Avg. direction $(^\circ)$")
    ytit = kwargs.pop('ytit', r"Estimation $\psi$ $(^\circ)$")

    es_nb = compute_estimation(nbdata['data'])
    es_b1 = compute_estimation(bdata['data'], ylabel='ph4')
    es_b2 = compute_estimation(bdata['data'])

    esax.plot(es_nb[0], es_nb[1], color=cs[0][0], label='No bump', lw=1.5)
    esax.plot(es_b1[0], es_b1[1], color=cs[1][0], label="Bump, $t=1$s", ls='--', lw=1.5)
    esax.plot(es_b2[0], es_b2[1], color=cs[1][0], label="Bump, $t=2$s", lw=1.5)
    esax.plot([-el, el], [-el, el], ls=dashed, color='black', lw=1.0)

    esax.legend(frameon=False, fontsize=8, loc='lower right')

    # Format ax
    ticks = [-el, -int(el / 2), 0, int(el / 2), el] if el % 2 == 0 else [-el, 0, el]
    mod_ax_lims(esax, xlims=(-el - 5, el + 5), ylims=(-el - 5, el + 5), xticks=ticks, yticks=ticks,
                xbounds=(-el, el), ybounds=(-el, el), xformatter='%d', yformatter='%d', xlabel=xtit, ylabel=ytit)

    # Load data for the rest of the plot
    stats = pd.read_csv('results/integration_stats_frames-8_t-2.0_new.csv')
    nbdata = stats.loc[(stats.sigma == 0.15) & (stats.bump == False) & (stats.frames == 8) & (stats.t == 2.0)].copy()
    nbdata.sort_values('i0', inplace=True)
    bdata = stats.loc[(stats.sigma == 0.15) & (stats.bump == True) & (stats.frames == 8) & (stats.t == 2.0) &
                      (stats.biased == biased)].copy()
    bdata.sort_values('i0', inplace=True)
    if not biased:
        bdata.ppk_slope = bdata.ppk_slope_u

    labels = ['No bump', 'Bump']
    for cpair, df, lbl in zip(cs, [nbdata, bdata], labels):
        # Estimation bias plot
        ebax.plot(df.i0, 1 - df.k_slope, color=cpair[0], lw=1.5, label=lbl)
        # ebax.plot(df.i0, df.k_slope, 'o', color=cpair[0], markersize=3.0)
        # Error plot
        stax.plot(df.i0, df.std_dev, color=cpair[0], lw=1.5)
        # Plot the estimated squared error for both?
        stax2.plot(df.i0, np.sqrt(df.sqr_error), ls=':', color=cpair[0], lw=1.5)
        # PK slope
        pkax.plot(df.i0, df.ppk_slope, color=cpair[0], lw=1.5)
        # pkax.plot(df.i0, df.ppk_slope, 'o', color=cpair[0], markersize=3.0)

    # Format axes
    xtit = r"Excitatory drive $I_0$"
    mod_ax_lims(ebax, xlims=(0, 0.21), ylims=(-0.05, 0.7), xticks=[0.01, 0.1, 0.2], yticks=[0, 0.3, 0.6],
                xbounds=(0.01, 0.2), ybounds=(0, 0.6), xformatter='%.2f', yformatter='%.1f', xlabel=xtit)
    ebax.legend(frameon=False)
    ebax.set_ylabel(r"Estimation bias" + '\n' + r"($1 - k_1$)", labelpad=0)
    ebax.plot([0.01, 0.2], [1, 1], ls=dashed, color='k', lw=0.5)
    # mod_ax_lims(stax, ylims=(-50, 1100), yticks=[0, 500, 1000],
    #             xbounds=(0.01, 0.2), ybounds=(0, 1000), xformatter='%.2f', yformatter='%d', xlabel=xtit,
    #             ylabel=r"Estimation error" + '\n' + r"(deg$^2$)")
    mod_ax_lims(stax, ylims=(-1, 32), yticks=[0, 15, 30],
                xbounds=(0.01, 0.2), ybounds=(0, 30), xformatter='%.2f', yformatter='%d', xlabel=xtit,
                ylabel=r"Std. deviation ($^\circ$)")
    mod_ax_lims(stax2, ylims=(-1, 32), yticks=[0, 15, 30],
                yformatter='%d', xbounds=(0.01, 0.2),
                ylabel=r"Estimation error")
    stax2.spines['right'].set_bounds((0, 30))
    stax2.tick_params(axis='y', direction='in')
    # plt.setp(stax.get_yticklabels(), rotation=45)
    # stax.tick_params(axis='y', which='major', pad=0)
    mod_ax_lims(pkax, ylims=(-0.6, 0.6), yticks=[-0.5, 0, 0.5],
                xbounds=(0.01, 0.2), ybounds=(-0.5, 0.5), xformatter='%.2f', yformatter='%.1f', xlabel=xtit,
                ylabel=r"PK slope")
    pkax.plot([0.01, 0.2], [0, 0], ls=dashed, color='k', lw=0.5)

    if save:
        directory = f"./results/paper_figs/fig5_bump_vs_nobump/"
        fig.set_tight_layout(False)
        filename = directory + f"fig_5_bump" + ('' if biased else '_supp')
        save_plot(fig, filename, overwrite=False, auto=True, saveformat=['png', 'pdf', 'svg'])

    return fig


# #### #### #### #### #### #### ###
# Figure behavior 1 of the paper ##
# #### #### #### #### #### #### ###
def plot_fig_b1(i0s=(0.08, 0.05, 0.02), save=False, **kwargs):
    """ Behavioral vs. simulation comparison: temporal integration. Psychophysical kernels for different subject
    groups that show either Primacy, Recency or Uniform. The psychophysical kernels of the simulation data are
    computed for different values of the general excitability $I_0$.

    :param i0s:
    :param save:
    :param kwargs:
    :return:
    """
    # Load simulation data
    sdata = []
    for i0 in i0s:
        datafile = f"results/simu_10000_nobump_sigma-0.15_i0-{i0:0.2f}"
        logging.info(f"Loading data from '{datafile}'...")
        sdata.append(pd.read_csv(datafile + '.csv'))

    # Load behavioral data
    # data = pd.read_csv('../behavioral_data_summerfield/pythonic_v2/PKs.txt')
    data = pd.read_csv('./data_summerfield_klaus/PKs.txt')
    betas_lbls = ['bw_%d' % k for k in range(1, 9)]

    # Filter the data, take only data from databases 1-4, exclude poor performance (1 subject)
    data = data.loc[data.dataset.isin([1, 2, 3, 4]) & (data.included != 'poor_performance')].copy()

    primacy = data.loc[data['PK_type'] == 'primacy']
    recency = data.loc[data['PK_type'] == 'recency']
    uniform = data.loc[data['PK_type'] == 'uniform']

    nsubj = len(data)
    nprimacy, nrecency, nuniform = (len(primacy), len(recency), len(uniform))

    # For debugging:
    logging.debug(f"Head of table:\n{sdata[0].head()}")

    # #### #### #### #### #### ####
    # Prepare figure      #### ####
    # #### #### #### #### #### ####
    custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 300

    # Set up grid
    gs1 = gridspec.GridSpec(2, 4)
    gs2 = gridspec.GridSpec(2, 4)

    # Axes location variables
    margins = dict(left=0.10, right=0.92, top=0.98, bottom=0.1, hspace=0.5, wspace=0.1)
    gs1.update(**margins)
    margins.update(right=0.98, wspace=0.0, left=0.3)
    gs2.update(**margins)

    # Create figure and axes
    figsize = kwargs.pop('figsize', (5.2, 2.9))
    fig = plt.figure(figsize=figsize)  # Size is in inches (calculated for a US Letter size in inkscape)
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(False)

    # behavioral data axes
    taxs = [fig.add_subplot(gs1[0, k]) for k in range(3)]
    # Histogram ax
    hax = fig.add_subplot(gs2[0, -1])
    # simulation data axes
    baxs = [fig.add_subplot(gs1[1, k], sharex=taxs[0]) for k in range(3)]

    axes = fig.get_axes()
    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.grid(False)

    # Delete some axes
    mod_axes(axes)
    for ax in (taxs[1:] + baxs[1:]):
        ax.spines['left'].set_visible(False)
    colors = ['blue', 'black', 'red']

    # Actual plotting starts here
    # #### #### #### #### #### ##
    # Behavioral data
    kernels = [r"\textbf{Primacy}: %d/%d" % (nprimacy, nsubj), r"\textbf{Uniform}: %d/%d" % (nuniform, nsubj),
               r"\textbf{Recency}: %d/%d" % (nrecency, nsubj)]
    xticklabels = list([""] * 8)
    xticklabels[0] = 1
    xticklabels[-1] = 8
    for k, (df, title, ax, c) in enumerate(zip([primacy, uniform, recency], kernels, taxs, colors)):
        ax.plot(range(1, 9), df[betas_lbls].T, lw=0.1, alpha=0.5, color='k')
        ax, pe, be = plot_error_filled(np.arange(1, 9), df[betas_lbls].mean(axis=0), df[betas_lbls].sem(axis=0),
                                       color=c, ax=ax, alpha=0.2)
        pe.set_label(title)
        ax.legend(frameon=False, fontsize=6)

        mod_ax_lims(ax, xlims=(0.5, 8.1), ylims=(-0.01, 1.0), xticks=list(range(1, 9)), yticks=[0, 0.5, 1.0],
                    ybounds=(0, 1), xbounds=(1, 8), xlabel='', xformatter='%d')

        if k != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_yticks([])
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.set_ylabel('Stimulus impact')

        ax.set_xticklabels(xticklabels)
        # Histogram
        hax.hist(df['PK_slope'], color=c, alpha=0.5, bins=np.arange(-0.2, 0.46, 0.04))
        hax.hist(df['PK_slope'], color=c, bins=np.arange(-0.2, 0.46, 0.04), histtype='step', lw=0.5)

        hax.plot(np.median(df['PK_slope']), [11], 'v', color=c, ms=3)

    hax.plot([0, 0], [0, 10], '--', color='k', lw=0.5)
    mod_ax_lims(hax, xlims=(-0.55, 0.55), xticks=(-0.5, 0, 0.5), xformatter='%.1f', xbounds=(-0.5, 0.5),
                ylims=(0, 12), ybounds=(0, 10), yticks=[0, 5, 10], ylabel='Subjects', xlabel='PK slope',
                yformatter='%d')

    # Psychophysical kernel (binary discrimination)
    xlabels = [f"x{k}" for k in range(1, 9)]
    for k, (df, ax, c, title) in enumerate(zip(sdata, baxs, colors, i0s)):
        # Compute ppk
        df['binchoice_old'] = (df.estim >= 0) * 2 - 1
        # TODO: change this. Wrong!!!
        m_r, mf_r, df = log_reg(df, xlabels, ylabel='binchoice_old')
        # Plot
        pkl = kwargs.pop('pkk_lims', (0, 0.06))
        factor = 1.0 / pkl[1]
        ax, pe, be = plot_error_filled(np.arange(1, 9), factor * mf_r.beta, factor * mf_r.errors, color=c, ax=ax,
                                       alpha=0.2)
        title = fr"$I_0 = {title}$"
        pe.set_label(title)
        ax.legend(frameon=False, fontsize=8, loc='upper left')

        mod_ax_lims(ax, xlims=(0.5, 8.1), ylims=(-0.01, 1.0), xticks=list(range(1, 9)), yticks=[0, 0.5, 1.0],
                    ybounds=(0, 1), xbounds=(1, 8), xformatter='%d')

        ax.set_xticklabels(xticklabels)

        if k != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_yticks([])
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.set_ylabel('Stimulus impact')
            ax.set_xlabel('Frame number')

    if save:
        directory = 'results/paper_figs/fig_b1/'
        fig.set_tight_layout(False)
        filename = directory + f"fig4"
        save_plot(fig, filename, overwrite=False, auto=True, saveformat=['png', 'pdf', 'svg'])

    return fig


# #### #### #### #### #### #### ###
# Figure behavior 2 of the paper ##
# #### #### #### #### #### #### ###
def plot_fig_b2(save=False, **kwargs):
    """ Behavioral vs. simulation comparison: focused vs. divided attention. Psychophysical kernels for different
    behavioral conditions. Possible mechanism that could lead to those results in our model.

    :param save:
    :param kwargs:
    :return:
    """
    # TODO: write the function using ipython notebook
    pass


# #### #### #### #### #### #### ###
# Figure behavior 3 of the paper ##
# #### #### #### #### #### #### ###
def plot_fig_b3(save=False, **kwargs):
    """ Behavioral vs. simulation comparison: response times and their correlation with integration regime.

    :param save:
    :param kwargs:
    :return:
    """
    # TODO: building... best strategy is to do a 4x3 plot
    # Load the behavioral data
    bdata = pd.read_csv('./data_summerfield_klaus/PKs_25_05_2021.txt')
    # Filter the data and select the response times
    rtdata = bdata.loc[bdata.dataset.isin([1, 2, 3]) & (bdata.included == 'yes')].copy()
    xlabels = [f"x_{k}" for k in range(1, 6)]
    cor_lbl = [f"c_corr_{k}" for k in range(1, 6)]
    inc_lbl = [f"c_incorr_{k}" for k in range(1, 6)]
    rt_corr = rtdata[cor_lbl].mean(axis=0) * 1000  # In milliseconds
    rt_inco = rtdata[inc_lbl].mean(axis=0) * 1000
    avg = lambda x: np.mean(x, axis=0)
    er_corr = np.std(bootstrap(rtdata[cor_lbl].to_numpy(), 10000, bootfunc=avg), axis=0) * 1000
    er_inco = np.std(bootstrap(rtdata[inc_lbl].to_numpy(), 10000, bootfunc=avg), axis=0) * 1000

    # Mean reaction times (z-scored for each dataset) vs. PK slopes
    rt_avg_normed = []
    for dataset in [1, 2, 3]:
        rt_sel = rtdata.loc[rtdata.dataset == dataset, 'c_mean'].copy()
        z_scored = (rt_sel - rt_sel.mean()) / rt_sel.std()
        rt_avg_normed.extend(z_scored.to_numpy())
    rt_avg_normed = np.array(rt_avg_normed)

    # Load the model results
    u_dc = np.load('data_summerfield_klaus/rt_v3/simu_10000_nobump_sigma-0.40_i0-0.390_frames-8_t-3.0.npy',
                   allow_pickle=True)['ddata']
    u_df = np.load('data_summerfield_klaus/rt_v3/simu_10000_nobump_sigma-0.40_i0-0.390_frames-8_t-3.0.npy',
                   allow_pickle=True)['data']
    u_rb = np.load('data_summerfield_klaus/rt_v3/simu_10000_nobump_sigma-0.40_i0-0.390_frames-8_t-3.0.npy',
                   allow_pickle=True)['rates']

    # #### #### #### #### #### ####
    # Prepare figure      #### ####
    # #### #### #### #### #### ####
    custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 300

    # Set up grid
    gs1 = gridspec.GridSpec(4, 3)
    gs2 = gridspec.GridSpec(1, 2)
    gs3 = gridspec.GridSpec(1, 2)
    gs4 = gridspec.GridSpec(4, 3)

    # Axes location variables
    margins = dict(left=0.08, right=0.99, top=0.98, bottom=0.07, hspace=0.5, wspace=0.6)
    gs1.update(**margins)
    gs4.update(**margins)
    margins.update(left=0.425, right=0.98, top=0.71, bottom=0.58, hspace=0.0, wspace=0.2)
    gs2.update(**margins)
    margins.update(left=0.425, right=0.98, top=0.49, bottom=0.35, hspace=0.0, wspace=0.2)
    gs3.update(**margins)

    # Create figure and axes
    figsize = kwargs.pop('figsize', (5, 5.63))
    fig = plt.figure(figsize=figsize)  # Size is in inches (calculated for a US Letter size in inkscape)
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(False)

    # behavioral data axes
    taxs = [fig.add_subplot(gs1[0, k]) for k in range(3)]
    # decision circuit axes
    daxs = [fig.add_subplot(gs2[0, k]) for k in range(2)]
    # integration circuit axes
    iaxs = [fig.add_subplot(gs3[0, k]) for k in range(2)]
    # response times statistics axes
    rtax1 = fig.add_subplot(gs4[3, 1])
    rtax2 = fig.add_subplot(gs4[3, 2])

    axes = fig.get_axes()
    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.grid(False)

    mod_axes(axes)

    # fig_grid(fig)
    # fig_grid(fig, orientation='v', shift=0.425)
    # Actual plotting starts here
    # #### #### #### #### #### ##
    # Behavioral data
    # mean RT vs. category level avg.
    ax = taxs[0]
    for rt, ert, ls, c in zip([rt_corr, rt_inco], [er_corr, er_inco], ['-', '--'], ['k', 'k']):
        _, _, eb = plot_error_filled(rtdata[xlabels].mean(axis=0), rt, ert, ls=ls, color=c, ax=ax, alpha=0.2)
        eb.set_linewidth(0.0)
    ax.plot([0, 0], [-40, 120], '--', color='k', lw=0.5)
    ylabel = r"Avg. RT $RT - \left<RT\right>$ (ms)"
    mod_ax_lims(ax, xlims=(-0.42, 0.4), xbounds=(-0.4, 0.4), xticks=(-0.4, -0.2, 0.0, 0.2, 0.4),
                xlabel="Cat. level avg.", ylims=(-50, 120), ybounds=(-40, 120), yticks=(-40, 0, 40, 80, 120),
                ylabel=r"Avg. RT (ms)", xformatter='%.1f', yformatter='%d')

    # RT vs. PK slope
    ax = taxs[1]
    ax.scatter(rtdata['PK_slope'], rt_avg_normed, color='k')
    lin_reg = linregress(rtdata['PK_slope'], rt_avg_normed)
    lin_x = np.arange(-0.15, 0.41, 0.01)
    ax.plot(lin_x, lin_x * lin_reg.slope + lin_reg.intercept,
            label=f"$r={lin_reg.rvalue:.2f}$\n$p={lin_reg.pvalue:.3f}$")
    ax.legend(frameon=False, fontsize=8)
    mod_ax_lims(ax, xlims=(-0.23, 0.45), xbounds=(-0.2, 0.4), xticks=(-0.2, 0, 0.2, 0.4), xlabel="PK slope",
                ylims=(-2.3, 3), ybounds=(-2, 3), yticks=(-2, -1, 0, 1, 2, 3), ylabel="normalized RT",
                xformatter='%.1f', yformatter='%d')
    ax.set_yticklabels([-2, "", 0, "", 2, ""])

    # % correct vs PK slope
    ax = taxs[2]
    ax.scatter(rtdata['PK_slope'], rtdata['p_corr'], color='k')
    lin_reg = linregress(rtdata['PK_slope'], rtdata['p_corr'])
    ax.plot(lin_x, lin_x * lin_reg.slope + lin_reg.intercept,
            label=f"$r={lin_reg.rvalue:.2f}$\n$p={lin_reg.pvalue:.3f}$")
    ax.legend(frameon=False, fontsize=8)
    mod_ax_lims(ax, xlims=(-0.23, 0.45), xbounds=(-0.2, 0.4), xticks=(-0.2, 0, 0.2, 0.4), xlabel="PK slope",
                ylims=(0.57, 0.9), ybounds=(0.6, 0.9), yticks=(0.6, 0.7, 0.8, 0.9), ylabel=r"Avg. \% correct",
                xformatter='%.1f', yformatter='%.1f')

    # Model results
    # Decision circuit firing rate (incorrect, correct)
    ax = daxs[0]
    u_ss = u_df.loc[u_df.chosen != -1].copy()
    tpoints = np.arange(0.0, 3.0, 0.01)
    selected_cat = 15
    mask = ((u_ss.binrt != -1) & (u_ss.category == selected_cat))

    # Left vs right (one stimuli, correct and incorrect: ~20)
    u_correct = (mask & (np.sign(u_ss.binchoice) == np.sign(u_ss.average_circ)))
    u_incorrect = (mask & (np.sign(u_ss.binchoice) != np.sign(u_ss.average_circ)))

    r = u_dc['d12']
    corr_right = r[:, 1, u_correct]
    corr_left = r[:, 0, u_correct]
    incorr_right = r[:, 1, u_incorrect]
    incorr_left = r[:, 0, u_incorrect]
    colors = ["#5159b6", "#c631bb"]

    for right, left, ls in zip([corr_right, incorr_right], [corr_left, incorr_left], ['-', '--']):
        _, lp, eb = plot_error_filled(tpoints[190:-1], right[190:-1].mean(axis=-1), right[190:-1].std(axis=-1), ls=ls,
                                      color=colors[0], ax=ax, alpha=0.0)
        eb.set_linewidth(0.0)
        _, lp, eb = plot_error_filled(tpoints[190:-1], left[190:-1].mean(axis=-1), left[190:-1].std(axis=-1), ls=ls,
                                      color=colors[1], ax=ax, alpha=0.0)
        eb.set_linewidth(0.0)
        # lp.set_alpha(0.4)

    ax.plot([1.9, 2.6], [50, 50], '--', color='k', lw=0.5, zorder=1)
    rect = Rectangle((2.0, 0), width=0.5, height=1.0, transform=ax.get_xaxis_transform(), color='orange', alpha=0.2,
                     linewidth=0.0, zorder=0)
    ax.add_patch(rect)
    mod_ax_lims(ax, xlims=(1.85, 2.6), xbounds=(1.9, 2.6), xticks=(2.0, 2.5), xlabel="Time (s)",
                ylims=(0, 80), ybounds=(0, 80), yticks=(0, 40, 80), ylabel=r"Firing rate (Hz)",
                xformatter='%.1f', yformatter='%d')

    input_ax1 = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(2.4, 15, 0.20, 30),
                           bbox_transform=ax.transData, borderpad=0)
    mod_axes(input_ax1)
    input_ax1.patch.set_alpha(0.0)
    input_ax1.grid(False)

    inp = u_dc['lri']
    corr_ri = inp[:, 1, u_correct]
    corr_li = inp[:, 0, u_correct]
    incorr_ri = inp[:, 1, u_incorrect]
    incorr_li = inp[:, 0, u_incorrect]

    bar_width = 0.25
    input_ax1.bar([0], [corr_li.mean()], bar_width, color=colors[1], ls='-')
    input_ax1.bar([0 + bar_width*1.4], [corr_ri.mean()], bar_width, color=colors[0], ls='-')
    input_ax1.bar([1], [incorr_li.mean()], bar_width, color=colors[1], ls='--')
    input_ax1.bar([1 + bar_width*1.4], [incorr_ri.mean()], bar_width, color=colors[0], ls='--')
    input_ax1.set_xticks([(bar_width * 1.4)/2, 1 + (bar_width * 1.4)/2])
    input_ax1.set_xticklabels(["Corr", "Incorr"])
    input_ax1.tick_params(axis='x', which='major', pad=1)
    # input_ax1.set_xticks([])
    plt.setp(input_ax1.get_xticklabels(), fontsize=5)

    mod_ax_lims(input_ax1, yticks=())
    input_ax1.set_ylabel('Input', fontsize=6, labelpad=2)

    # Integration circuit firing rate, incorrect and correct
    # (mean firing rate at the end of the trial for 2 different cat. levels)
    ax = iaxs[0]
    theta = np.arange(200) / 200 * 360 - 180

    corr_r = get_phases_and_amplitudes_auto(u_rb[180:200, u_correct], aligned_profiles=False)[0].mean(axis=(0, 1))
    incorr_r = get_phases_and_amplitudes_auto(u_rb[180:200, u_incorrect], aligned_profiles=False)[0].mean(axis=(0, 1))
    # Smooth the firing rate for better representation
    corr_r = sp.ndimage.filters.gaussian_filter(corr_r, 5, mode='wrap')
    incorr_r = sp.ndimage.filters.gaussian_filter(incorr_r, 5, mode='wrap')

    # # Set the phase of the last frame (200)
    # # corr_ph = np.rad2deg(get_phases_and_amplitudes_auto(u_rb[200, u_correct][0])[1])
    # corr_ph = selected_cat
    # corr_idx = np.abs(theta - corr_ph).argmin()
    # corr_r = np.roll(corr_r, corr_idx + 200 // 2)
    # incorr_ph = np.rad2deg(get_phases_and_amplitudes_auto(u_rb[200, u_incorrect][0])[1])
    # incorr_idx = np.abs(theta - incorr_ph).argmin()
    # incorr_r = np.roll(incorr_r, incorr_idx + 200 // 2)

    # ax.plot(theta, u_rb[200, u_correct][0], '-', color='k', lw=0.2)
    # ax.plot(theta, u_rb[200, u_incorrect][0], '-', color='gray', lw=0.2)

    ax.plot(theta, corr_r, '-', color='k', label='Correct')
    ax.plot(theta, incorr_r, '--', color='k', label='Incorrect')

    ax.legend(frameon=False, fontsize=8)
    ax.plot([0, 0], [0, 50], '--', color='k', lw=0.5, zorder=1)
    mod_ax_lims(ax, xlims=(-185, 180), xbounds=(-180, 180), xticks=(-180, -90, 0, 90, 180),
                xlabel=r"$\theta$ ($^\circ$)", ylabel="Firing rate (Hz)", xformatter='%d', yformatter='%d',
                ylims=(-2, 50), ybounds=(0, 50), yticks=(0, 25, 50))

    # ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
    # Decision circuit firing rate (i0 low, i0 high) (and inset)
    ax = daxs[1]
    axb = iaxs[1]
    input_ax1 = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(2.4, 15, 0.20, 30),
                           bbox_transform=ax.transData, borderpad=0)
    mod_axes(input_ax1)
    input_ax1.patch.set_alpha(0.0)
    input_ax1.grid(False)

    i0s = [0.46, 0.39]
    alphas = [1.0, 0.5]

    for k, i0 in enumerate(i0s):
        # Load the model results
        data = np.load(f"data_summerfield_klaus/rt_v3/simu_10000_nobump_sigma-0.40_i0-{i0:.3f}_frames-8_t-3.0.npy",
                       allow_pickle=True)
        df = data['data'].loc[data['data'].chosen != -1].copy()

        # Select only correct trials
        mask = ((df.binrt != -1) & (df.category == selected_cat) & (np.sign(df.binchoice) == np.sign(df.average_circ)))
        corr_right = data['ddata']['d12'][:, 1, mask]
        corr_left = data['ddata']['d12'][:, 0, mask]
        # Input
        right_i = data['ddata']['lri'][:, 1, mask]
        left_i = data['ddata']['lri'][:, 0, mask]

        # Bump
        bump = get_phases_and_amplitudes_auto(data['rates'][180:200, mask], aligned_profiles=False)[0].mean(axis=(0, 1))
        bump = sp.ndimage.filters.gaussian_filter(bump, 5, mode='wrap')
        # ph = np.rad2deg(get_phases_and_amplitudes_auto(data['rates'][200, mask][0])[1])
        # ph = selected_cat
        # idx = np.abs(theta - ph).argmin()
        # bump = np.roll(bump, idx + 200 // 2)

        # Plot firing rates
        _, lp, eb = plot_error_filled(tpoints[190:-1], corr_right[190:-1].mean(axis=-1),
                                      corr_right[190:-1].std(axis=-1), ls='-', color=colors[0], ax=ax,
                                      alpha=0.0)
        eb.set_linewidth(0.0)
        lp.set_alpha(alphas[k])
        _, lp, eb = plot_error_filled(tpoints[190:-1], corr_left[190:-1].mean(axis=-1),
                                      corr_left[190:-1].std(axis=-1), ls='-', color=colors[1], ax=ax,
                                      alpha=0.0)
        eb.set_linewidth(0.0)
        lp.set_alpha(alphas[k])

        # Plot input bars
        bar_width = 0.25
        input_ax1.bar([k], [left_i.mean()], bar_width, color=colors[1], ls='-', alpha=alphas[k])
        input_ax1.bar([k + bar_width*1.4], [right_i.mean()], bar_width, color=colors[0], ls='-', alpha=alphas[k])

        # Plot bump
        axb.plot(theta, bump, ls='-', color='k', label=f"$I_0 = {i0:.2f}$", alpha=alphas[k])

    ax.plot([1.9, 2.6], [50, 50], '--', color='k', lw=0.5, zorder=1)
    rect = Rectangle((2.0, 0), width=0.5, height=1.0, transform=ax.get_xaxis_transform(), color='orange', alpha=0.2,
                     linewidth=0.0, zorder=0)
    ax.add_patch(rect)
    mod_ax_lims(ax, xlims=(1.85, 2.6), xbounds=(1.9, 2.6), xticks=(2.0, 2.5), xlabel="Time (s)",
                ylims=(0, 80), ybounds=(0, 80), yticks=(),
                xformatter='%.1f')
    ax.spines['left'].set_visible(False)
    plt.setp(ax.get_yticklabels(), visible=False)

    input_ax1.set_xticks([(bar_width * 1.4)/2, 1 + (bar_width * 1.4)/2])
    input_ax1.set_xticklabels(i0s)
    input_ax1.tick_params(axis='x', which='major', pad=1)
    plt.setp(input_ax1.get_xticklabels(), fontsize=5)

    mod_ax_lims(input_ax1, yticks=())
    input_ax1.set_ylabel('Input', fontsize=6, labelpad=2)

    axb.plot([0, 0], [0, 50], '--', color='k', lw=0.5, zorder=1)
    axb.legend(frameon=False, fontsize=8)
    mod_ax_lims(axb, xlims=(-185, 180), xbounds=(-180, 180), xticks=(-180, -90, 0, 90, 180),
                xlabel=r"$\theta$ ($^\circ$)", xformatter='%d',
                ylims=(-2, 50), ybounds=(0, 50), yticks=())
    axb.spines['left'].set_visible(False)
    plt.setp(axb.get_yticklabels(), visible=False)

    # Response times vs. cat level avg
    # bin the results according to the category level average
    # We define the categories for the data points and the model points/line
    # def compute_error(dat):
    #     avg = lambda x: np.mean(x, axis=0)
    #     return np.std(bootstrap(dat.to_numpy(), 10000, bootfunc=avg))

    # Load all the databases, filter the data and select the necessary columns, additionally, compute the PK slope
    i0s = np.arange(0.33, 0.47, 0.01)
    data = []
    compute = kwargs.get('compute_pk', False)
    xlabels = [f"x{k+1}" for k in range(8)]
    if compute:
        performance = []
        betas = []
        pk_slopes = []
        rts = []
    else:
        performance, betas, pk_slopes, rts = np.load('data_summerfield_klaus/rt_v3/pk_slopes.npy', allow_pickle=True)

    for i0 in i0s:
        df = pd.read_csv(f"data_summerfield_klaus/rt_v3/simu_10000_nobump_sigma-0.40_i0-{i0:.3f}_frames-8_t-3.0.csv")
        # Compute the performance, the PK and the PK slope before filtering the data
        if compute:
            p = np.sum(np.sign(df.binchoice) == np.sign(df.average_circ)) / 10000 * 100
            b, _, _ = compute_circ_reg_ml(df, xlabels, 'estim')
            pks = compute_ppk_slope(b)

            performance.append(p)
            betas.append(b)
            pk_slopes.append(pks[0])

        # Filter the data
        df = df.loc[(df.binrt != -1)].copy()
        if compute:
            rts.append(df['binrt'].mean())

        # Normalize the response times
        df['binrt'] = df['binrt'] - df['binrt'].mean()

        # Select only the necessary columns
        df = df[['category', 'binrt', 'average_circ', 'binchoice']]
        data.append(df)

    if compute and save:
        np.save('data_summerfield_klaus/rt_v3/pk_slopes', [performance, betas, pk_slopes, rts], allow_pickle=True)

    data = pd.concat(data, ignore_index=True)

    # Divide between correct and incorrect
    correct = data.loc[(np.sign(data.binchoice) == np.sign(data.average_circ))].copy()
    incorrect = data.loc[(np.sign(data.binchoice) != np.sign(data.average_circ))].copy()
    # Group the data into equally populated bins
    correct['bin'] = equalbins(correct.average_circ, 15)
    incorrect['bin'] = equalbins(incorrect.average_circ, 15)

    # Group the data by the binning
    ylabel = 'binrt'
    group_label = 'bin'
    gr1 = correct.groupby(group_label)
    x_avg_c = gr1.average_circ.mean()
    y_avg_c = gr1[ylabel].mean()
    y_std_c = gr1[ylabel].sem()
    gr2 = incorrect.groupby(group_label)
    x_avg_i = gr2.average_circ.apply(circmean, low=-180, high=180)
    y_avg_i = gr2[ylabel].mean()
    y_std_i = gr2[ylabel].sem()

    ax = rtax1
    for x, rt, ert, ls, c in zip([x_avg_c, x_avg_i], [y_avg_c, y_avg_i], [y_std_c, y_std_i], ['-', '--'], ['k', 'k']):
        _, _, eb = plot_error_filled(x, rt*1000, ert*1000, ls=ls, color=c, ax=ax, alpha=0.2)
        eb.set_linewidth(0.0)
    ax.plot([0, 0], [-40, 120], '--', color='k', lw=0.5)
    # ylabel = r"Avg. RT $RT - \left<RT\right>$ (ms)"
    mod_ax_lims(ax, xlims=(-37, 35), xbounds=(-35, 35), xticks=(-35, -15, 0.0, 15, 35),
                xlabel="Avg. orientation ($^\circ$)", ylims=(-50, 120), ybounds=(-40, 120), yticks=(-40, 0, 40, 80, 120),
                ylabel=r"Avg. $\overline{\mathsf{RT}}$ (ms)", xformatter='%d', yformatter='%d')

    # Response times vs. PK slope
    logging.info('Computing linear regression...')
    rts = np.array(rts, dtype=float) * 1000
    pk_slopes = np.array(pk_slopes, dtype=float).ravel()
    linreg = linregress(pk_slopes, rts)
    print(linreg)
    mylog(0)
    ax = rtax2
    # ax.scatter(pk_slopes, rts, color='k')
    # ax.plot(pk_slopes, np.array(pk_slopes)*linreg.slope + linreg.intercept, '-')
    ax.plot(pk_slopes, rts)
    mod_ax_lims(ax, xlims=(-0.35, 0.5), xbounds=(-0.3, 0.5), xticks=(-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5),
                xlabel="PK slope",
                ylims=(130, 350), ybounds=(150, 350), yticks=(150, 250, 350), ylabel="Avg. RT (ms)",
                xformatter='%.1f', yformatter='%d')
    ax.set_xticklabels(["", -0.2, "", 0, "", 0.2, "", 0.4, ""])

    if save:
        directory = 'results/paper_figs/fig_b3/'
        fig.set_tight_layout(False)
        filename = directory + f"fig_b3"
        save_plot(fig, filename, overwrite=True, auto=True, saveformat=['png', 'pdf', 'svg'])

    return fig

def plot_fig_s2(save=False, **kwargs):
    # Compute some results
    i0s = kwargs.pop('i0range', np.arange(0.01, 0.105, 0.005))
    mus = kwargs.pop('i0s', [0.02, 0.05, 0.08])
    thetas = kwargs.pop('thetas', np.linspace(0, 150, 51))
    durations = kwargs.get('durations', [0.250, 0.125])
    if kwargs.pop('compute_impact', False):
        datas = []
        logging.debug("Computing stimulus efficacy...")
        for duration in durations:
            options = dict(tmax=duration, tframe=duration, nframes=1, orientations=np.array([90]),
                           sigmaOU=kwargs.get('sigmaOU', 0.0), i0s=i0s, thetas=thetas, i1=kwargs.get('i1', 0.1),
                           translate=kwargs.get('translate', True), rinit=-2)
            datas.append(measure_delta_ang(**options))
            datas[-1]['duration'] = duration

        data = pd.concat(datas)
        data.to_csv('figs/stim_impact/data_medium.csv')
    else:
        data = pd.read_csv('figs/stim_impact/data_medium.csv')

    custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 300

    # Set up grid
    gs1 = gridspec.GridSpec(1, 1)
    gs2 = gridspec.GridSpec(1, 2)

    # Axes location variables
    (left, right, top, bottom) = (0.07, 0.4, 0.94, 0.25)
    hspace = 0.0
    wspace = 0.0
    gs1.update(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)

    wspace = 0.1
    (left, right, top, bottom) = (0.52, 0.98, 0.94, 0.25)
    gs2.update(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)

    figsize = kwargs.pop('figsize', (4, 1.6))
    fig = plt.figure(figsize=figsize)  # Size is in inches (calculated for a US Letter size in inkscape)
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(False)

    # Psi evolution ax
    psiax = fig.add_subplot(gs1[0, 0])
    # change of phase vs. i0
    i0ax = fig.add_subplot(gs2[0, 0])
    # change of phase vs. stim orientation
    thetax = fig.add_subplot(gs2[0, 1], sharey=i0ax)

    # Kernels
    pkax1 = [inset_axes(psiax, width="25%", height="30%", loc=k, borderpad=0) for k in [1, 7, 4]]

    axes = fig.get_axes()
    for axb in axes:
        axb.patch.set_alpha(0.0)
        axb.grid(False)

    # Delete some axes
    mod_axes(axes)

    # Actual plotting starts here
    # #### #### #### #### #### ##

    # psiax: Evolution of the phase, psi vs. time
    dashed = (0, (10, 3))
    tframe = int(0.250 / 2E-4)
    markers = ('o', 's', '^')

    t = []
    label = r"$I_0 = " if kwargs.get('translate', False) else r"$\tilde{I}_0 = "
    options = kwargs.copy()
    options.update(sigmaOU=0.0, correction=(2 * kwargs.get('sigmaOU', 0.15) ** 2), rinit=-2)
    i0colors = []
    for k, mu in enumerate(mus):  # Simulations are run for each value of mu (aka i0)
        t, r, p, _, iv, _, _ = simulation(tmax=1.0, nframes=1, tframe=1.0, orientations=np.array([90]), mu=mu,
                                          theta_0=0, init_time=0.0, **options)

        line, = psiax.plot(t, np.rad2deg(p), label=(label + (r"%.2f$" % mu)))
        color = line.get_color()
        i0colors.append(color)
        psiax.plot([-0.1, 0], [0, 0], color=color)

        psiax.plot(t[tframe], np.rad2deg(p[tframe]), markers[k], color=color, markersize=4)
        t0, psi_aprox = psi_evo_r_constant((0, 1.0), r=r[-1], i1=iv[-2])  # We also plot the approximation
        psiax.plot(t0, psi_aprox, color=color, alpha=0.5)

    psiax.plot(t, 90 * np.ones_like(t), linestyle=dashed, color='black', lw=0.2, zorder=1)
    psiax.plot([0.25], [95], 'v', color='black', markersize=5, clip_on=False)
    psiax.plot([0.125], [95], 'v', color='darkgray', markersize=5, clip_on=False)

    sl = kwargs.pop('psilim', 90)
    # psiax.legend(frameon=False, fontsize=6)
    mod_ax_lims(psiax, xlims=(t[0] - 0.1, t[-1] + 0.5), xticks=[0, 0.5, 1.0], ylims=(-5, sl), yticks=[0, 45, 90],
                xbounds=(-0.1, 1.0), ybounds=(0, 90), ylabel=r"$\psi(t)\ (^\circ)$",
                xformatter='%.1f', yformatter='%d')
    psiax.set_xlabel(r'Time (s)', position=(0, 0), ha='left')

    # i0ax and thetax: Phase displacement as a function of mu (i0) for two stimulus durations
    #                  and phase displacement as a function of the stimulus angle for 3 different mu (i0)
    colors = ['black', 'darkgray']
    for m, d in enumerate(durations):
        df = data.loc[(data.duration == d)]
        df0 = df.loc[(np.abs(df.theta - 90) < 1)]

        i0ax.plot(df0.i0, df0.psi, color=colors[m], label=r"$%d$ ms" % int(d * 1000), linestyle='solid')
        i0ax.plot(df0.i0, df0.psi_aprox, color=colors[m], alpha=0.3, linestyle='dashed')

        # To plot the phase displacement as a function of the stimulus angle for three different mu
        i0_uniq = np.sort(df.i0.unique())
        tol = (i0_uniq[1] - i0_uniq[0]) / 2
        if m == 0:
            for k, mu in enumerate(mus):
                data_mu = df.loc[np.abs(df.i0 - mu) < tol]

                line, = thetax.plot(data_mu.theta, data_mu.psi)
                c = line.get_color()

                i0ax.plot(mu, data_mu.loc[np.abs(data_mu.theta - 90) < 1].psi, markers[k], color=c, markersize=4)
                thetax.plot([90], data_mu.loc[np.abs(data_mu.theta - 90) < 1].psi, markers[k], color=c, markersize=4)
                thetax.plot(data_mu.theta, data_mu.psi_aprox, color=c, alpha=0.5)

    # Additional elements of both plots (i0ax and thetax)
    xlabel = r"Excitatory drive, $I_0$" if kwargs.get('translate', False) else r"Excitatory drive, $\tilde{I}_0$"
    ylabel = r"$\theta^{\mathrm{stim}}\ (^\circ)$"

    i0ax.plot([i0s[0], i0s[-1]], [90, 90], linestyle=dashed, color='black', lw=0.2, zorder=1)
    mod_ax_lims(i0ax, xlims=(i0s[0] / 2, i0s[-1] + i0s[0] / 2), ylims=(-2, 45), xticks=list(np.array(mus)),
                yticks=[0, 15, 30, 45], xbounds=(i0s[0], i0s[-1]), ybounds=(0, 45), xformatter='%.2f',
                yformatter='%d', ylabel=r"$\Delta\psi\ (^\circ)$", xlabel=xlabel)
    i0ax.legend(frameon=False, fontsize=8)

    thetax.plot([0, 150], [0, 150], linestyle=dashed, color='black', lw=0.2, zorder=1)
    mod_ax_lims(thetax, xlims=(-5, 91), xticks=[0, 45, 90, 150], yticks=[0, 15, 30, 45],
                xbounds=(0, 150), ybounds=(0, 45), xformatter='%d',
                xlabel=ylabel)
    plt.setp(thetax.get_yticklabels(), visible=False)

    # Psychophysical kernels
    _, (w_ring, w_amp), _ = get_amp_eq_stats(mus, rinit=kwargs.get('pkrinit', 0.2), **kwargs)
    # Get kernels for the unbiased bump condition and biased bump condition
    df = pd.read_csv('./results/integration_stats_frames-8_t-2.0_new.csv')
    sel = df.loc[(df.bump == True) & (df.biased == True) & (df.sigma == 0.15)].copy()
    xlabels = ['x%d' % k for k in range(1, 9)]
    w_ring = [sel.loc[sel.i0 == i0, xlabels].copy().to_numpy().ravel() for i0 in mus]
    # sel = sel[xlabels + ['i0']].copy()

    for k, (wr, wa) in enumerate(zip(w_ring, w_amp)):
        frames = list(range(1, len(wr) + 1))
        pkax1[k].plot(frames, wr, label=(label + (r"%.2f$" % mus[k])), color=i0colors[k])
        pkax1[k].plot(frames, np.ones_like(frames) * 0.125, ls='--', color='k', lw=0.1, zorder=1)
        pkax1[k].spines['left'].set_visible(False)
        # pkax1[k].spines['bottom'].set_visible(False)
        mod_ax_lims(pkax1[k], xlims=(0.5, 8.5), ylims=(-0.05, 0.3),
                    xticks=[], yticks=[], xbounds=(1, 8), ybounds=(0, 0.3))
        plt.setp(pkax1[k].get_yticklabels(), visible=False)
        plt.setp(pkax1[k].get_xticklabels(), visible=False)
        pkax1[k].patch.set_alpha(0.3)
        pkax1[k].set_facecolor('darkgray')

    if save:
        directory = 'figs/stim_impact/'
        fig.set_tight_layout(False)
        filename = directory + f"fig_s2"
        save_plot(fig, filename, overwrite=False, auto=True, saveformat=['png', 'pdf', 'svg'])

    plt.show()
    return data


def plot_fig_supp_distr(save=False, **kwargs):
    """ Supplementary figure showing the evolution of the alpha variable (t·R) as a function of the Von-Mises
    concentration parameter (kappa). And also the minimum estimation error as a function of kappa using the amplitude
    equation.

    :param bool save: whether to save the figure or not.
    :param kwargs: additional options.
    :return: plt.Figure
    """
    # Prepare the data
    filename = 'results/supplementary_alpha_nobump_8_10000'
    logging.info(f"Loading data from '{filename}.npy' ...")
    results = load_obj(filename, extension='.npy')
    kappas = [0.1, 1, 10, 100]
    i0s = np.arange(1.0, 1.101, 0.001)
    stim = results['stim'][0]
    phi = results['phi'][0]
    filename = 'results/supplementary_alpha_nobump_8_10000_fine2'
    logging.info(f"Loading data from '{filename}.npy' ...")
    results_fine = load_obj(filename, extension='.npy')
    kappas_fine = np.concatenate((np.arange(1, 3, 0.1), np.arange(3, 10, 1)))
    i0s_fine = np.arange(0.0, 0.02, 0.001) + 1
    stim_fine = results_fine['stim'][0]
    phi_fine = results_fine['phi'][0]

    nframes, ntrials = stim[0].shape

    # #### #### #### #### #### ####
    # Prepare figure      #### ####
    # #### #### #### #### #### ####
    custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 300

    # Create figure and axes
    figsize = kwargs.pop('figsize', (3.9, 2.9))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.22, hspace=0.0, wspace=0.5)

    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.grid(False)

    # Delete some axes
    mod_axes(axes)

    # Colors and styles
    reds = plt.get_cmap(kwargs.get('cmap', 'Reds'))

    # Actual plotting starts here
    # #### #### #### #### #### ##
    # Distribution and alpha, and also errors (some)
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    t = np.arange(nframes) + 1
    minimum = []
    for k, kappa in enumerate(kappas):
        label = r"$\kappa = %s$" % ('%d' % kappa if kappa >= 1 else '%.1f' % kappa)
        color = reds(0.2 + k * 0.2)
        ax1.hist(np.rad2deg(stim[k].ravel()), bins=100, density=True, alpha=0.5, color=color, label=label)
        th = results['stim'][0, k]
        _, r, _, _ = running_avg(np.exp(1j * th))
        ax2.plot(t, t * r.mean(axis=1), color=color, label=label)

        # Errors
        logging.debug(f"Computing estimation errors for kappa = {kappa} ...")
        sqr_err = []
        for j, i0 in enumerate(i0s):
            df = pd.DataFrame(dict(estim=np.rad2deg(phi[k, j, 1]),
                                   average_circ=np.rad2deg(circmean(stim[k], low=-np.pi, high=np.pi, axis=0))))
            true = np.deg2rad(df.average_circ)
            y = np.deg2rad(df.estim)
            sqr_err.append(np.mean(np.rad2deg(circ_dist(y, true)) ** 2))
        ax3.plot(i0s, np.sqrt(sqr_err), color=color)
        argmin = np.argmin(sqr_err).ravel()[0]
        ax3.plot(i0s[argmin], np.sqrt(sqr_err[argmin]), 'o', color=color, markersize=2)
        minimum.append((i0s[argmin], np.sqrt(sqr_err[argmin])))

    ax2.plot(t, t, '--', color='k', label=r"$t$")
    ax2.plot(t, np.sqrt(t), ':', color='k', label=r"$\sqrt{t}$")

    mod_ax_lims(ax1, xlims=(-180, 180), xticks=(-180, -90, 0, 90, 180), ylims=(0, 0.03),
                xlabel=r"$\theta^{\mathrm{stim}}_j$", ylabel=r"Density", xformatter="%d", yformatter='%.2f',
                xbounds=(-180, 180), ybounds=(0, 0.025))
    mod_ax_lims(ax2, xlims=(0.5, 8), xticks=(1, 4, 8), ylims=(0.5, 8), xlabel=r"Time $(t)$, a.u.",
                ylabel=r"$\left<R(t)\right>$", xformatter="%d", yformatter='%d',
                xbounds=(1, 8), ybounds=(1, 8))

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)

    # Error
    ax = axes[2]
    # Get errors for intermediate kappas
    minimum_fine = []
    for k, kappa in enumerate(kappas_fine):
        sqr_err = []
        logging.debug(f"Computing estimation errors for kappa = {kappa} ...")
        for j, i0 in enumerate(i0s_fine):
            df = pd.DataFrame(dict(estim=np.rad2deg(phi_fine[k, j, 1]),
                                   average_circ=np.rad2deg(circmean(stim_fine[k], low=-np.pi, high=np.pi, axis=0))))
            true = np.deg2rad(df.average_circ)
            y = np.deg2rad(df.estim)
            sqr_err.append(np.mean(np.rad2deg(circ_dist(y, true)) ** 2))
        argmin = np.argmin(sqr_err).ravel()[0]
        minimum_fine.append((i0s_fine[argmin], np.sqrt(sqr_err[argmin])))

    # All min points
    # kappas_t = np.concatenate((kappas[0:2], kappas_fine, kappas[2:]))
    minimum_t = np.concatenate((minimum[0:2], minimum_fine, minimum[2:]))

    ax.plot(minimum_t[:, 0], minimum_t[:, 1], '-', label='Minimum', color='k')
    mod_ax_lims(ax, xlims=(0.998, 1.04), xticks=(1, 1.02, 1.04), ylims=(-1, 20),
                xlabel=r"Excitatory drive $I_\mathrm{exc}$", ylabel=r"Estimation error $(^\circ)$",
                xformatter="%.2f", yformatter='%d', xbounds=(1, 1.04), ybounds=(0, 20))
    ax.set_xticklabels([r"$I_{\mathrm{crit}}=1$", 1.02, 1.04])
    ax.legend(frameon=False)

    if save:
        directory = 'results/paper_figs/fig_supp_running_avg/'
        fig.set_tight_layout(False)
        filename = directory + f"fig_supp_ra_lim1"
        save_plot(fig, filename, overwrite=False, auto=True, saveformat=['png', 'pdf', 'svg'])
        mod_ax_lims(ax1, xlims=(-180, 180), xticks=(-180, -90, 0, 90, 180), ylims=(0, 0.07),
                    xlabel=r"$\theta^{\mathrm{stim}}_j$", ylabel=r"Density", xformatter="%d", yformatter='%.2f',
                    xbounds=(-180, 180), ybounds=(0, 0.07))
        filename = directory + f"fig_supp_ra_lim2"
        save_plot(fig, filename, overwrite=False, auto=True, saveformat=['png', 'pdf', 'svg'])

    return None


def plot_fig_supp_error(save=False, **kwargs):
    """ Supplementary figure showing the estimation error as a function of time for different conditions.

    :param bool save: whether to save the figure or not.
    :param kwargs: additional options.
    """

    # Prepare the data
    def compute_sqr_error(dataframe, xls):
        frames = len(xls)
        true = np.deg2rad(circmean(dataframe[xls].to_numpy(), low=-180, high=180, axis=1))
        y = np.deg2rad(dataframe[f"ph{frames}"])
        es_error = np.mean(np.rad2deg(circ_dist(y, true)) ** 2)
        return es_error

    directory = './results/variable_frames/'
    mod_ic = True
    if mod_ic:
        directory = './estimation_error/new_data/'
    else:
        directory = './estimation_error/'
    directory_s = directory[2:-1]

    filename = directory + "simu_10000_nobump_sigma-{sigma:.2f}_i0-{i0}_frames-16_t-4.0.csv"
    sigma = kwargs.pop('sigmaOU', 0.15)
    i0s = list(np.array([0.02, 0.05, 0.08, 0.10]) - 2 * (sigma - 0.15) ** 2)
    pk_frames = [4, 8, 16]
    betas, errors = {}, {}

    # Compute PVI errors
    df_pvi = pd.read_csv('estimation_error/pvi_10000.csv')
    errors_pvi = []
    for nframes in range(1, 17):
        xlabels = [f"x{k}" for k in range(1, nframes + 1)]
        es_err = compute_sqr_error(df_pvi, xlabels)
        errors_pvi.append(es_err)

    if kwargs.pop('compute_impact', False):
        for i0 in i0s:
            if np.round(i0 * 100) == i0 * 100:
                fn = filename.format(sigma=sigma, i0=f"{i0:.2f}")
            else:
                fn = filename.format(sigma=sigma, i0=f"{i0:.3f}")
            logging.info(f"Loading data from '{fn}' ...")
            df = pd.read_csv(fn)
            betas[i0], errors[i0] = [], []
            for nframes in range(1, 17):
                xlabels = [f"x{k}" for k in range(1, nframes + 1)]
                es_err = compute_sqr_error(df, xlabels)
                errors[i0].append(es_err)
                if nframes in pk_frames:
                    b, _, _ = compute_circ_reg_ml(df, xlabels, f"ph{nframes}")
                    betas[i0].append(b)
        if save:
            errors['pvi'] = errors_pvi
            data = dict(betas=betas, errors=errors)
            save_obj(data, f"supp_errors_data_sigma-{sigma}", directory=directory_s)
    else:
        data = load_obj(f"supp_errors_data_sigma-{sigma}", directory=directory_s)
        betas, errors = data['betas'], data['errors']

    # Override saving
    errors['pvi'] = errors_pvi
    data = dict(betas=betas, errors=errors)
    save_obj(data, 'supp_errors_data', directory='estimation_error')

    # #### #### #### #### #### ####
    # Prepare figure      #### ####
    # #### #### #### #### #### ####
    custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 300

    # Set up grid
    gs1 = gridspec.GridSpec(1, 1)

    # Axes location variables
    margins = dict(left=0.1, right=0.4, top=0.96, bottom=0.2, hspace=0.0, wspace=0.0)
    gs1.update(**margins)
    gsk = []
    wspace = 0.05 / 3
    pks_w = 1 - margins['right'] - 0.1 - wspace * 3
    total_frames = np.sum(pk_frames)
    margins.update(hspace=0.4, left=(margins['right'] + 0.1))
    for k, nframes in enumerate(pk_frames):
        width = nframes / total_frames * pks_w
        if k != 0:
            margins.update(left=margins['right'] + wspace)
        margins.update(right=margins['left'] + width)
        gsk.append(gridspec.GridSpec(len(i0s) + 1, 1))
        gsk[-1].update(**margins)

    # Create figure and axes
    figsize = kwargs.pop('figsize', (4.875, 2.9))
    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(False)

    # Main plot (estimation error)
    eax = fig.add_subplot(gs1[0, 0])

    # PKs axes
    kax = [[fig.add_subplot(gsk[n][m, 0]) for n in range(len(pk_frames))] for m in range(len(i0s) + 1)]

    axes = fig.get_axes()
    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.grid(False)

    # Delete some axes
    mod_axes(axes)

    # Colors and styles
    cs = ['red', 'black', 'blue', 'Green', (0.6, 0.6, 0.6)]
    styles = ['-', '-', '-', '-', '--']

    # Actual plotting starts here
    # #### #### #### #### #### ##
    # Estimation errors (log-log scale)
    for k, (i0, err) in enumerate(errors.items()):
        label = r"$I_0 = %.2f$" % i0 if i0 != 'pvi' else r"PVI"
        l, = eax.loglog(np.arange(1, 17) * 0.250, np.sqrt(np.array(err)), label=label, color=cs[k], basex=2,
                        lw=1.5, ls=styles[k], zorder=5)
        if i0 == 'pvi':
            l.set_zorder(1)
    eax.get_xaxis().set_major_formatter(ScalarFormatter())
    eax.get_yaxis().set_major_formatter(ScalarFormatter())

    mod_ax_lims(eax, xlims=(0.0, 4.0), xticks=(0.250, 1, 2, 4), ylims=(5, 30), yticks=(5, 10, 20, 30),
                xbounds=(0.250, 4.0), ybounds=(5, 30), xlabel='Stimulus duration (s)', ylabel=r'Estimation error ($^\circ$)')
    eax.set_xticklabels(['0.25', '1', '2', '4'])
    eax.set_yticklabels(['5', '10', '20', '30'])
    eax.legend(frameon=False, fontsize=8)
    # eax.set_yticklabels(['1', '20', '40', '60'])

    # Kernels
    cs = [(0.6, 0.6, 0.6), 'red', 'black', 'blue', 'Green']
    for m, i0 in enumerate(['null'] + i0s):
        for n, nframes in enumerate(pk_frames):
            if i0 == 'null':
                weights = np.ones(nframes) * (1.0 / nframes)
                style = '--'
            else:
                weights = betas[i0][n]
                style = '-'
            kax[m][n].plot(range(1, nframes + 1), weights, color=cs[m], lw=1.5, ls=style)
            kax[m][n].plot([1, nframes], [0, 0], '--', color='k', lw=0.5)
            mod_ax_lims(kax[m][n], xlims=(0.5, nframes + 0.5), xticks=(), ylims=(-0.01, 0.4), yticks=(0, 0.4),
                        xbounds=(0, nframes + 1), ybounds=(0, 0.4), yformatter='%.1f')
            if n == 0:
                if m == 2:
                    kax[m][n].set_ylabel(r"Stimulus impact")
                if m != 4:
                    plt.setp(kax[m][n].get_yticklabels(), visible=False)
            else:
                plt.setp(kax[m][n].get_yticklabels(), visible=False)
                kax[m][n].spines['left'].set_visible(False)
                kax[m][n].tick_params(axis='y', which='both', length=0)

            kax[m][n].spines['bottom'].set_visible(False)

    if save:
        directory = 'results/paper_figs/fig_supp_error/'
        fig.set_tight_layout(False)
        filename = directory + f"fig_supp_error"
        save_plot(fig, filename, overwrite=False, auto=True, saveformat=['png', 'pdf', 'svg'])

    return None


init_options = dict(datafile="results/simu_10000_nobump_sigma-0.15_i0-0.06.npy", perfect=False,
                    save=False, db='INFO', fig=1, gui=False, pheight=60, sheight=40, figsize=(800.0, 1080.0),
                    linewidth=2.0, outline=False, base_r0=2, eslim=45, pmlim=45, ppk_lims=(0.0, 0.3), compute_fit=False,
                    lim=90, nbins=41, i0s=(0.02, 0.05, 0.08), cmap='Spectral', compute_impact=False, translate=False,
                    sigmaOU=0.15, i1=0.01, simulate_amp_eq=False, save_amp_eq=False, pkrinit=0.2, threed=False,
                    acc_label='pmy25', ang_tol_b=2.0, ang_tol_nb=1.5, supp=False, unbiased_pk=False,
                    ntrials=1, trials=[100], fframe=-1, transitions=2, compute_pk=False)

if __name__ == '__main__':
    # Parsing, logging, debugging
    pars = Parser(desc="Paper's figures generator.", conf=init_options)
    conf_options = pars.opts
    logger, mylog = log_conf(pars.args.db, name='fig_generator')
    init_options.update(conf_options)
    if pars.args.fig == 1:
        # options: linewidth=3.0, trial=100, perfect=True, figsize=(800.0, 1080.0)
        init_options.update(trial=pars.args.trials[0])
        fs = init_options.pop('figsize')
        if pars.args.threed:
            all_trials = set(range(500))
            possible_trials = all_trials - set(pars.args.trials)
            for t in range(pars.args.ntrials):
                sel_trial = pars.args.trials.pop() if pars.args.trials else possible_trials.pop()
                init_options.update(trial=sel_trial)
                plot_fig1_3d(figsize=(800, 1080), **init_options)
        if pars.args.ntrials > 1:
            exit(0)
        # options: eslim=60, pmlim=60, figsize=(1.5, 3.55), nbins=11
        figure = plot_fig1_stats(eslim=init_options.pop('eslim', 45), pmlim=init_options.pop('pmlim', 45),
                                 save=init_options.get('save', False), figsize=fs, nbins=init_options.get('nbins', 41))
    elif pars.args.fig == 2:
        # python3 -W ignore figs_paper.py -fig 2 -perfect -linewidth 5.0 -trial 0 -base_r0 2 -sheight 20 -outline
        # -datafile "results/paper_figs/fig2_potential_transitions/simu_1_nobump_sigma-0.00_i0-0.08_01.npy"
        # -figsize 1080 1080 -transitions 2
        # options: linewidth=5.0, trial=0, perfect=True, base_r0=2, sheight=20, figsize=(1080, 1080), outline=True
        plot_fig2_3d(**init_options)
    elif pars.args.fig == 3:
        # options: figsize=(4.2, 3.6), eslim=90, i0s=(0.02, 0.05, 0.08)
        plot_fig3_ppk(**init_options)
    elif pars.args.fig == 4:
        # options: figsize=(4.5, 4.3), eslim=60, pmlim=60, acc_label='pmyr26'
        plot_fig4_est(**init_options)
    elif pars.args.fig == 5:
        # options: figsize=(6.2, 3), eslim=60, ang_tol_nb=1.5
        plot_fig5_bump(**init_options)
    elif pars.args.fig == 11:  # Behavioral 1
        # Options: figsize=(3.9, 2.4), i0s=(0.02, 0.05, 0.07)
        # New Options (histogram): figsize=(5.2, 2.4), i0s=(0.02, 0.05, 0.07)
        dummy = init_options.pop('i0s', None)
        plot_fig_b1(i0s=(0.07, 0.05, 0.02), **init_options)
    elif pars.args.fig == 13:  # Behavioral 3
        plot_fig_b3(**init_options)
    elif pars.args.fig == 22:  # Supplementary figure 2
        # Options: cmap='Spectral', save=False, compute=True, i0s=np.arange(0.01, 0.11, 0.01), thetas=np.linspace(0,
        # 150, 51), i1=0.01, mus=[0.02, 0.05, 0.08], translate=True, sigmaOU=0.15
        # Options new: translate=True, figsize=(6, 3.4), sigmaOU=0.15
        plot_fig_s2(**init_options)
    elif pars.args.fig == 55:  # Supplementary figure 5 (uses fig 3 and 5)
        # options: figsize=(4.2, 3.6), eslim=90, i0s=(0.02, 0.05, 0.08)
        init_options.update(supp=True, figsize=(4.2, 3.6), eslim=90, i0s=(0.02, 0.05, 0.08))
        plot_fig3_ppk(**init_options)
        # options: figsize=(6.2, 3), eslim=60, ang_tol_nb=1.5
        init_options.update(figsize=(6.2, 3), eslim=60, ang_tol_nb=1.5)
        plot_fig5_bump_supp(**init_options)
    elif pars.args.fig == 66:  # Supplementary figure: distributions
        init_options.update(supp=True)
        plot_fig_supp_distr(**init_options)
    elif pars.args.fig == 77:  # Supplementary figure: estimation error
        init_options.update(supp=True)
        plot_fig_supp_error(**init_options)
    if init_options.get('gui', False):
        gui = GUI()
        gui.start_event_loop()
    else:
        plt.show()
