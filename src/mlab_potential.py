"""
Plot of the potential using Mayavi mlab
=======================================

.. currentmodule:: mlab_potential
   :platform: Linux
   :synopsis: module for plotting and simulating the amplitude equation and its potential

.. moduleauthor:: Jose M. Esnaola-Acebes <josemesnaola@gmail.com>

This module contains an interactive visualization of the potential and several functions to process and analyze the
dynamics of the amplitude equation.
"""
import matplotlib

matplotlib.use('Qt5Agg')

import locale
import sys
import logging

logging.getLogger('mlab_potential').addHandler(logging.NullHandler())

sys.path.append('lib')
from lib_ring import potential, psi_evo_r_constant, r_roots
from lib_ring import amp_eq_simulation as simulation
from lib_sconf import log_conf, Parser, check_overwrite
from lib_plotting import hex_to_rgb
import pandas as pd

locale.setlocale(locale.LC_NUMERIC, "C.UTF-8")

from mayavi import mlab
import numpy as np

from pyface.api import GUI
from mayavi.core.api import Engine  # The core Engine.
from mayavi.core.ui.engine_view import EngineView

from traits.api import HasTraits, Instance, Range, on_trait_change
from traitsui.api import View, Item, HGroup, Group, Action
from mayavi.core.ui.api import SceneEditor, MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.filters.data_set_clipper import DataSetClipper


@mlab.show
class Visualization(HasTraits):
    """ Tool to visualize the potential in 3D. It plots and animates the potential and its dynamics based on
    simulations of the amplitude equation.
    """
    mu = Range(0.0, 1.0, 1.0)
    i1 = Range(0.0, 0.4, 0.0)
    c = Range(-2.0, 0.0001, -1.0)
    theta = Range(-180, 180, 0, mode='slider')
    theta_b = Range(-180, 180, 0, mode='slider')
    scene = Instance(MlabSceneModel, ())
    last_time = -1
    t = 0
    r_ball = -1
    animating = False

    def __init__(self, **kwargs):
        HasTraits.__init__(self)
        self.colormap = kwargs.pop('colormap', 'Reds')

        mu = kwargs.pop('mu_init', self.mu)
        i1 = kwargs.pop('i1_init', self.i1)
        c = kwargs.pop('c', self.c)
        theta = kwargs.pop('theta_s', self.theta)

        self.zscale = kwargs.get('zscale', 4.0)
        self.options = kwargs.copy()

        # Get trajectories by simulating the amplitude equation
        self.dt = 2E-4
        self.step = 10
        self.options['dt'] = self.dt
        self.options['c'] = c
        self.options['rinit'] = 0.00001
        self.options['init_time'] = 0.1
        self.tpoints, self.r, self.phi, self.mus, self.i1s, self.thetas, _ = simulation(smooth=True, **self.options)
        self.r[0] = 0.0
        self.tmax = self.tpoints[-1]
        self.options['mu'] = mu
        self.options['i1'] = i1
        self.i2 = self.options['i2']

        # Create potential
        x, y, z, self.rmax = create_mesh(do_plot=False, mu=mu, c=c, i1=i1, theta_s=theta, **kwargs)
        mesh = self.scene.mlab.mesh(x, y, z, scalars=z, colormap=self.colormap)
        self.plots = plot_potential(front_mesh=mesh, scene=self.scene, **kwargs)
        self.engine = mlab.get_engine()

        # Plot lines
        self.base = self._plot_base()
        self.profile = self._plot_potential_profile()
        # Clip the profile line
        clipper = DataSetClipper()
        vtk_data_source = self.engine.current_scene.children[-1]
        self.engine.add_filter(clipper, vtk_data_source)
        clipper.add_child(vtk_data_source.children[0])

        # Create ball
        self.ball = self._plot_ball()

        # Change camera and set user-defined parameters
        self.trait_set(mu=mu, i1=i1, theta=theta, c=c)
        self._setup_visualization()

        self.scene.movie_maker.record = kwargs.get('record', False)

        # Modify the clip
        clipper.widget.widget_mode = 'ImplicitPlane'
        clipper.widget.widget.normal_to_z_axis = True
        # clipper.widget.widget.normal = np.array([0., 0., 1.])
        clipper.filter.inside_out = True
        clipper.filter.use_value_as_offset = True
        clipper.filter.value = 0.5
        clipper.widget.widget.enabled = False

    def _setup_visualization(self):
        """ Initializes the gui. Configures the traits and sets the 3d view.

        :return: None
        """
        self.configure_traits()
        self._set_view_default()

    def _set_view_default(self):
        """ Sets the view of the 3d representation.

        :return: None
        """
        self.scene.mlab.view(azimuth=30, elevation=70, focalpoint=[0, 0, -0.5], distance=6.0)  # good view

    def _update_plot(self, **kwargs):
        """ Updates the 3d representation when the parameters are changed

        :param kwargs: contains user defined options (keyword arguments).
        :return: None
        """
        self.scene.disable_render = True
        self.options.update(i1=self.i1, theta_s=self.theta, mu=self.mu, c=self.c)
        x, y, z, self.rmax = create_mesh(do_plot=False, **{**kwargs, **self.options})
        for plot in self.plots:
            plot.mlab_source.set(x=x, y=y, z=z, scalars=z)
        self._plot_base(update=True)
        self._plot_potential_profile(update=True)
        self._plot_ball(update=True)
        self.scene.disable_render = False

    @on_trait_change(['mu', 'i1', 'theta', 'c'])
    def _update_plot_any(self):
        """ Method that updates the 3d representation when a parameter affecting the potential is changed.
        The updating is done once in each time step when the _animation is running.

        :return: None
        """
        if self.last_time != self.t or not self.animating:
            self._update_plot()
        if self.animating:
            self.last_time = self.t

    @on_trait_change('theta_b')
    def _update_plot_theta_b(self):
        """ Method that updates the 3d representation when the position of the ball changes. It only affects the
        representation of the ball. The potential is not changed.

        :return: None
        """
        self._plot_ball(update=True)

    def _do_animate(self):
        """ Start the _animation. This happens when the button `animate` is pressed.

        :return: None
        """
        self._set_view_default()
        self.animating = True
        self.t = 0
        self._animation()

    # noinspection PyTypeChecker
    def _plot_base(self, update=False):
        """ Plots the 3d line that defines the bottom of the well.

        :param bool update: whether the line is updated or is created.
        :return: None or a 3d mlab line.
        """
        r, th = r_roots(self.mu, self.i1, self.theta, c=self.c, i2=self.i2)
        x = r * np.cos(th)
        y = r * np.sin(th)
        z = self.zscale * potential(r, th, mu=self.mu, i1=self.i1, i2=self.i2, c=self.c, theta_s=self.theta, polar=True)
        if update:
            self.base.mlab_source.reset(x=x, y=y, z=z)
        else:
            return mlab.plot3d(x, y, z, color=(0.0, 0.0, 0.0), tube_radius=None)

    # noinspection PyTypeChecker
    def _plot_potential_profile(self, update=False):
        """ Plots the 3d line that defines the vertical cut of the potential at `self.theta`.

        :param bool update: whether the line is updated or is created.
        :return: None or a 3d mlab line.
        """
        r = np.linspace(-self.rmax, self.rmax, 101)
        z = self.zscale * potential(r, np.deg2rad(self.theta), mu=self.mu, i1=self.i1, i2=self.i2, c=self.c,
                                    theta_s=self.theta, polar=True)
        mask = z > self.options['zmax']
        z = z[~mask]
        r = r[~mask]
        x = r * np.cos(np.deg2rad(self.theta))
        y = r * np.sin(np.deg2rad(self.theta))
        if update:
            self.profile.mlab_source.reset(x=x, y=y, z=z)
        else:
            profile = mlab.plot3d(x, y, z, color=(0.0, 0.0, 0.0), tube_radius=None)
            return profile

    # noinspection PyTypeChecker
    def _plot_ball(self, update=False):
        """ Plots the 3d sphere that represents the state of the network on the surface of the potential.

        :param bool update: whether the glyph is updated or is created.
        :return: None or a 3d mlab glyph.
        """
        if not self.animating:  # Set the position based on the `self.theta_b` slider.
            bx, by, bz = get_ball_position(theta_ball=np.deg2rad(self.theta_b), **self.options)
        else:  # If animating: set the position based on the simulated data.
            k = int(self.t / self.dt)
            bx, by = (self.r[k] * np.cos(np.deg2rad(self.theta_b)), self.r[k] * np.sin(np.deg2rad(self.theta_b)))
            bz = self.zscale * potential(bx, by, polar=False, theta_s=self.theta, mu=self.mu, i1=self.i1, i2=self.i2,
                                         c=self.c)
        if update:
            self.ball.mlab_source.reset(x=bx, y=by, z=bz)
        else:
            ball = mlab.points3d(bx, by, bz, resolution=20, color=(0, 0, 0), scale_factor=0.25)
            # TODO: compute the distance to the surface using the normal vector (tangent to the gradient of the surface)
            ball.glyph.glyph_source.glyph_source.center = (0, 0, 0.5)  # Displace the glyph in the z-axis
            ball.actor.property.color = (0.345, 0.7, 0.95)
            return ball

    @mlab.show
    @mlab.animate(delay=10)
    def _animation(self):
        """ Animating method. Creates a loop in time that updates the 3d parameters according to the simulated data.

        :return: None
        """
        while self.t < self.tmax:
            k = int(self.t / self.dt)
            self.trait_set(theta_b=int(np.rad2deg(self.phi[k])))
            self.trait_set(mu=self.mus[k], i1=self.i1s[k], theta=int(np.rad2deg(self.thetas[k])))
            self.t += self.dt * self.step
            self.r_ball = self.r[k]
            if self.t > self.tmax:
                self.animating = False
            yield

    # We define the groups of traits that let us modify the parameters in the Mayavi scene.
    group1 = Group(
        Item(
            'mu',
            style='simple',
            label='Mu'
        ),
        Item(
            'i1',
            style='simple',
            label='I_1'
        ),
        Item(
            'c',
            style='simple',
            label='c'
        )
    )
    group2 = Group(
        Item(
            'theta',
            style='simple',
            label='Stimulus'
        ),
        Item(
            'theta_b',
            style='simple',
            label='Ball'
        )
    )
    animate = Action(name='Animate', action='_do_animate')

    view = View(Item('scene', width=600, height=600, show_label=False, editor=SceneEditor(scene_class=MayaviScene)),
                HGroup(Item('_'), group1, Item('_'), group2, Item('_')), buttons=[animate],
                resizable=True)


def get_base_numerically(x, z):
    """ Obtains the base of the potential (where the partial derivative of the potential with respect R vanishes and the
    second derivative is positive) numerically by taking the minimum value of `z` that is assumed to have been computed
    using polar coordinates. (This function is not used because it gives a very bad representation).

    :param np.ndarray x: x-axis coordinates.
    :param np.ndarray z: z-axis coordinates.
    :return: cartesian coordinates of the line.
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """
    rn, tn = z.shape
    zmin_theta_arg = np.argmin(z, axis=0)
    rmax = np.max(x)
    r = np.linspace(0, rmax, rn)[zmin_theta_arg]
    theta = np.linspace(0, 2 * np.pi, tn)
    xc = r * np.cos(theta)
    yc = r * np.sin(theta)
    zc = np.min(z, axis=0)
    return xc, yc, zc


def get_ball_position(theta_ball=0.0, **kwargs):
    """ Computes the position of the ball given the parameters in `kwargs` and the angle `theta_ball`.

    In order to obtain a single branch solution the value passed to r_roots must be between -np.pi/2 and np.pi/2,
    this way we obtain r[0] the 'left' solution and r[1] the 'right' solution.
    Additionally, we have to pass the relative value between the ball angle and the stimulus angle.

    :param float theta_ball: current position of the ball.
    :param kwargs: additional keyword arguments that define the potential and properties of the stimulus.
    :return: cartesian coordinates of the ball.
    :rtype: (float, float, float)
    """

    logger.debug("Obtaining ball's position at the radial equilibrium...")
    logger.debug(f"\t... for parameters: {kwargs} ...")
    signed_distance = np.angle(np.exp(1j * theta_ball) / np.exp(1j * np.deg2rad(kwargs.get('theta_s', 0.0))))
    transformed_theta = (signed_distance + np.pi / 2) % np.pi - np.pi / 2
    r, th = r_roots(theta=transformed_theta, **kwargs)
    rsol = r[0] if ((np.abs(signed_distance) < np.pi / 2) or (len(r) == 1)) else r[1]
    logger.debug(f"\tat (r, theta_b) = ({rsol}, {theta_ball})")
    x = rsol * np.cos(theta_ball)
    y = rsol * np.sin(theta_ball)
    kwargs.pop('polar', True)
    z = kwargs.get('zscale', 4.0) * potential(rsol, theta_ball, polar=True, **kwargs)
    return x, y, z


def create_mesh(do_plot=True, **kwargs):
    """ Function that draws a mesh-like surface of the potential using :func:`mlab.mesh` or returns its coordinates
    to be drawn externally. The function is built such that the user may define the extension of the mesh by means
    of the optional keyword arguments `zmax` and/or `rmax`. The mesh is computed in `polar` coordinates by default,
    due to the radial symmetry of the potential.

    :param bool do_plot: whether to plot and return the surface or just return the cartesian coordinates.
    :param kwargs: additional key-word arguments to customize the surface.
    :return: A mlab mesh or a tuple with the cartesian coordinates of the mesh.
    :rtype: mlab.mesh or (np.ndarray, np.ndarray, np.ndarray, float)
    """
    # # My code
    mesh_size = kwargs.pop('mesh_size', (101, 101))
    zscale = kwargs.pop('zscale', 4.0)
    colormap = kwargs.pop('colormap', 'Reds')
    polar = kwargs.pop('polar', True)
    zmax = kwargs.pop('zmax', 0.0)
    rmax = kwargs.pop('rmax', None)

    # Compute `rmax` in case none is given
    if rmax is None:
        rmax = 3.001
        zm = zmax + 0.1
        theta_s = np.deg2rad(kwargs.get('theta_s', 0.0))
        while zm > zmax:
            rmax -= 0.001
            zm = zscale * potential(rmax, theta_s, polar=True, **kwargs)

    if polar:
        r, theta = np.mgrid[0:rmax:mesh_size[0] * 1j, 0:2 * np.pi:mesh_size[1] * 1j]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = zscale * potential(r, theta, polar=polar, **kwargs)
    else:
        xmax = rmax / np.sqrt(2)
        x, y = np.mgrid[-xmax:xmax:mesh_size[0] * 1j, -xmax:xmax:mesh_size[1] * 1j]
        z = zscale * potential(x, y, **kwargs)

    if do_plot:
        return mlab.mesh(x, y, z, scalars=z, colormap=colormap)
    else:
        return x, y, z, rmax


def plot_potential(front_mesh=None, engine=None, scene=None, **kwargs):
    """ Function that plots the potential using Mayavi's mlab module. The potential is drawn using two meshes. The
    front mesh is translucent and its backface is not shown, while the rear mesh is opaque but its frontface is not
    shown. This combination gives a comprehensible representation of the potential that allows to 'look' inside.

    :param front_mesh: main mesh that will give shape to the 3d plot. If none given one is created from scratch.
    :param engine: a Mayavi engine to use. If none is given a new engine is created and initialized.
    :param scene: a Mayavi scene where the potential will be drawn. If none is given a new scene will be produced.
    :param kwargs: additional key-word arguments to customize the plot.
    :return: front and rear meshes that correspond to the potential.
    :rtype: list
    """
    logger.info('Plotting the surface of the potential...')
    if engine is None:
        engine = mlab.get_engine()

    zrange = kwargs.pop('zrange', (-1.2, 0.5))
    # Create figure
    fig_size = kwargs.pop('figsize', (1920, 1080))
    fig = mlab.figure(engine=engine, size=fig_size) if scene is None else scene
    if scene is None:
        fig.name = 'Potential'

    # Create the first mesh and build upon this
    front_mesh = create_mesh(**kwargs) if front_mesh is None else front_mesh
    front_mesh.trait_set(name='Front')
    scene = engine.current_scene
    # View
    scene.scene.background = (1.0, 1.0, 1.0)
    scene.scene.foreground = (0.0, 0.0, 0.0)

    src_data = scene.children[0]
    normals = src_data.children[0]
    module_manager = normals.children[0]
    module_manager.scalar_lut_manager.reverse_lut = kwargs.get('reverse_cmap', False)
    colormap = module_manager.scalar_lut_manager.lut_mode

    # Produce second surface (mesh)
    rear_mesh = mlab.pipeline.surface(normals, name='Rear')
    module_manager.scalar_lut_manager.lut_mode = colormap
    # Freeze the range of the scalar representation
    module_manager.scalar_lut_manager.trait_set(use_default_range=False, data_range=zrange)
    module_manager.scalar_lut_manager.reverse_lut = kwargs.get('reverse_cmap', False)
    meshes = [front_mesh, rear_mesh]

    # Modify meshes accordingly
    front_mesh.actor.property.emissive_factor = np.array([1., 1., 1.])
    front_mesh.actor.property.frontface_culling = True
    front_mesh.actor.property.opacity = 0.2

    rear_mesh.actor.property.emissive_factor = np.array([1., 1., 1.])
    rear_mesh.actor.property.backface_culling = True

    # Add filters
    clip = mlab.pipeline.user_defined(normals, filter='ClipPolyData')
    clip.filter.value = 0.5
    clip.filter.inside_out = True
    clip.add_child(module_manager)

    return meshes


def plot_base(mu, i1, theta_s, c=-1.0, base=None, **kwargs):
    """ Plots the 3d line that defines the bottom of the well where the partial derivative of the potential with
    respect the radius is zero.

    :param float mu: parameter that determines the depth of the potential (aka i0) along with `c`.
    :param float i1: amplitude of the input.
    :param float theta_s: orientation of the input.
    :param float c: parameter that determines the quartic term of the potential.
    :param base: a previously generated 3d line that will be modified with the new coordinates.
    :param kwargs: additional key-word arguments that control several aspects of the plot.
    :return: None or a mlab 3d plot.
    """
    logger.info('Plotting the line at the base of the potential...')
    r, th = r_roots(mu, i1, theta_s, i2=kwargs.get('i2', 0.0), c=c)
    x = r * np.cos(th)
    y = r * np.sin(th)
    zscale = kwargs.get('zscale', 4.0)
    z = zscale * potential(r, th, mu=mu, i1=i1, i2=kwargs.get('i2', 0.0), c=c, theta_s=theta_s, polar=True)
    if base is not None:
        base.mlab_source.reset(x=x, y=y, z=z)
    else:
        return mlab.plot3d(x, y, z, color=tuple(kwargs.get('line_color', (0.0, 0.0, 0.0))), tube_radius=None)


def plot_potential_profile(mu, i1, theta_s, rmax, c=-1.0, profile=None, **kwargs):
    """ Plots the 3d line corresponding to the vertical cut of the potential at the angular position `theta_s`.

    :param float mu: parameter that determines the depth of the potential (aka i0) along with `c`.
    :param float i1: amplitude of the input.
    :param float theta_s: orientation of the stimulus (if any) and also of the cut.
    :param float rmax: maximum radial distance to plot.
    :param float c: parameter that determines the quartic term of the potential.
    :param profile: a previously generated 3d line that will be modified with the new coordinates.
    :param kwargs: additional key-word arguments that control several aspects of the plot.
    :return: None or a mlab 3d plot.
    """
    logger.info(f"Plotting the profile of the potential at theta = {theta_s} degrees.")
    r = np.linspace(-rmax, rmax, 101)
    zscale = kwargs.get('zscale', 4.0)
    z = zscale * potential(r, np.deg2rad(theta_s), mu=mu, c=c, i1=i1, i2=kwargs.get('i2', 0.0), theta_s=theta_s,
                           polar=True)
    mask = z > kwargs.get('zmax', 0.5)
    z = z[~mask]
    r = r[~mask]
    x = r * np.cos(np.deg2rad(theta_s))
    y = r * np.sin(np.deg2rad(theta_s))
    if profile is not None:
        profile.mlab_source.reset(x=x, y=y, z=z)
    else:
        return mlab.plot3d(x, y, z, color=tuple(kwargs.get('line_color', (0.0, 0.0, 0.0))), tube_radius=None)


def plot_ball(theta_b, tstep=-1, r=(0,), ball=None, **kwargs):
    """ Plots (or updates) the 3d glyph that corresponds to the state of the network in the potential representation.

    :param int or float theta_b: current angular position of the ball.
    :param int tstep: time step that determines a particular radius of the ball given by `r`.
    :param tuple or list or np.ndarray r: a list of values for the radius of the position of the ball.
    :param ball: a previously generated 3d glyph that will be modified with the new coordinates.
    :param kwargs: additional key-word arguments that control several aspects of the plot.
    :return: None or a mlab 3d glyph.
    """
    logger.info(f"Plotting the ball...")
    logger.debug(f"\t... for parameters: {kwargs} ...")
    opacity = kwargs.pop('opacity', 1.0)
    if tstep == -1:
        bx, by, bz = get_ball_position(theta_ball=np.deg2rad(theta_b), **kwargs)
    else:
        logger.debug(f"\tat (r, theta_b) = ({r[tstep]}, {theta_b})")
        zscale = kwargs.get('zscale', 4.0)
        bx, by = (r[tstep] * np.cos(np.deg2rad(theta_b)), r[tstep] * np.sin(np.deg2rad(theta_b)))
        bz = zscale * potential(bx, by, polar=False, **kwargs)
    if ball is not None:
        ball.mlab_source.reset(x=bx, y=by, z=bz)
    else:
        ball = mlab.points3d(bx, by, bz, resolution=20, color=(0, 0, 0), scale_factor=0.25)
        ball.glyph.glyph_source.glyph_source.center = (kwargs.get('ball_xcenter', 0.0), 0, 0.55)
        ball.actor.property.color = tuple(kwargs.get('ball_color', (0.345, 0.7, 0.95)))
        ball.actor.property.opacity = opacity
        if opacity < 1.0 and kwargs.get('transparent', False):
            ball.actor.actor.force_opaque = True
        return ball


def snapshot(**kwargs):
    """ Function that draws a 3d potential. Unlike the :class:`Visualization`, this representation is static and
    does not support changes on the parameters once the plot is created. This function is extensively used to create
    some parts of the figures of the paper. In particular, panel A in figure 2 and panel B in figure 3.

    The snapshot admits several plots of the ball at different positions in order to give an idea of the dynamics
    of the amplitude equation.

    :param kwargs: key-word arguments that control the shape of the potential and its aesthetics.
    :return: Mayavi engine, lines corresponding to the base and vertical cut,
             glyphs representing the states of the network.
    """
    # Create the MayaVi engine and start it.
    e = Engine()
    # Starting the engine registers the engine with the registry and notifies others that the engine is ready.
    e.start()

    # Do this if you need to see the MayaVi tree view UI.
    EngineView(engine=e)

    mu = kwargs.get('mu', 1.0)
    i1 = kwargs.get('i1', 0.1)
    theta_s = kwargs.get('theta_s', 0.0)
    zmax = kwargs.get('zmax', 0.5)
    zscale = kwargs.get('zscale', 4.0)
    kwargs.update(mu=mu, i1=i1, theta_s=theta_s, zmax=zmax, zscale=zscale)

    plot_potential(**kwargs)  # Draw the potential surface(s)

    # Base line
    line1 = plot_base(**kwargs)
    line1.actor.property.line_width = kwargs.get('line_width', 2.0)

    # Potential profile (vertical cut)
    x, y, z, rmax = create_mesh(do_plot=False, **kwargs)
    kwargs.update(rmax=rmax)
    line2 = plot_potential_profile(**kwargs)
    line2.actor.property.line_width = kwargs.get('line_width', 2.0)

    # Central z-axis line
    if kwargs.get('rot_axis', False):
        arr_color = (0.22, 0.49, 0.78)
        arr_color = (0, 0, 0)
        z0 = np.arange(0, kwargs.get('zrange', [-2, 1])[1], 0.1)
        x0 = np.zeros_like(z0)
        mlab.plot3d(x0, x0, z0, color=arr_color, tube_radius=0.02, tube_sides=50)
        if kwargs.get('arrow', False):
            p2 = (0, 0, kwargs.get('zrange', [-2, 1])[1])
            arrow = mlab.points3d(*p2, mode='cone', color=arr_color)
            arrow.glyph.glyph_source.glyph_source.resolution = 50
            arrow.glyph.glyph_source.glyph_source.height = 0.2
            arrow.glyph.glyph_source.glyph_source.angle = 20
            arrow.glyph.glyph_source.glyph_source.direction = np.array((0, 0, 1.0))

    # Draw ball with or without movement representation
    balls = []
    theta_b = kwargs.get('theta_b', 0.0)
    theta_b2 = kwargs.get('theta_b2', theta_s)
    logger.debug(f"Computing steps between theta_ball 1 ({theta_b}) and theta_ball 2 ({theta_b2})...")
    angular_distance = np.angle(np.exp(1j * np.deg2rad(theta_b2)) / np.exp(1j * np.deg2rad(theta_b)))
    angular_distance = angular_distance * kwargs.get('ball_path', 1.0)
    logger.debug(f"\tangular distance is: {angular_distance}")
    nballs = kwargs.get('nballs', 1)
    center = kwargs.pop('center', False)  # Option that forces the ball to be at the center of the potential.
    const_arc = kwargs.pop('constant_arc', False)  # Forces the translation of the ball to have a fixed time-duration
    color_transition = kwargs.pop('color_transition', False)
    colors = ["#e24b34", "#b55c5b", "#876f85", "#5c80ab", "#02a5fc"]
    for k, c in enumerate(colors):
        colors[k] = hex_to_rgb(c, mode='1')
    opacity_bias = kwargs.pop('opacity_bias', 1.0)
    first_ball_color = kwargs.pop('first_ball', False)
    ball_color = colors[0]

    rballs, ball_xcenter, thetas = [], [], []
    if center:
        bx, by, _ = get_ball_position(theta_ball=np.deg2rad(theta_b), **kwargs)
        rbase = np.sqrt(bx ** 2 + by ** 2)  # Compute the equilibrium point at the bottom of the well
        rballs = np.linspace(0, rbase, nballs)  # radial positions of the ball from the center to the bottom
        # The (x, y, z)-position of the ball should be computed taking into account the normal to the surface.
        # This is a provisional solution that slightly moves the ball in the x-direction
        if nballs <= 5:
            ball_xcenter = [0.0, 0.16, 0.23, 0.25, 0.0]

    if const_arc:  # If we want to capture the displacement made by a stimulus of a given duration
        bx, by, _ = get_ball_position(theta_ball=np.deg2rad(theta_b), **kwargs)
        rbase = np.sqrt(bx ** 2 + by ** 2)
        _, thetas = psi_evo_r_constant((0, 0.300, nballs), r=rbase, i1=i1, theta=theta_s, psi0=theta_b)

    step_distance = int(np.rad2deg(angular_distance) / nballs) if nballs > 0 else 0
    logger.debug(f"\tstep distance is: {step_distance}")
    for n in range(nballs):
        # Draw ball
        step_theta_b = theta_b + step_distance * n
        if const_arc:
            step_theta_b = thetas[n]
        opacity = 1.0 / nballs * (n + 1) * opacity_bias
        opacity = 1.0 if opacity > 1.0 else opacity
        kwargs.update(theta_b=step_theta_b, opacity=opacity)
        if center:
            kwargs.update(tstep=n, r=rballs, ball_xcenter=ball_xcenter[n])
        if color_transition and nballs <= 5:
            kwargs.update(ball_color=colors[n])
        if first_ball_color:
            if n == 0:
                ball_color = kwargs.get('ball_color', colors[0])
                kwargs.update(ball_color=colors[0], opacity=0.5)
            elif n == 1:
                kwargs.update(ball_color=ball_color)
        if kwargs.get('noball', False):
            kwargs.update(opacity=0.0)
        ball = plot_ball(**kwargs)
        balls.append(ball)

    # Set the preferred view
    azimuth = kwargs.pop('azimuth', 30)
    mlab.view(azimuth=azimuth, elevation=kwargs.get('elevation', 70), focalpoint=kwargs.get('fp', [0, 0, -0.4]),
              distance=kwargs.get('dist', 6.0))
    scene = e.current_scene.scene
    # Set top view if requested
    if kwargs.get('top_view', False):
        mlab.view(azimuth=-90, elevation=0, focalpoint=[0, 0, -0.0], distance=7.0)
        if kwargs.get('parallel_projection', False):
            scene.parallel_projection = True

    # Clip the profile
    clipper = DataSetClipper()
    vtk_data_source = e.current_scene.children[2]  # should be the line2 data
    e.add_filter(clipper, vtk_data_source)
    clipper.add_child(vtk_data_source.children[0])  # Move the module manager to be a child of the filter
    # Modify the clip
    clipper.widget.widget_mode = 'ImplicitPlane'
    clipper.widget.widget.normal_to_z_axis = True
    clipper.filter.inside_out = True
    clipper.filter.use_value_as_offset = True
    clipper.filter.value = 0.5
    clipper.widget.widget.enabled = False

    # save the generate plot
    if kwargs.get('save', False):
        top_view = '_top' if kwargs.get('top_view', False) else ''
        center = '_center' if center else ''
        fig_name = f"./figs/potentials/fig_potential_mu-{mu:.2f}_i1-{i1:.2f}{top_view}_balls-{nballs}{center}.png"
        fig_name = kwargs.get('fig_name', fig_name)
        force = kwargs.pop('overwrite', False)
        auto = kwargs.pop('auto', False)
        logger.info(f"Saving file to '{fig_name}'...")
        fig_name = check_overwrite(fig_name, force=force, auto=auto)
        mlab.savefig(fig_name, magnification=1.0)

        if kwargs.get('transparent', False):  # Does not work as expected (the ball must be opaque)
            import matplotlib.pyplot as plt
            imgmap = mlab.screenshot(mode='rgba', antialiased=True)
            mlab.close()
            fig2 = plt.figure(figsize=(5, 5))
            fig2.patch.set_alpha(0.0)
            fig_name = fig_name[:-4] + '_alpha' + '.png'
            fig_name = check_overwrite(fig_name, force=force, auto=auto)
            plt.imsave(arr=imgmap, fname=fig_name, dpi=600)

    return e, [line1, line2], balls


init_options = dict(snapshot=False, theta_s=0, i1=0.0, mu=1.0, tmax=1.8, colormap='Reds', reverse_cmap=False,
                    mu_init=0.0, zmax=0.5, i1_init=0.0, db='INFO', theta_b=0, theta_b2=0, nballs=1,
                    figsize=(600, 600), auto=False, overwrite=False, data=False, zrange=(-1.2, 0.5),
                    ball_color=(0.345, 0.7, 0.95), line_color=(0.0, 0.0, 0.0), line_width=2.0, save=False,
                    top_view=False, gui=False, center=False, constant_arc=False, azimuth=30,
                    transparent=False, record=False, zscale=4.0, sigmaOU=0.2, i2=0.0, color_transition=False,
                    opacity_bias=1.0, first_ball=False, noball=False, rot_axis=False, elevation=70, fp=[0.0, 0.0, -0.4],
                    dist=6.0, arrow=False)

# To produce a snapshot with the ball falling down the center of the potential
# -colormap 'gist_gray' -zrange -2.00 1.0 -ball_color 0.89 0.29 0.2 -line_color 1 1 1 -line_width 5.0  -i1 0.00 -mu 1.0
# -azimuth 60 -gui -center -snapshot -nballs 5
# To produce a snapshot with the ball rolling down to the stimulus
# -colormap 'gist_gray' -zrange -2.00 1.0 -ball_color 0.89 0.29 0.2 -line_color 1 1 1 -line_width 5.0  -i1 0.1 -mu 1.0
# -azimuth 60 -gui  -snapshot -nballs 5 -theta_s 90 -theta_b 0 -theta_b2 70
# To produce a snapshot with the ball rolling down to the stimulus with custom colors (red and blue)
# -colormap 'gist_gray' -zrange -2.00 1.0 -ball_color 0.008 0.65 0.988 -line_color 1 1 1 -line_width 5.0  -i1 0.1
# -mu 1.0 -azimuth 60 -gui  -snapshot -nballs 5 -theta_s 90 -theta_b 0 -theta_b2 70 -first_ball

# To produce the potentials of Fig 3 (i1, i0) = ((0.1, 0.1, 0.2), (-0.1, 0.7, 1.5))
# python3 -W ignore mlab_potential.py  -colormap 'gist_gray' -zrange -1.50 1.0 -ball_color 0.008 0.65 0.988
# -line_color 1 1 1 -line_width 5.0   -azimuth 30 -gui  -snapshot -noball -theta_s 90 -zscale 1 -rot_axis -dist 8
# -i1 0.2 -mu 1.5 -save -transparent

if __name__ == '__main__':
    # Parsing, logging, debugging
    pars = Parser(desc='Ring network potential visualization.',
                  conf=init_options)
    conf_options = pars.opts
    logger, mylog = log_conf(pars.args.db, name='potential')
    init_options.update(conf_options)

    if pars.args.snapshot:
        logger.info('Plotting a snapshot...')
        if init_options.get('data', False):
            # load data
            datafile = init_options.get('datafile', 'simulation_data.csv')
            logger.info(f"Loading data from '{datafile}' ...")
            df = pd.read_csv(datafile)
            trial = init_options.get('trial', 150)
            data_trial = df.iloc[trial]
            stim_labels = [f"x{k}" for k in range(1, 9)]
            stim_orientations = data_trial[stim_labels].to_list()
            logger.debug(f"Stimulus orientations are: {stim_orientations}")
            phases_labels = [f"ph{k}" for k in range(1, 9)]
            phases = data_trial[phases_labels].to_list()
            phases = [0] + phases
            logger.debug(f"Phases are: {phases}")
            init_options['theta_s'] = 0.0
            init_options['theta_b'] = 0.0
            number_balls = init_options.get('nballs', 8)
            init_options['nballs'] = 1
            snapshot(**init_options)
            init_options['i1'] = 0.1
            init_options['nballs'] = number_balls

            for k, (stim, phase) in enumerate(zip(stim_orientations, phases)):
                init_options['theta_s'] = stim
                init_options['theta_b'] = phase
                init_options['theta_b2'] = phases[k + 1]
                snapshot(**init_options)
        else:
            eng, lines, ball0 = snapshot(**init_options)
        # Create a GUI instance and start the event loop.  We do this here so that
        # # main can be run from IPython -wthread if needed.
        if init_options.get('gui', False) and not init_options.get('transparent', False):
            gui = GUI()
            gui.start_event_loop()
    else:
        logger.info('Launching interactive visualization...')
        stim_theta = [50, 80, -30, 0, 40, -15, 90, -20]
        for key in ['i1', 'mu']:
            init_options.pop(key)
        v = Visualization(cut_planes=False, orientations=stim_theta, **init_options)
