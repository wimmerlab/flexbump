"""
Ring attractor network simulation library: auxiliary tools
==========================================================

.. module:: lib_ring
   :platform: Linux
   :synopsis: auxiliary library for the ring attractor network simulation

.. moduleauthor:: Jose M. Esnaola-Acebes <josemesnaola@gmail.com>


The Neural field equation is:

.. math::  \\tau dt/dr = -r + \\phi(r \\otimes w + x)

This library contains functions that numerically obtain theoretical results such as stability
related parameters (eigenvalues, critical points), bifurcation boundaries, normal form
approximations, etc.

It also contains functions that extract features related to the activity of the network, such as
the phase and the amplitude of the bump.


Auxiliary methods
-----------------

.. autosummary::
   :toctree: generated/

   circ_dist                       Pairwise difference around the circle.
   running_avg                     Computes the cumulative running average of a
                                    stream of inputs (real or complex).
   running_avg_circ                Computes the cumulative running average and
                                    variance of a vector of angles.
   generate_table                  Generates a table (dictionary) of all the
                                    useful parameters

Network architecture methods
----------------------------

.. autosummary::
   :toctree: generated/

   connectivity                    Creates the connectivity matrix.

Linear stability of the *SHS* methods
-------------------------------------

.. autosummary::
   :toctree: generated/

   lambda_k                        Computes eigenvalues of the *SHS*.
   r0f                             Rate of change of the firing rate.
   r0f_prima                       Derivative of the rate of change of the
                                    firing rate.
   fix_r0                          Computes the firing rate at the fix point of
                                    the *SHS*.
   lambda_1_i0                     Eigenvalue corresponding to k = 1.
   icritical                       Computes critical value of the external
                                         input, where the Bump is born.

Amplitude equation, methods            
---------------------------        
                                   
.. autosummary::                   
   :toctree: generated/            
                                   
   coefficient_c                   Computes the `c` coefficient of the amplitude
                                    equation.
   coefficient_a                   Computes the `a` coefficient of the amplitude
                                    equation
   r_roots                         Obtains the fixed points of R (mod of amp. eq.)
   potential                       Computes the potential associated to the amp. eq.
   psi_evo_r_constant              Computes the approximate evolution of the phase.
   amp_eq_simulation               Simulation of the evolution of the amplitude
                                    equation
                                   
Feature computation, methods           
----------------------------       
                                   
.. autosummary::                   
   :toctree: generated/            
                                   
   compute_phase                   Computes the phase of the bump.
   antiphase                       Given the phase, it computes the opposite
                                    phase.
   get_amplitude                   Given the phase, it computes the amplitude of
                                    the bump.
   get_phases_and_amplitudes_auto  Automatically identify the shape of the firing
                                    rate matrix and extract the phases and
                                    amplitudes.
   align_profiles                  Align the firing rate profiles, such that
                                    their phases are equal to zero.

Initial conditions, methods
---------------------------

.. autosummary::
   :toctree: generated/

   load_ic                         Loads the initial conditions of the network
                                    from previously saved data
   save_ic                         Saves the initial conditions for a given set
                                    of parameters.
   compute_hash                    Auxiliary function for :mod:`save_ic`

Implementation
--------------                                                                                             
                                                                                                           
.. todo::                                                                                                  
                                                                                                           
   Give a brief description about this library.

"""

import os
import sys

sys.path.append('./lib')
import logging
import numpy as np
import pandas as pd
import timeit
from easing import easing
from scipy.optimize import fsolve
from scipy.linalg import circulant
from scipy.stats import circmean
from lib_NeuroDyn import sigmoid_pw_v, sigmoid_pw_v_prima, ou_process
from lib_sconf import yesno_question, Parser

logging.getLogger('lib_ring').addHandler(logging.NullHandler())

__author__ = 'Jose M. Esnaola-Acebes'
__docformat__ = 'reStructuredText'

"""
Auxiliary methods                    
-----------------
"""


def circ_dist(x, y):
    """Pairwise difference x_i-y_i around the circle computed efficiently.

    References: Biostatistical Analysis, J. H. Zar, p. 651
    PHB 3/19/2009

    By Philipp Berens, 2009
    berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html

    :param x: sample of linear random variable.
    :param y:  sample of linear random variable or one single angle.
    :return: matrix with differences.
    """

    if np.shape(x) != np.shape(y) and np.size(y) != 1:
        raise ValueError(f"operands could not be broadcast together with shapes {np.shape(x)} {np.shape(y)}")
    return np.angle(np.exp(1j * x) / np.exp(1j * y))


def running_avg(x, dt=2E-4):
    """ Compute the running average of x.

    :param np.ndarray x: a vector or a 2d array of quantities that will be averaged, with dimensions (t, n_trials)
    :param float dt: integration time step.
    :return: a tuple with the time range, the modulus, the argument and the complex number of the cumulative running
             average.
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    x_avg = np.zeros_like(x, dtype=x.dtype)
    # Initial conditions
    x_avg[-1] = x[0]
    trange = np.arange(len(x)) * dt
    t = dt * 2
    for k, xn in enumerate(x[1:]):
        x_avg[k] = x_avg[k - 1] + dt * ((xn - x_avg[k - 1]) / t)
        t += dt
    x_avg = np.roll(x_avg, 1, axis=0)
    return trange, np.real(np.sqrt(x_avg * np.conjugate(x_avg))), np.rad2deg(np.angle(x_avg)), x_avg


def running_avg_circ(x, deg=False, **kwargs):
    """ Compute the running average and variance of x (radians by default) using the circular average function,
    :func:`circmean` and variance :func:`circvar`.

    :param np.ndarray x: a vector or a 2d array of quantities that will be averaged, with dimensions (t, n_trials)
    :param bool deg: whether to do the computations as degrees (by default are taken to be radians between -pi and pi).
    :param kwargs: keyword arguments to control the circular mean options.
    :return: the cumulative running average of the angles x
    :rtype: np.ndarray
    """
    circ_options = dict(low=-np.pi, high=np.pi)
    if deg is True:
        circ_options.update(low=-180, high=180)
    circ_options.update(**kwargs)
    x_avg = []
    x_var = []
    for k in range(1, len(x)):
        x_avg.append(circmean(x[0:k], **circ_options))
        x_var.append(circmean(x[0:k], **circ_options))

    return np.array(x_avg), np.array(x_var)


def generate_table(*args, **kwargs):
    """ Generates all the parameters used in the simulations and other meaningful quantities, and creates a table
    using dictionary syntax.

    :param str args: arguments passed to the Parser, using command line format (i.e. -arg1 value1, --arg2=value2).
    :param kwargs: other optional arguments.
    :return: a nested dictionary.
    :rtype: dict
    """
    # Load the parameters from the configuration file
    conf_file = kwargs.pop('conf', 'conf_paper.txt')
    pars = Parser(args=list(args), desc="Table of parameters", conf=conf_file)
    p = pars.args

    w0, w1, w2 = p.m
    # Amplitude equation and its derivation (critical values: values at the bifurcation point)
    icrit = icritical(p.m[0], w1, p.tau)
    noise_correction = 2 * p.sigmaOU ** 2
    r0 = fix_r0(icrit, w0, p.tau).ravel()[0]
    i0 = kwargs.get('i0', p.i0)
    i0range = np.array(kwargs.get('i0range', (0.01, 0.20)))
    # Derivatives of the transfer function
    sigmoid_p = kwargs.get('sigmoid_p', sigmoid_pw_v_prima)
    sigmoid_pp = kwargs.get('sigmoid_pp', sigmoid_pw_v_prima_prima)
    sigmoid_ppp = kwargs.get('sigmoid_ppp', sigmoid_pw_v_prima_prima_prima)

    # Values of the input and derivatives of the transfer function at the bifurcation point
    x0 = p.tau * w0 * r0 + icrit
    phi0p = sigmoid_p(x0, tau=p.tau)
    phi0pp = sigmoid_pp(x0, tau=p.tau)
    phi0ppp = sigmoid_ppp(x0, tau=p.tau)

    # Ring model (network parameters, time constant, ...)
    ring = dict(tau=("Time constant of the network", r"$\tau$", p.tau * 1000, 'ms'),
                w_k=("Fourier modes of the connectivity", r"$w_k$, {\small $k=0,1,2$}", p.m, 'a.u.'),
                I_exc=("Global net excitatory drive", r"$I_\mathrm{exc}$", tuple(icrit + i0range), 'a.u.'),
                xi=("Noise term", r"$\xi \left(\theta, t\right)$", '---', 'a.u.'),
                sigma=("Noise amplitude", r"$\sigma_\mathrm{OU}$", p.sigmaOU, 'a.u.'),
                tauOU=("Noise time constant", r"$\tau_\mathrm{OU}$", p.tauOU * 1000, 'ms'),
                Phi=("Transfer function", r"$\Phi \left(\cdot\right)$", '---', 'a.u.'),
                icrit=("Critical excitatory drive", r"$I_\mathrm{crit}$", icrit, 'a.u.'),
                r0=(r"Stationary FR at $I_\mathrm{exc} = I_\mathrm{crit}$", r"$r_{\infty}$", r0, 'Hz'))

    # Stimulus
    stim = dict(I_1=("Stimulus amplitude", r"$I_1$", p.i1, 'a.u.'),
                T=("Duration of the stimulus", r"$T$", (p.nframes * p.cue_duration), 's'),
                N=("Number of stimulus frames", r"$N$", p.nframes, ''))

    a = coefficient_a(w0, i0range - noise_correction, icrit, tau=p.tau)
    c = coefficient_c(w0, w2, icrit, tau=p.tau)
    tilde_i1 = 1.0 / (p.tau * w1) * p.i1 / 2.0

    # Get the theoretical amplitude of the bump
    # lambda0 = 1 - p.tau * w0 * phi0p
    # lambda2 = 1 - p.tau * w2 * phi0p
    # rsol = r_roots(mu=a, i1=tilde_i1, c=c, theta=0)[0][0]
    # r_max = r0 + 2 * rsol + phi0p / lambda0 * (i0 - noise_correction) \
    #         + phi0pp * (lambda2 + lambda0) / (phi0p ** 2 * lambda0 * lambda2) * rsol ** 2

    ampeq = dict(I_0=(r"Additional excitatory drive, $I_\mathrm{exc} - I_\mathrm{crit}$", r"$I_0$",
                      tuple(i0range), 'a.u.'),
                 phi0p=(r"Slope of $\Phi(\cdot)$ at $I_\mathrm{exc}=I_\mathrm{crit}$", r"$\Phi_\infty'$",
                        phi0p, 'a.u.'),
                 phi0pp=(r"2nd derivative of $\Phi(\cdot)$ at $I_\mathrm{exc}=I_\mathrm{crit}$", r"$\Phi_\infty''$",
                         phi0pp, 'a.u.'),
                 phi0ppp=(r"3rd derivative of $\Phi(\cdot)$ at $I_\mathrm{exc}=I_\mathrm{crit}$", r"$\Phi_\infty'''$",
                          phi0ppp, 'a.u.'),
                 a=("1st order prefactor", r"$\tilde{I}_0$", tuple(a), 'a.u.'),
                 c=("Cubic prefactor", r"$c$", -c, 'a.u.'),
                 i1tilde=("Scaled stimulus amplitude ", r"$\tilde{I}_1$", tilde_i1, 'a.u.'),
                 xi=("Noise terms", r"$\xi_i \left(t\right)$,\ i=1,2", '---', 'a.u.'))

    # Simulation ...
    simu = dict(dt=("Integration time step", r"$\Delta t$", p.dt * 1E3, 'ms'),
                n=("Spatial discretization", r"$n$", p.n, ''),
                k=("Number of trials", r"$K$", p.ntrials, ''))

    # Table
    table = dict(ring=('Ring network ' + r"(\Cref{eq:ring-eq})", ring), stim=('Stimulus', stim),
                 ampeq=('Amplitude equation ' + r"(\Cref{eq:amp-eq})", ampeq),
                 simu=('Numerical integration', simu))

    return table


"""
Network architecture methods 
----------------------------
"""


def connectivity(n, modes):
    """Creates a (``n`` x ``n``) connectivity matrix for the ring neural field model.

    :param int n: number of rate neurons/population.
    :param collections.Iterable[float] modes: amplitude of the modes of connectivity ordered from 0 to n_modes.
    :return: connectivity matrix.
    :rtype: np.ndarray
    """
    [i_n, j_n] = np.meshgrid(range(n), range(n))
    ij = (i_n - j_n) * (2.0 * np.pi / n)

    jphi = 0
    for k, mode in enumerate(modes):
        if k == 0:
            jphi = mode
        else:
            jphi += 2.0 * mode * np.cos(k * ij)

    return jphi


def add_noise(cnt, sigma=1, max_difference=0.1):
    """Adds noise to the connectivity matrix but preserves Fourier modes (almost) (that is the aim...)"""
    # Compute the modes
    from scipy.linalg import eigvals
    modes = (eigvals(cnt) / len(cnt))[::2]  # Take only even eigenvalues
    new_cnt = cnt + sigma**2 * np.random.randn(*cnt.shape)
    new_modes = (eigvals(new_cnt) / len(new_cnt))[::2]
    diff = np.abs(new_modes - modes)
    if np.all(diff >= 0.1):
        logging.warning(f"Difference between modes is larger "
                        f"than {max_difference} for modes {np.argwhere(diff > max_difference).ravel()}")
    return new_cnt


def gaussian_connectivity(gee, gie, gii, gei, sigma, ne, **kwargs):
    """Creates a (2``ne`` x 2``ne``) connectivity matrix using circular gaussian functions.
    
    :param float gee: strength of excitation to excitatory neurons.
    :param float gie: strength of excitation to inhibitory neurons.
    :param float gii: strength of inhibition to inhibitory neurons.
    :param float gei: strength of inhibition to excitatory neurons.
    :param float sigma: width of the Gaussian connection profiles.
    :param int ne: number of excitatory neurons.
    :param kwargs: additional keyword arguments.
    :return: connectivity matrix.
    :rtype: np.ndarray
    """

    ni = kwargs.pop('ni', ne)
    if ni != ne:
        raise ValueError(f"Number of excitatory ({ne}) and inhibitory ({ni}) neurons must be equal.")
    theta = np.arange(ne) / ne * (2 * np.pi) - np.pi  # Preferred cue of each exc. neuron
    v = np.exp(-circ_dist(theta, -np.pi) ** 2 / (2.0 * sigma ** 2))  # Recurrent connectivity for the first neuron
    cnt_e = np.concatenate((gee / ne * circulant(v), -gei / ni * circulant(v)))
    cnt_i = np.concatenate((gie / ne * circulant(v), -gii / ni * circulant(v)))
    return np.array([cnt_e.T, cnt_i.T]).reshape(2 * ne, 2 * ne)


"""
Linear stability of the *SHS* methods 
-------------------------------------
"""


def lambda_k(wk, r0, w0, i0, tau=0.02, **kwargs):
    """ Compute the eigenvalue of wavenumber 'k' of the neural field equation at the spatially homogeneous state.

    :param float wk: k-th Fourier coefficient of the connectivity profile.
    :param np.ndarray of float r0: firing rate at the spatially homogeneous state
    :param float w0: 0-th Fourier coefficient of the connectivity profile.
    :param np.ndarray of float i0: constant external current.
    :param float tau: time constant.
    :param kwargs: keyword arguments passed to the transfer function.
    :return: eigenvalue associated to the k-th mode.
    :rtype: np.ndarray of float
    """

    kwargs['tau'] = tau
    return -1.0 / tau + wk * sigmoid_pw_v_prima(tau * w0 * r0 + i0, **kwargs)


def r0f(x, i0, w0, tau):
    """ Dynamical equation of the spatially homogeneous state, to be used with :py:func:`scipy.optimize.fsolve`.

    :param np.ndarray of float x: Firing rate of the homogeneous state.
    :param np.ndarray of float i0: constant external current.
    :param float w0: 0-th Fourier coefficient of the connectivity profile.
    :param float tau: time constant.
    :return: :math:`dx/dt`.
    :rtype: np.ndarray of float
    """
    return -x + sigmoid_pw_v(tau * w0 * x + i0, tau=tau)


def r0f_prima(x, i0, w0, tau):
    """ Derivative of the Dynamical equation of the spatially homogeneous state.

    :param np.ndarray of float x: Firing rate of the homogeneous state.
    :param np.ndarray of float i0: constant external current.
    :param float w0: 0-th Fourier coefficient of the connectivity profile.
    :param float tau: time constant.
    :return: :math:`\\frac{d^2 x}{dt^2}`.
    :rtype: np.ndarray of float
    """
    return -1.0 + tau * w0 * sigmoid_pw_v_prima(tau * w0 * x + i0, tau=tau)


def fix_r0(i0, w0, tau=0.02, **kwargs):
    """ Computes the fix point of the spatially homogeneous state.

    :param float i0: constant external current.
    :param float w0: 0-th Fourier coefficient of the connectivity profile.
    :param float tau: time constant.
    :return: possible stationary firing rates.
    :rtype: np.ndarray of float
    """

    initial_r = kwargs.get('init_r', [0.01, 10.0, 50.0, 100.0, 1E5])
    rsols = []
    for r_0 in initial_r:
        rsol, infod, ier, mesg = fsolve(r0f, r_0, args=(i0, w0, tau), fprime=r0f_prima, full_output=True)
        if ier == 1:
            if not np.any(np.isclose(np.array(rsols), rsol)):
                rsols.append(rsol)
    return np.array(rsols)


def lambda_1_i0(i0, w0, w1, tau=0.02, **kwargs):
    """Computes the eigenvalue corresponding to the 1st wavenumber given a input x.

    :param float i0: constant external current.
    :param float w0: 0-th Fourier coefficient of the connectivity profile.
    :param float w1: 1st Fourier coefficient of the connectivity profile.
    :param float tau: time constant.
    :param kwargs: additional arguments to be passed to fix_r0 or lambda_k
    :return: eigenvalue lambda_1
    :rtype: np.ndarray of float
    """

    r0 = fix_r0(i0, w0, tau=tau, **kwargs)
    r00 = r0.flatten()[0]
    l1 = lambda_k(w1, r00, w0, np.array(i0), tau=tau, **kwargs)
    return l1


def icritical(w0, w1, tau=0.02, **kwargs):
    """ Computes the critical value of the external input I for which the system undergoes a Turing
    bifurcation.

    :param float w0: 0-th Fourier coefficient of the connectivity profile.
    :param float w1: 1st Fourier coefficient of the connectivity profile.
    :param float tau: time constant.
    :return: critical external current.
    :rtype: float
    """
    iinit = np.array(kwargs.get('iinit', [0.0]))
    icr = fsolve(lambda_1_i0, np.array(iinit), args=(w0, w1, tau))
    return icr[0]


"""
Amplitude equation, methods
---------------------------
"""

from lib_NeuroDyn import sigmoid_pw_v_prima_prima, sigmoid_pw_v_prima_prima_prima


def coefficient_c(w0, w2, icr, tau=0.02, **kwargs):
    """ Computes the constant coefficient of the 3rd order term of the amplitude equation (normal form of
    the super-critical Turing bifurcation).

    :param float w0: 0-th Fourier coefficient of the connectivity profile.
    :param float w2: 2nd Fourier coefficient of the connectivity profile.
    :param float icr: value of the constant external current at the bifurcation point.
    :param float tau: time constant.
    :return: c coefficient.
    :rtype: float
    """

    kwargs['tau'] = tau
    r0 = fix_r0(icr, w0, **kwargs)
    r00 = r0.flatten()[0]
    x0 = tau * w0 * r00 + icr
    phi1 = sigmoid_pw_v_prima(x0, **kwargs)
    phi2 = sigmoid_pw_v_prima_prima(x0, **kwargs)
    phi3 = sigmoid_pw_v_prima_prima_prima(x0, **kwargs)
    try:
        return phi2 ** 2 / phi1 ** 3 * (
                tau * w0 / (1.0 - tau * w0 * phi1) + tau * w2 / (2.0 * (1.0 - tau * w2 * phi1))) + phi3 / (
                       2.0 * phi1 ** 3)
    except ZeroDivisionError:
        return np.inf


def coefficient_a(w0, i0_over, icr, tau=0.02, **kwargs):
    """ Computes the constant coefficient of the 1st order term of the amplitude equation (normal form of
    the super-critical Turing bifurcation).

    :param float w0: 0-th Fourier coefficient of the connectivity profile.
    :param float i0_over: additional spatially uniform external input injected to
                          the network (I = I_cr + epsilon^2 * I0).
    :param float icr: value of the constant external current at the bifurcation point.
    :param float tau: time constant.
    :return: a coefficient.
    :rtype: float
    """

    kwargs['tau'] = tau
    r0 = fix_r0(icr, w0, **kwargs).flatten()[0]
    x0 = tau * w0 * r0 + icr
    phi1 = sigmoid_pw_v_prima(x0, **kwargs)
    phi2 = sigmoid_pw_v_prima_prima(x0, **kwargs)

    try:
        return phi2 * i0_over / (phi1 * (1.0 - tau * w0 * phi1))
    except ZeroDivisionError:
        return np.inf


def r_roots(mu=1.0, i1=0.1, theta_s=0.0, theta_s2=90, c=-1.0, i2=0.0, **kwargs):
    """ Obtain the fixed points of the modulus of the amplitude equation (R) by solving the third order
    polynomial. It can also compute the radius (modulus of the amp. eq.) at a specific theta(s).

    :param float mu: first order coefficient.
    :param float i1: input coefficient (first order).
    :param float theta_s: input angle.
    :param float theta_s2: phase of the second mode of the input.
    :param float c: third order coefficient (should be negative).
    :param float i2: input coefficient (second order).
    :param kwargs: additional key-word arguments.
    :return: solutions of r and theta (angle).
    :rtype: (np.ndarray, np.ndarray)
    """
    n_points = kwargs.get('n_points', 101)
    theta_s2 = np.deg2rad(theta_s2)
    if i1 != 0.0:
        theta = kwargs.get('theta', np.linspace(-np.pi / 2, np.pi / 2, n_points))
        if not isinstance(theta, (list, np.ndarray)):
            theta = np.array([theta])
        p, q = (mu / c + i2 / c * np.cos(2 * (theta - theta_s2)), i1 / c * np.cos(theta))
        # Check number of roots
        discr = -(4*p**3 + 27*q**2)
        if np.all(discr < 0):  # Single root at the input orientation
            logging.debug(f"Value of the discriminant is always < 0.")
            theta = np.ones_like(theta) * np.deg2rad(theta_s)
            p, q = (mu / c + i2 / c * np.cos(2 * (theta - theta_s2)), i1 / c * np.cos(theta - np.deg2rad(theta_s)))
        d0, d1 = (-3 * p, 27 * q)
        carg1 = (d1 + np.sqrt(0j + d1 ** 2 - 4 * d0 ** 3)) / 2.0
        carg2 = (d1 - np.sqrt(0j + d1 ** 2 - 4 * d0 ** 3)) / 2.0
        carg = carg1[:]
        carg[carg1 == 0] = carg2[carg1 == 0]
        c0 = carg[:]
        c0[carg > 0] = carg[carg > 0] ** (1 / 3)
        c0[carg < 0] = -(-carg[carg < 0]) ** (1 / 3)
        chi = (-1 + np.sqrt(-3 + 0j)) / 2
        xk = []
        for k in range(3):
            xk.append(-1 / 3 * (chi ** k * c0 + d0 / (chi ** k * c0)))

        if np.all(discr < 0):
            rsol = np.abs(np.real(np.concatenate((xk[0], xk[0]))))
            theta_sol = np.concatenate((theta, theta))
        else:
            # Concatenate solutions (number of points are doubled)
            end = 0
            if np.any(discr < 0):
                xk = np.array(xk)
                prob = np.argwhere(discr < 0)
                xk[2, prob] = np.ones_like(xk[2, prob]) * xk[2, prob][0]
                theta2 = theta.copy() + np.pi
                theta2[prob] = np.ones_like(theta2[prob]) * theta2[prob][0]
                # Roll the vectors such that the last valid solution is at the end of the vector
                end = prob[0]
            else:
                theta2 = theta.copy() + np.pi
            rsol = np.abs(np.real(np.concatenate((xk[0], xk[2]))))
            theta_sol = np.concatenate((theta, theta2)) + np.deg2rad(theta_s)
            if end != 0:
                rsol = np.roll(rsol, end)
                theta_sol = np.roll(theta_sol, end)
    else:
        theta_sol = kwargs.get('theta', np.linspace(0, 2 * np.pi, n_points)) + np.deg2rad(theta_s)
        if not isinstance(theta_sol, (list, np.ndarray)):
            theta_sol = np.array([theta_sol])
        if mu >= 0:
            rsol = np.sqrt((mu + i2 * np.cos(2 * (theta_sol - theta_s2))) / -c)
        else:
            rsol = theta_sol * 0
    return rsol, theta_sol


# noinspection PyUnusedLocal
def potential(x, y, mu=1.0, c=-1.0, i1=0.0, i2=0.0, theta_s=0.0, theta_s2=90, polar=False, **kwargs):
    """Computes the potential given (x, y) cartesian coordinates, with parameters
    `mu`, `i1` and `theta_s`. If the optional argument `polar` is true, the coordinates (x,y)
    are assumed to be given in polar form.

    :param x: x-cartesian coordinates (ideally obtain using numpy.mgrid[startx:stopy:Nxj, starty:stopy:Nyj]).
    :param y: y-cartesian coordinates (same size as x or scalar).
    :param float mu: controls the depth and radius of the circular well.
    :param float c: parameter that controls the amplitude of the quartic term (should be negative).
    :param float i1: controls the strength of the stimulus by biasing the circular well towards `theta_s`.
    :param float i2: second spatial mode of the stimulus.
    :param flaot theta_s: stimulus orientation.
    :param flaot theta_s2: orientation of the second spatial mode of the stimulus.
    :param bool polar: whether the coordinates are assumed to be of polar form.
    :param kwargs: additional keyword arguments that are not used.
    :return: potential scalar value of size x (or y).
    """

    if not polar:
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
    else:
        r, theta = (x, y)
    theta_s = np.deg2rad(theta_s)
    theta_s2 = np.deg2rad(theta_s2)
    return -(mu + i2 * np.cos(2.0 * (theta - theta_s2))) / 2.0 * r ** 2 - c / 4.0 * r ** 4 \
           - r * i1 * np.cos(theta - theta_s)


# noinspection PyUnusedLocal
def perfect_potential(x, y, i1=0.0, theta_s=0.0, polar=False, **kwargs):
    """Computes the potential given some (x, y) cartesian coordinates, with parameters
    `i1` and `theta_s`. If the optional argument `polar` is true, the coordinates (x,y)
    are assumed to be given in polar form.

    :param x: x-cartesian coordinates (ideally obtain using numpy.mgrid[startx:stopy:Nxj, starty:stopy:Nyj]).
    :param y: y-cartesian coordinates (same size as x or scalar).
    :param float i1: controls the strength of the stimulus by biasing the circular well towards `theta_s`.
    :param flaot theta_s: stimulus orientation.
    :param bool polar: whether the coordinates are assumed to be of polar form.
    :param kwargs: additional keyword arguments that are not used.
    :return: potential scalar value of size x (or y).
    """

    if not polar:
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
    else:
        r, theta = (x, y)
    theta_s = np.deg2rad(theta_s)
    return - r * i1 * np.cos(theta - theta_s)


def pvi_simulation():
    pass


def psi_evo_r_constant(trange=(0, 5, 1000), tau=0.02, r=1.0, i1=0.1, theta=90.0, psi0=0.0):
    """ Approximate evolution of the phase, assuming that the amplitude of the bump is constant.

    :param tuple trange: time range of the evolution defined as (initial, last, number of time steps).
    :param float tau: time constant.
    :param float r: 'amplitude' of the bump.
    :param float i1: scaled amplitude of the input.
    :param float theta: orientation of the stimulus in degrees.
    :param float psi0: initial position of the bump (initial phase), in degrees.
    :return: time steps, evolution of the phase
    :rtype: (np.ndarray, np.ndarray)
    """
    t = np.linspace(*trange)
    theta = np.deg2rad(theta)
    psi0 = np.deg2rad(psi0)
    arg = theta - 2 * np.arctan(np.tan((theta - psi0) / 2.0) * np.exp(-i1 / tau / r * t))
    return t, np.rad2deg(arg)


# noinspection PyPep8Naming
def amp_eq_simulation(translate=False, smooth=False, save_evo=True, **kwargs):
    """ Simulate the time-evolution of the amplitude equation using euler integration. The function admits raw values
    of the parameters as in equation (??) or meaningful parameters that match those used with the ring equation.

    The `smooth` option (False by default) smoothens the transitions of the dynamics, useful for the visualization
    of the "particle on the potential".

    :param bool translate: translates meaningful parameters (used with the ring model) into the amplitude equation.
    :param bool smooth: activates smoothing of input transitions and parameter changes.
    :param bool save_evo: whether the entire evolution must be saved or not.
    :param kwargs: additional key-word arguments that may contain the values of the parameters an other simulation
                   related parameters, such as number of stimulus frames, orientations duration, etc.
    :return: time-points, amplitude modulus and phase, depth of the potential (mu), stimulus amplitude and phase, raw
             complex amplitude variable.
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """

    # Model parameters
    modes = kwargs.get('modes', [-2.0, 1.0, 0.5])
    tau = kwargs.get('tau', 0.02)
    i0_over = kwargs.get('mu', 1.0)
    i0_over = kwargs.get('i0', i0_over)
    i1 = kwargs.get('i1', 0.1)
    icrit = icritical(modes[0], modes[1], tau=tau)
    if translate:  # Interpret the parameters as those in the ring model, and compute the amp.eq. parameters from them
        a = coefficient_a(modes[0], i0_over, icrit, tau)
        c = coefficient_c(modes[0], modes[2], icrit, tau)
        pre_d = 1.0 / 2.0 * 1.0 / tau / modes[1]  # See Comment 3.1 in :doc:`notes.pdf`, except for prefactor 1/2
        d = pre_d * i1
    else:
        a = i0_over
        c = kwargs.get('c', -1)
        pre_d = 1.0
        d = i1

    dt = 2E-4
    tmax = kwargs.get('tmax', 2.2)
    nframes = kwargs.get('nframes', 8)
    init_time, post_time = kwargs.get('init_time', 0.0), kwargs.get('post_time', 0.0)
    its, pts = int(init_time / dt), int(init_time / dt)
    frame_duration = kwargs.get('tframe', (tmax - init_time - post_time) / nframes)
    frame_duration = kwargs.get('cue_duration', frame_duration)
    orientations = kwargs.get('orientations', np.random.randint(0, 180, nframes))
    sigmaou = kwargs.get('sigmaOU', 0.15)

    # Setup arrays, number of trials is given by the orientations matrix
    oshape = np.shape(orientations)
    if len(oshape) > 1:
        ntrials = oshape[1]
    else:
        ntrials = 1
    tpoints = np.arange(0, tmax, dt)
    nsteps = len(tpoints)
    rsteps = nsteps if (save_evo or ntrials == 1) else 2
    if ntrials > 1:
        r = np.ones((rsteps, ntrials)) * 0.0001
    else:
        r = np.ones_like(tpoints) * 0.0001
    phi = np.zeros_like(r)
    A = np.ones_like(r, dtype=np.complex) * 0.0

    # Noise and noise-induced correction
    if sigmaou or kwargs.get('correction', 0):
        noise = np.zeros((nsteps, ntrials, 2))
        noise[its:] = ou_process(dt, nsteps - its, 0.0, sigmaou, 0.001, ntrials, 2)
        noise_1 = np.sqrt(2) * pre_d / np.sqrt(200) * (noise[:, :, 0] + 1j * noise[:, :, 1])
        if translate:
            noise_0 = coefficient_a(modes[0], -kwargs.get('correction', 2 * sigmaou ** 2), icrit, tau)
        else:
            noise_0 = -kwargs.get('correction', 0)
        noise_steps = nsteps * 1
    else:
        noise_1 = np.zeros(2)
        noise_0 = 0.0
        noise_steps = 2

    # Initial state (the one corresponding to the ring model simulations are A[0] = 0)
    rbase0 = r_roots((a + noise_0), i1=d, c=c, theta=0)[0][0]
    rbase1 = np.sqrt((a + noise_0) / -c) if (a + noise_0) >= 0 else 0
    rbase = (rbase0 + rbase1) / 2
    r[-1] = kwargs.get('rinit', rbase)
    if kwargs.get('rinit', rbase) == -1:
        r[-1] = 2 * sigmaou * np.random.randn(ntrials)
    if kwargs.get('rinit', rbase) == -2:  # This initial condition takes into account the effect of the input
        r[-1] = rbase0
    r[0] = r[-1]
    phi[-1] = kwargs.get('theta_0', np.deg2rad(np.random.randint(-180, 180, size=ntrials)))
    if kwargs.get('theta_0', 0) == -1:
        phi[-1] = np.deg2rad(orientations[0])

    phi[0] = phi[-1]
    A[-1] = r[-1] * np.exp(1j * phi[-1])
    A[0] = r[0] * np.exp(1j * phi[0])

    # Set-up stimulus
    i1_v = np.ones(rsteps) * d
    if ntrials > 1:
        ts_v = np.zeros((nsteps, ntrials))
    else:
        ts_v = np.zeros(nsteps)
    if its:
        ts_v[0:its] = 0.0
        if rsteps == nsteps:
            i1_v[0:its] = 0.0
    for nframe, orientation in enumerate(orientations):  # orientations: nframes x ntrials
        t0 = int((init_time + frame_duration * nframe) / dt)
        t1 = int((init_time + frame_duration * (nframe + 1)) / dt)
        ts_v[t0:t1] = np.deg2rad(orientation)

    # Smooth the transitions for better visualization
    if smooth:
        # Change in the depth of the potential, from a paraboloid to a "squeezer" shaped potential.
        if init_time != 0:
            mu_points = int(tmax / init_time)
            mu = np.ones(mu_points) * a
            mu[0] = kwargs.pop('mu_init', 0.0)
            mu = easing.Eased(mu).power_ease(2, int(len(tpoints) / mu_points))
        else:
            mu = np.ones(rsteps) * a

        # Smooth the input changes, both amplitude and phase
        eased_vars = easing.Eased(np.array([i1_v[::100], ts_v[::100]]).T).power_ease(2, 100)
        i1_v, ts_v = eased_vars[:, 0], eased_vars[:, 1]
    else:
        mu = np.ones(2) * a
    mu_steps = len(mu)

    # Initialize simulation variables
    # i1_v[-1] = d
    ts_v[-1] = phi[-1]
    tstep, temps = (0, 0)
    ttau = dt / tau

    # Simulate
    time_init = timeit.default_timer()
    while temps < tmax - dt:
        k = (tstep + rsteps - 1) % rsteps
        kp = tstep % rsteps
        m = tstep % mu_steps
        n = tstep % noise_steps
        t = tstep % nsteps

        # Complex amplitude
        A[kp] = A[k] + ttau * ((mu[m] + noise_0) * A[k] + i1_v[k] * np.exp(1j * ts_v[t]) + noise_1[n]
                               + c * (A[k] * np.conj(A[k])) * A[k])
        tstep += 1
        temps += dt
    # Stop the timer
    r = np.real(np.sqrt(A * A.conjugate()))
    phi = np.angle(A)
    logging.debug('Total time: {}.'.format(timeit.default_timer() - time_init))
    if kwargs.get('full_stim', False):
        stim = noise_1.ravel() + i1_v * np.exp(1j * ts_v)
        i1_v = np.real(np.sqrt(stim * np.conjugate(stim)))
        ts_v = np.angle(stim)
    return tpoints, r, phi, mu + noise_0, i1_v, ts_v, A


"""
Feature computation, methods
----------------------------
"""


def compute_phase(x, n, c=(0.0,), s=(0.0,), wavenumber=1):
    """ Function that gives an approximate value of the phase of a spatial profile by projecting the vector x
    into a cosine function.

    :param Cython.Includes.numpy.ndarray x: firing rates matrix (or vector) with shape (d1, d2, ..., n).
    :param int n: spatial dimension (spatial discretization).
    :param np.ndarray of float c: cosine function, precompute it for faster computation.
    :param np.ndarray of float s: sine function, precompute it for faster computation.
    :param int wavenumber: wavenumber of the corresponding phase that we want to compute.
    :rtype: np.ndarray of float
    """

    # Check external cosine and sine functions
    if len(c) != n or len(s) != n:
        c = np.cos(np.arange(n) * wavenumber * 2.0 * np.pi / n - np.pi)
        s = np.sin(np.arange(n) * wavenumber * 2.0 * np.pi / n - np.pi)

    # norm of sine and cosine functions depends on n (discretization of the space)
    return np.arctan2((np.dot(x, s) / (n / 2.0) ** 2), (np.dot(x, c) / (n / 2.0) ** 2))


def antiphase(phase):
    """ Get the opposite phase of 'phase'

    :param Cython.Includes.numpy.ndarray phase: vector of phases of the bump.
    :rtype: np.ndarray of float
    """

    return ((phase + 2 * np.pi) % (2 * np.pi)) - np.pi


def get_amplitude(r, phases, theta_matrix):
    """ Get the amplitude(s) of the firing rate profiles 'r' at 'phases'

    :param np.ndarray of float r: (n_trials x n_orientations) sized matrix of firing rate. Alternatively,
                                   the function also accepts a matrix of size (tsteps x n_trials x n_orientations).
    :param np.ndarray of float phases: vector of phases with size ``n_trials``. Alternatively, the function also
                                       accepts a matrix of size (``tsteps`` x ``n_trials``).
    :param np.ndarray of float theta_matrix: (``n_orientations`` x ``n_trials``) sized matrix containing the phases
                         in :math:`[\\pi, \\pi)`.
                         It should be a matrix equal to ``np.repeat(np.array([theta]), n_trials, axis=0).T``,
                         where ``theta = np.arange(n_orientations) / n_orientations * (2 * np.pi) - np.pi``.
                         Note: if r has size (tstep x n_trials x n_orientations) then this matrix must have
                         size (n_orientations x n_steps x n_trials), which can be obtained with
                         ``np.tile(theta, (ntrials, tsteps, 1)).T``.
    :return: vector of size '`n_trials`' with the firing rate amplitudes. Alternatively, it will return a
             matrix of size (``tsteps`` x ``n_trials``).
    :rtype: np.ndarray of float
    """

    rshape = r.shape
    n = rshape[-1]
    idx = (np.abs(theta_matrix - phases)).argmin(axis=0).ravel()
    r_r = r.reshape(len(idx), n)
    return r_r[range(len(idx)), idx].reshape(rshape)


def get_phases_and_amplitudes_auto(r, aligned_profiles=False):
    """ Get the phases (radians) and amplitude(s) of the firing rate profiles 'r' at those phases automatically
    detecting shapes of the matrices.

    :param Cython.Includes.numpy.ndarray r: matrix of firing rates. Last axis must be the spatial
                                            axis, i.e. ``r.shape[-1] = n``.
    :param bool aligned_profiles: whether to return the firing rates with their phases aligned to 0.
    :return: tuple of arrays containing (firing rates , phases,  firing rate amplitudes at phases,
                                         and firing rate amplitudes at phases + pi)
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """

    rshape = r.shape
    n = rshape[-1]
    theta = np.arange(n) / n * (2 * np.pi) - np.pi
    # Compute the phases
    phases = compute_phase(r, n)  # dim: rshape[0:-1]
    theta_matrix = np.moveaxis(np.tile(theta, rshape[0:-1] + (1,)), -1, 0)  # dim: n x rshape[0:-1]
    # Look for the indexes that correspond to those phases (and antiphases)
    idx = (np.abs(theta_matrix - phases)).argmin(axis=0).ravel()
    idx_anti = (np.abs(theta_matrix - antiphase(phases))).argmin(axis=0).ravel()
    r_r = r.reshape(len(idx), n)  # Reshaping the matrix makes everything easier

    if aligned_profiles:  # Get aligned firing rate profiles if necessary
        new_r = np.array([np.roll(r_row, -idx_row + n // 2) for r_row, idx_row in zip(r_r, idx)]).reshape(rshape)
    else:
        new_r = r
    amplitudes = r_r[range(len(idx)), idx].reshape(rshape[0:-1])
    amplitudes_m = r_r[range(len(idx)), idx_anti].reshape(rshape[0:-1])
    return new_r, phases, amplitudes, amplitudes_m


def align_profiles(r):
    """ Get an aligned firing rate matrix where all phases are set to 0.

    :param np.ndarray of np.ndarray r: (rsteps, ntrials, n) sized matrix of firing rates.
    :return: (rsteps, ntrials, n) sized matrix of realigned firing rates.
    :rtype: np.ndarray of float
    """

    (rsteps, ntrials, n) = r.shape
    theta = np.arange(n) / n * (2 * np.pi) - np.pi
    theta_matrix = np.repeat(np.array([theta]), ntrials, axis=0).T
    new_r = np.ones(r.shape)

    for k, rprofile in enumerate(r):
        phases = compute_phase(np.array(rprofile), n, np.cos(theta), np.sin(theta))
        idx = (np.abs(theta_matrix - phases)).argmin(axis=0)
        # TODO: This takes a long time, it should be faster using advance indexing.
        new_r[k] = np.array([np.roll(row, -idx_row + n // 2) for row, idx_row in zip(rprofile, idx)])

    return new_r


"""
Initial condition storing methods
---------------------------------
"""


# noinspection PyTypeChecker
def load_ic(db_file, critical=(None,), **kwargs):
    """Load simulation initial conditions from binary ``db_file`` based on a set of
    parameters (``kwargs``) and critically required parameters ``critical``. Previously saving initial conditions
    with :func:`save_ic` guarantees that the format of the database is correct.

    :param str db_file: Path to database. The database must be a pickled python dictionary with at least the following
                        keys: ``registry`` (where the combination of parameters are stored), ``hashrule`` (where
                        the rule to compute the ``hash`` is located) and ``ic`` (where the actual data of the initial
                        conditions is stored). See :func:`save_ic` for more details.
    :param tuple of str or tuple of None critical: list or tuple with the names of the critical
                                                   parameters in ``kwargs``.
    :param kwargs: keywords containing the name and value of the relevant parameter of the initial
                   condition database. Use the same order as in :func:`save_ic`.
    :return: initial conditions stored at database ``db_file``. If the process fails, returns False.
    :rtype: bool or np.ndarray
    """
    logging.debug('Requested parameters for initial condition loading: %s' % kwargs)
    # Load the database
    db_file, extension = os.path.splitext(db_file)
    try:
        db = dict(np.load(db_file + '.npy', allow_pickle=True, encoding='latin1').item())
    except IOError:
        logging.error('Initial condition database not found or not accessible.')
        return False

    registry = db['registry']
    ic_db = db['ic']
    hash_rule = db.get('hashrule')
    allprmts = kwargs.copy()
    # Get the critical parameter(s) if any, and check
    if not isinstance(critical, (list, tuple)):
        critical = [critical]
    if isinstance(critical, tuple):
        critical = list(critical)
    critical_values = [kwargs.pop(crit, None) for crit in critical]
    registry = registry.loc[np.all(registry[critical] == critical_values, axis=1)]
    if registry.empty:  # Combination of critical parameters not found
        logging.warning('Critical parameters requirement not met. Initial condition loading interrupted.')
        return False

    t_lbls = list(kwargs.keys())  # Get the rest of the arguments
    t_prmts = list(kwargs.values())

    t_hash = compute_hash(hash_rule, list(allprmts.values()))  # Compute hash of target parameters
    target_ic = registry.loc[registry.hash == t_hash]  # type: pd.DataFrame
    if not target_ic.empty:  # Target parameters found
        if target_ic.shape[0] > 1:  # Check that a single set was obtained
            target_ic = target_ic.iloc[0]
            logging.warning('Registry contains non unique hash values.')  # Raise a warning but do nothing
        if not np.all(target_ic[t_lbls].eq(t_prmts)):  # Check that parameters are indeed the same
            logging.warning('Registry may contain non unique hash values.')  # Raise a warning but do nothing
        if isinstance(target_ic, pd.DataFrame):  # Make sure that the target database is a pd.Series
            target_ic = target_ic.iloc[0]  # type: pd.Series
    else:  # Find similar set of parameters
        target_ic = registry.loc[registry.hash.apply(lambda x: np.abs(float(x) - float(t_hash))).idxmin(axis=0)]

    logging.debug('Selected parameters for the initial condition loading: %s' % target_ic[t_lbls].to_dict())
    # Load data from ic database
    return ic_db[target_ic['id']]


# noinspection PyTypeChecker
def save_ic(db_file, initial_conditions, hashrule=None, force=False, **kwargs):
    """Save simulation's results in binary format to be loaded as initial conditions in future simulations using
    :func:`load_ic`. The simulation parameters that determine the initial state are given as separated keyword
    arguments through ``kwargs``. The ``hashrule`` tries to ensure that different parameter combinations give
    a unique hash number (experimental).

    :param str db_file: Path to file containing database.
    :param np.ndarray initial_conditions: Data to be stored as initial condition of the simulation.
    :param None or collections.Iterable[tuple] hashrule: see :func:`compute_hash` to get an idea.
    :param bool force: force overwriting without confirmation.
    :param kwargs: keyword arguments with the name and corresponding value of the parameters.
    :return: 0 if saving was done without errors. 1 if saving was not done.
    :rtype: int
    """
    logging.debug('Parameters for initial condition saving: %s' % kwargs)
    # Try to load the database
    db_file, extension = os.path.splitext(db_file)
    try:
        db = dict(np.load(db_file + '.npy', allow_pickle=True, encoding='latin1').item())
    except IOError:  # File does not exist: create new database
        logging.debug('Initial condition database not found or not accessible.')
        logging.debug('Creating new database at {db_file}.')
        registry = pd.DataFrame(kwargs, index=[1])
        if hashrule is None:
            hashrule = np.repeat([np.array([8, 3])], len(kwargs), axis=0)
        registry['hash'] = compute_hash(hashrule, list(kwargs.values()))
        registry['id'] = 1
        db = dict(registry=registry, hashrule=hashrule, ic={1: initial_conditions})
        np.save(db_file, db, allow_pickle=True)
        return 0

    logging.debug('Searching for initial conditions with the same parameters.')
    hashrule, registry, ic_db = db['hashrule'], db['registry'], db['ic']
    t_hash = compute_hash(hashrule, list(kwargs.values()))
    target_ic = registry.loc[registry['hash'] == t_hash]  # type: pd.DataFrame

    if not target_ic.empty:  # Parameter combination found in registry (probably)
        if target_ic.shape[0] > 1:  # Multiple entries have the same hash-value
            logging.error('Database has bad format. Repeated hash values found.')
            target_ic = target_ic.iloc[0]
        logging.warning('These parameters have been previously registered.')
        if not force:
            force = yesno_question('Do you want to overwrite previous results?', default_answer='Y')
        if force:
            logging.debug('Overwriting initial conditions.')
            ic_db[target_ic.iloc[0]['id']] = initial_conditions
        else:
            logging.debug('Initial conditions not saved.')
            return 1
    else:  # Add new line to registry
        kwargs.update({'id': registry.shape[0] + 1, 'hash': t_hash})
        registry = registry.append(pd.DataFrame(kwargs, index=[registry.shape[0] + 1]), sort=False)
        ic_db[kwargs['id']] = initial_conditions

    db['hashrule'], db['registry'], db['ic'] = hashrule, registry, ic_db
    np.save(db_file, db, allow_pickle=True)
    return 0


def compute_hash(hashrule, values):
    t_hash = [f"{value:0{h[0]}.{h[1]}f}".replace('.', '').replace('-', '') for h, value in zip(hashrule, values)]
    t_hash = ''.join(t_hash)
    t_hash = t_hash[0:len(t_hash) // 2] + '.' + t_hash[len(t_hash) // 2:]
    return t_hash
