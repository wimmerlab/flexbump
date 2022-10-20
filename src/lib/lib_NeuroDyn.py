"""
Neural dynamics library
=======================

.. module:: lib_NeuroDyn
   :platform: Linux
   :synopsis: a miscellaneous of neural activity related methods.

.. moduleauthor:: Jose M. Esnaola-Acebes <josemesnaola@gmail.com>


Noisy signals
-------------

.. autosummary::
   :toctree: generated/

   ou_process   Generates Ornstein-Uhlenbeck processes.

Distributions
-------------

.. autosummary::
   :toctree: generated/

   lorentz      Generator of deterministic Lorentzian samples.
   gauss        Generator of deterministic Gaussian samples.

Input-output transfer functions and derivatives
-----------------------------------------------

.. autosummary::
   :toctree: generated/

   sigmoid_ww   Wong-Wang (2006) transfer function.
   sigmoid_pw   Piece-wise transfer function.


Implementation
--------------

.. todo::

   Implementation notes.
"""

import numpy as np
import logging
from scipy import stats
from scipy.special import erfcx
from scipy.signal import lfilter

logging.getLogger('lib_NeuroDyn').addHandler(logging.NullHandler())

__author__ = 'Jose M. Esnaola-Acebes'
__docformat__ = 'reStructuredText'


"""
Noisy signals
-------------
"""


def ou_process(dt, nsteps, mu, sigma, tau, trials=1, n_spatial=1, seed=None):
    """ Generates a good approximation of a single(or trials) Ornstein-Uhlenbeck process(es)
        for a single(oumodes) units.

    :param float dt: integration time-step.
    :param int nsteps: number of time steps.
    :param float mu: offset of the processs.
    :param float sigma: standard deviation of the process.
    :param float tau: time scale of the process (< dt).
    :param int trials: number of processes to be generated (trials).
    :param int n_spatial: number of spatially distributed units or modes of Fourier.
    :param int or None seed: seed for the random number generator.
    :return: time series: array (nsteps x trials x oumodes).
    :rtype: np.ndarray
    """
    # Constants
    a = np.exp(-dt / tau)
    b = mu * (1.0 - a)
    c = sigma * np.sqrt(1.0 - a ** 2)

    if seed is not None:
        np.random.seed(seed)

    s = lfilter(np.ones(1), [1, -a], b + c * np.random.randn(nsteps, trials, n_spatial), axis=0)
    return s


"""
Distributions
-------------
"""


def lorentz(n, center, width):
    """Obtain a vector of distributed values deterministically computed from the inverse cumulative 
    `Lorentz` (Cauchy) distribution function.
    
    :param int n: number of evenly distributed elements to be computed.
    :param float center: center of the Cauchy distribution. 
    :param float width: width of the Cauchy distribution.
    :return: a vector of evenly distributed values.
    :rtype: np.ndarray of float 
    """
    k = (2.0 * np.arange(1, n + 1) - n - 1.0) / (n + 1.0)  # Avoid extreme values (1 or -1)
    y = center + width * np.tan((np.pi / 2.0) * k)
    return y


def gauss(n, center, width):
    """Obtain a vector of distributed values deterministically computed from the inverse cumulative 
    Gaussian distribution function.

    :param int n: number of evenly distributed elements to be computed.
    :param float center: center of the Cauchy distribution. 
    :param float width: width of the Cauchy distribution.
    :return: a vector of evenly distributed values.
    :rtype: np.ndarray of float 
    """
    k = (np.arange(1, n + 1)) / (n + 1.0)
    y = center + width * stats.norm.ppf(k)
    return y


"""
Input-Output transfer functions
-------------------------------
"""


def sigmoid(x):
    """Roxin, Ledberg 2008
    
    :param np.ndarray of float x: input current.
    :return: firing rate response.
    :rtype: np.ndarray of float 
    """
    alpha = 1.5
    beta = 2.5
    i0 = 1.0
    return alpha / (1 + np.exp(-beta * (x - i0)))


def sigmoid_abi(x, alpha=1.0, beta=1.0, i0=30.0):
    """Wong-Wang (2005??) input to frequency transfer function. Optional parameters.

    :param np.ndarray of float x: input current.
    :param float alpha: maximum amplitude.
    :param float beta: gain.
    :param float i0: gain offset.
    :return: firing rate response.
    :rtype: np.ndarray of float
    """
    return alpha / (1 + np.exp(-beta * (x - i0)))


def ssn_f(x):
    """Input-output function (Rubin et al. 2015)

    :param np.ndarray of float x: input current.
    :return: firing rate response.
    :rtype: np.ndarray of float
    """
    return 0.04 * np.maximum(x, 0)**2


def sigmoid_brunel_hakim(x):
    """Brunel-Hakim (???) input to frequency transfer function.

    :param np.ndarray of float x: input current.
    :return: firing rate response.
    :rtype: np.ndarray of float 
    """
    return 1 + np.tanh(x)


# noinspection PyUnusedLocal
def sigmoid_qif(x, tau=1.0, delta=1.0, **kwargs):
    """Transfer function corresponding to a heterogeneous all-to-all QIF population.
    See MPR 2015.
    
    :param np.ndarray of float x: input current.
    :param float tau: membrane time constant.
    :param float delta: level of heterogeneity.
    :param kwargs: dummy keyword arguments, with no effect.
    :return: firing rate response.
    :rtype: np.ndarray of float
    """
    return (1.0 / (tau * np.pi * np.sqrt(2.0))) * np.sqrt(x + np.sqrt(x * x + delta * delta))


# Transfer function of an noisy (sigma) all-to-all LIF population
def sigmoid_lif(x, tau=1.0, sigma=1.0, vr=-68.0, vth=-48.0, vrevers=-50.0, dt=10E-3):
    """Transfer function of an noisy all-to-all LIF population
    See Roxin et al (2006??).
     
    :param float x: input current.
    :param float tau: membrane time constant.
    :param float sigma: noise amplitude.
    :param float vr: reset potential.
    :param float vth: threshold potential.
    :param float vrevers: reverse potential.
    :param float dt: integration step.    
    :return: firing rate response.
    :rtype: float
    """
    x1 = (vr - vrevers - x) / sigma
    x2 = (vth - vrevers - x) / sigma
    dx = (x2 - x1) / 2000.0
    x = np.arange(x1, x2 + dx, dx, dtype=np.float64)
    fx = erfcx(-x)
    t = tau * np.sqrt(np.pi) * np.sum(fx * dx) + dt * tau
    return 1.0 / t


# Wong and Wang 2006 Transfer function
# noinspection PyUnusedLocal
def sigmoid_ww(x, a=270, b=108, d=0.154, **kwargs):
    """Wong and Wang (2006) transfer function, scalar version.
    
    :param float x: input current
    :param float a: 
    :param float b: 
    :param float d: 
    :param kwargs: dummy keyword arguments, with no effect. 
    :return: firing rate response.
    :rtype: float
    """
    if isinstance(x, np.ndarray):
        return sigmoid_ww_v(x)
    m = a * x - b

    if m == 0 or np.isnan(m) or np.isnan(x):
        return 1.0 / d
    else:
        return m / (1.0 - np.exp(-d * m))


# noinspection PyUnusedLocal
def sigmoid_ww_v(x, a=270, b=108, d=0.154, **kwargs):
    """Wong and Wang (2006) transfer function, vectorial version.
    
    :param np.ndarray of float x: input current
    :param float a: 
    :param float b: 
    :param float d: 
    :param kwargs: dummy keyword arguments, with no effect. 
    :return: firing rate response.
    :rtype: np.ndarray of float
    """
    if isinstance(x, float):
        return sigmoid_ww(x)
    m = a * x - b
    mask_zero = ((m == 0.0) | (np.isnan(m)) | (np.isnan(x)))
    if np.isinf(x).any():
        logging.error('The input exploded!')
        m[np.isinf(x)] = 0.0
        return m * 0.0
    mask_large = (x > 1.0)
    mask_small = (x < -1.0)
    normal = ~(mask_zero | mask_large | mask_small)
    m[normal] = m[normal] / (1.0 - np.exp(-d * m[normal]))
    m[mask_small] = 0.0
    m[mask_zero] = 1.0 / d
    return m


def sigmoid_pw_v(x, a=0.0000, **kwargs):
    """Piece-wise transfer function, vectorial version.
    
    :param np.ndarray of float x: input current
    :param float a: optional minimal firing rate (default = 0)
    :param kwargs: processed keyword arguments (``tau``, ``gamma``).
    :return: firing rate response.
    :rtype: np.ndarray of float
    """
    if not isinstance(x, (list, np.ndarray)):
        return sigmoid_pw(float(x), a, **kwargs)
    tau = kwargs.get('tau', 1.0)
    gamma = kwargs.get('gamma', 1.0)
    phi = x * 1.0
    # Conditions:
    m1 = (x <= 0.0)
    m3 = (x >= 1.0)
    m2 = ~(m1 | m3)
    # Function
    phi[m1] = a
    phi[m2] = a + x[m2] ** 2
    phi[m3] = a + 2.0 * np.sqrt(x[m3] - 3.0 / 4.0)
    return phi * gamma / tau


def sigmoid_pw(x, a=0.0000, **kwargs):
    """Piece-wise transfer function, scalar version.

    :param float x: input current
    :param float a: optional minimal firing rate (default = 0)
    :param kwargs: processed keyword arguments (``tau``, ``gamma``).
    :return: firing rate response.
    :rtype: float
    """
    if isinstance(x, (list, np.ndarray)):
        return sigmoid_pw_v(x, a, **kwargs)
    tau = kwargs.get('tau', 1.0)
    gamma = kwargs.get('gamma', 1.0)
    if x <= 0.0:
        return a * gamma / tau
    elif 0.0 < x < 1.0:
        return (a + x ** 2.0) * gamma / tau
    else:
        return (a + 2.0 * np.sqrt(x - 3.0 / 4.0)) * gamma / tau


def sigmoid_pw_v_prima(x, a=0.0000, **kwargs):
    """Derivative of the piece-wise transfer function :fun:`sigmoid_pw_v`, vectorial version.

    :param np.ndarray of float x: input current
    :param float a: optional minimal firing rate (default = 0)
    :param kwargs: processed keyword arguments (``tau``, ``gamma``).
    :return: firing rate response.
    :rtype: np.ndarray of float
    """
    if not isinstance(x, (list, np.ndarray)):
        return sigmoid_pw_prima(float(x), a, **kwargs)
    tau, gamma = kwargs.get('tau', 1.0), kwargs.get('gamma', 1.0)
    phi, (m1, m2, m3) = x * 1.0, get_mask(x)
    # Function
    phi[m1], phi[m2], phi[m3] = 0.0, 2.0 * x[m2], 1.0 / np.sqrt(x[m3] - 3.0 / 4.0)
    return phi * gamma / tau


def sigmoid_pw_prima(x, a=0.0000, **kwargs):
    """Derivative of the piece-wise transfer function :fun:`sigmoid_pw`, scalar version.

    :param float x: input current
    :param float a: optional minimal firing rate (default = 0)
    :param kwargs: processed keyword arguments (``tau``, ``gamma``).
    :return: firing rate response.
    :rtype: float
    """
    if isinstance(x, (list, np.ndarray)):
        return sigmoid_pw_v_prima(x, a, **kwargs)
    tau = kwargs.get('tau', 1.0)
    gamma = kwargs.get('gamma', 1.0)
    if x <= 0.0:
        return 0.0
    elif 0.0 < x < 1.0:
        return 2.0 * x * gamma / tau
    else:
        return (1.0 / np.sqrt(x - 3.0 / 4.0)) * gamma / tau


def sigmoid_pw_v_prima_prima(x, a=0.0000, **kwargs):
    """Second derivative of the piece-wise transfer function :fun:`sigmoid_pw_v`, vectorial version.

    :param np.ndarray of float x: input current
    :param float a: optional minimal firing rate (default = 0)
    :param kwargs: processed keyword arguments (``tau``, ``gamma``).
    :return: firing rate response.
    :rtype: np.ndarray of float
    """
    if not isinstance(x, (list, np.ndarray)):
        return sigmoid_pw_prima_prima(float(x), a, **kwargs)
    tau, gamma = kwargs.get('tau', 1.0), kwargs.get('gamma', 1.0)
    phi, (m1, m2, m3) = x * 1.0, get_mask(x)
    # Function
    phi[m1], phi[m2], phi[m3] = 0.0, -2.0, -1.0 / (2.0 * (x[m3] - 3.0 / 4.0)) ** (3.0 / 2.0)
    return phi * gamma / tau


def sigmoid_pw_prima_prima(x, a=0.0000, **kwargs):
    """Second derivative of the piece-wise transfer function :fun:`sigmoid_pw`, scalar version.

    :param float x: input current
    :param float a: optional minimal firing rate (default = 0)
    :param kwargs: processed keyword arguments (``tau``, ``gamma``).
    :return: firing rate response.
    :rtype: float
    """
    if isinstance(x, (list, np.ndarray)):
        return sigmoid_pw_v_prima_prima(x, a, **kwargs)
    tau = kwargs.get('tau', 1.0)
    gamma = kwargs.get('gamma', 1.0)
    if x <= 0.0:
        return 0.0
    elif 0.0 < x < 1.0:
        return 2.0 * gamma / tau
    else:
        return (-1.0 / (2.0 * (x - 3.0 / 4.0) ** (3.0 / 2.0))) * gamma / tau


def sigmoid_pw_v_prima_prima_prima(x, a=0.0000, **kwargs):
    """Third derivative of the piece-wise transfer function :fun:`sigmoid_pw_v`, vectorial version.

    :param np.ndarray of float x: input current
    :param float a: optional minimal firing rate (default = 0)
    :param kwargs: processed keyword arguments (``tau``, ``gamma``).
    :return: firing rate response.
    :rtype: np.ndarray of float
    """
    if isinstance(x, float):
        return sigmoid_pw_prima_prima_prima(x, a, **kwargs)
    tau, gamma = kwargs.get('tau', 1.0), kwargs.get('gamma', 1.0)
    phi, (m1, m2, m3) = x * 1.0, get_mask(x)
    # Function
    phi[m1], phi[m2], phi[m3] = 0.0, 0.0, 3.0 / (4.0 * (x[m3] - 3.0 / 4.0)) ** (5.0 / 2.0)
    return phi * gamma / tau


def sigmoid_pw_prima_prima_prima(x, a=0.0000, **kwargs):
    """Third derivative of the piece-wise transfer function :fun:`sigmoid_pw`, scalar version.

    :param float x: input current
    :param float a: optional minimal firing rate (default = 0)
    :param kwargs: processed keyword arguments (``tau``, ``gamma``).
    :return: firing rate response.
    :rtype: float
    """
    if isinstance(x, (list, np.ndarray)):
        return sigmoid_pw_v_prima_prima_prima(x, a, **kwargs)
    tau = kwargs.get('tau', 1.0)
    gamma = kwargs.get('gamma', 1.0)
    if x <= 0.0:
        return 0.0
    elif 0.0 < x < 1.0:
        return 0.0
    else:
        return (3.0 / (4.0 * (x - 3.0 / 4.0) ** (5.0 / 2.0))) * gamma / tau


def get_mask(x):
    """Get mask for the piece-wise transfer functions.

    :param np.ndarray of float x: input currents.
    :return: masks
    :rtype: (np.ndarray of bool, np.ndarray of bool, np.ndarray of bool)
    """
    # Conditions:
    m1 = (x <= 0.0)
    m3 = (x >= 1.0)
    m2 = ~(m1 | m3)
    return m1, m2, m3
