"""
Ring attractor simulation. Analysis library (:mod:`lib_analysis`)
=================================================================

.. currentmodule:: lib_analysis
   :platform: Linux
   :synopsis: module for analyzing simulated data from the ring attractor network.

.. moduleauthor:: Jose M. Esnaola-Acebes <josemesnaola@gmail.com>

This library contains functions to process and analyze the data obtained through the scripts:
 * :mod:`ring_fast_frames`
 * :mod:`ring_fast_frames_bifurcation`
 * :mod:`ring_simulation`
 * Any other script that produces similar data frames.

The library contains two type of methods, those intended for statistical analysis and those that
produce figures with the processed data. The methods that are described as *(Incorrect)* are correctly
implemented but their theoretical background is not completely correct. For example the methdo that
computes the logistic regression assumes that the average orientation is computed linearly (arithmetic
mean) instead of the circular mean.

Analytical methods
------------------

.. autosummary::
   :toctree: generated/

   log_reg             Logistic regression over the binary decision
                        making task data. *(Incorrect)*
   log_reg2d           2-dimensional logistic regression over the
                        binary decision making task data. *(Incorrect)*
   noise_ppk           Psychophysical kernel that measure the impact
                        of fluctuations around the equilibrium state. *(Incorrect)*
   compute_estimation  Compute the estimation curve given a data-frame.
   compute_performance Gives a (bad) estimate of the performance. *(Incorrect)*
   compute_lin_reg     Computes linear regression by means of OLS.
   compute_circ_reg_by Computes the circular regression by means of Bayesian sampling.
   circ_reg_model      Model of the stimulus estimation, and its log-likelihood.
   circ_probit_model   Model of the stimulus categorization, and its log-likelihood.
   compute_circ_reg_ml Computes the circular regression by means of the max.likelihood method.
   probit_circular     Computes the binary circular regression by means of the max.likelihood method.
   compute_pm_reg      Computes the psychometric function using :func:`probit_circular`.
   compute_pm_data     Computes the psychometric data-based points.
   get_amp_eq_stats    Computes the estimates and the psychophysical kernell for the amplitude equation.
   measure_delta_ang   Measures the impact of the stimulus on the phase of the bump using the
                        amplitude equation formalism.
   compute_ppk_slope   Compute the slope of the psychophysical kernel.
   slope_vs_time       Computes the slope of the PK for different number of frames of the
                        same duration.
   compute_sqr_error   Computes the estimation squared error given the PK weights. It also computes
                        the error of its corresponding weighted circular average (without noise).
   get_sqr_errors      High level function that computes necessary and missing statistics in order to
                        compute the estimation errors using :func:`compute_sqr_error`.

Plotting methods
----------------

.. autosummary::
   :toctree: generated/

   plot_pm             Psychometric function
   plot_estimation     Estimation function
   plot_estimation_avg Average estimation function
   plot_ppk            Psychophysical kernel
   plot_2dppk          2-dimensional Psychophysical kernel
   plot_bifurcation    Bifurcation diagram
   plot_bifurcation_aq Bifurcation diagram of the amplitude equation.
   plot_bump_profiles  Bifurcation diagram and firing rate profiles
   plot_profiles_comp  Comparison between firing rate profiles
   plot_trial          Single trial summary.
   plot_stats_summ     Summary plot with PM, PK and estimation curves.
   plot_rp             Plots recency-primacy figure
   plot_decision_circ  Plots dynamics of the decision circuit.
   plot_response_time  Plots statistics related to the response times.
   plot_rt_summ        Response times for different input currents.
   plot_temp_integ     Integration regimes for different input currents,
                        includes PM, Estimation and PPK.
   plot_bump_vs_nobump Comparison of integration regimes for bump and
                        no-bump conditions.
   plot_trial_ic       Comparison of bump vs. no-bump conditions with
                        a single trial.
   plot_trial_ic_two   Same as :func:`plot_trial_ic` version 2.
   plot_multiple_phase Comparison of bump vs. no-bump condition with
                        phase evolution of multiple trials.
   plot_trial_full     Summary of a single trial, for the full ciruit.
   plot_ring_vs_amp    Plots estimation, PK and PM data of both the ring model
                        and the amplitude equation for different I0


Auxiliary methods
-----------------

.. autosummary::
   :toctree: generated/

   round_general        Rounds a float number or array.
   normalize_data       Normalizes the simulated data
   load_data            Loads and pre-processes the simulated data
   save_plot            Simple auxiliary saving method.
   equalbins            Creates an equally populated binning vector for the given
                         data vector using quantiles.
   get_all_files       Creates a data-frame with the data-files and their parameters for easier
                        access in other functions.

Implementation
--------------

.. todo::

   Give a brief description about the data-frames that are use in the different functions in this library.
"""
import logging
from operator import itemgetter
from typing import Union
import pathlib

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import pandas as pd
import statsmodels.api as sm
import statsmodels.tools.sm_exceptions as sm_exceptions
from scipy.stats import logistic, circmean, vonmises, circstd
from scipy.interpolate import interp2d
from scipy.special import i0 as besseli
from scipy.optimize import minimize
import re
from collections.abc import Iterable

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
except ImportError:
    logging.error("Some functions cannot be used without R.")

from lib_plotting import *
from lib_ring import align_profiles, get_phases_and_amplitudes_auto, circ_dist, psi_evo_r_constant, running_avg
from lib_ring import amp_eq_simulation as simulation
from lib_sconf import load_obj, check_overwrite, MyLog, save_obj

logging.getLogger('lib_analysis').addHandler(logging.NullHandler())

__author__ = 'Jose M. Esnaola-Acebes'
__docformat__ = 'reStructuredText'

"""
Auxiliary methods                    
-----------------
"""


def round_general(f, up_down='up', even_odd='even', dtype=np.int64):
    """ Function that rounds up/down ``f`` to the closest even/odd number

    :param np.ndarray of float f: n-dim array of numbers to be rounded to integers.
    :param str up_down: whether to round ``'up'``, ``'down'`` or to the closest integer.
    :param str even_odd: whether to round to the closest ``'even'``, ``'odd'`` or any closes integer.
    :param type dtype: desired type of the returned values.
    :return: n-dim array of rounded integers.
    :rtype: np.ndarray of type
    """

    fr = np.round(f) if up_down is None else np.ceil(f) if up_down == 'up' else np.floor(f)
    if even_odd is not None:
        m = np.array(fr % 2 if even_odd == 'even' else (fr + 1) % 2, dtype=bool)  # Identify even/odd values
        fr[m] = fr[m] + ((2 * ((fr - f) <= 0)[m] - 1) if up_down is None else 1 if up_down == 'up' else -1)
    return np.cast[dtype](fr)


def normalize_dataframe(data, copy=False, clean=True, **kwargs):
    """ Post-processing of the dataframe. Normalize and clean the data.

    :param pd.DataFrame data: dataframe containing simulation results.
    :param bool copy: whether to do a copy of the data (no use, probably). (default: False)
    :param bool clean: whether to clean the data, which means removing columns with labels ``clean_labels``.
                       (default is True)
    :param kwargs: Additional keyword arguments, such as ``clean_labels`` or ``normalize_labels``.
    :return: the modified dataframe and the new labels of the stimuli.
    :rtype: (:py:class:`pandas.DataFrame`, list of str)
    """

    if copy:
        data = data.copy()
    normalize_labels = kwargs.get('normalize_labels', ['average', 'average_circ', 'category', 'estim'])
    for label in normalize_labels:  # Normalize the results to [-1, 1]
        data['norm_%s' % label] = data[label] / 90
    # Create columns with normalized [-1, 1] data
    labels = []
    for col in data.columns:
        if re.fullmatch("x*[0-9][0-9]?", col):  # Only accept columns of the type xXX, where X = 0, 1, 2, ..., 9
            labels.append('%s_norm' % col)
            data[labels[-1]] = data[col] / 90  # Normalize the frame orientations

    # Clean the data
    if clean:
        clean_labels = kwargs.get('clean_labels', ['index', 'sub_trial', 'Trial', 'bincorrect'])
        try:
            data = data.drop(columns=clean_labels)
        except KeyError:
            logging.debug("Cleaning the data-frame failed...")
    return data, labels


def load_data(filename, directory='.', num_trials=160000, load_prmts=False, load_rates=False):
    """Load the data, normalize it and clean it.

    :param str filename: name of the file containing the data
    :param str directory: directory where the data-file is (not necessary if full path is given).
    :param int num_trials: maximum number of trials to be loaded.
    :param bool load_prmts: whether to also load the associated parameters.
    :param bool load_rates: whether to also load the associated firing rate data.
    :return: :py:mod:`pandas` dataframe containing the results of the simulation, labels pointing to the normalize
             design matrix. In case of selecting ``load_prmts=True``, or/and ``load_rates=True`` a dictionary with
             the simulation configuration and the firing rates (if any) will be returned.
    """
    filename, ext = os.path.splitext(filename)
    simu_data = {}
    if load_prmts or load_rates:
        logging.info(f"Loading data from {os.path.join(directory, filename)}.npy ...")
        simu_data = load_obj(filename, directory, extension='.npy')
        if simu_data is False:
            logging.error('Loading process failed.')
            return -1

        df = simu_data['data']
    else:
        path = os.path.join(directory, filename + '.csv')
        logging.info(f"Loading data from {path} ...")
        try:
            df = pd.read_csv(path)
        except IOError:
            logging.error('Loading process failed.')
            return -1

    df, labels = normalize_dataframe(df, copy=False, clean=True)
    df = df.loc[~np.isnan(df.estim.astype(float))]

    # TODO: select the data evenly among the different categories.
    if len(df) > num_trials:
        df = df.sample(num_trials)
    logging.debug(df.head())

    if load_prmts and load_rates:
        if np.any(simu_data.get('rates', False)):
            return df, labels, simu_data['conf'], simu_data['rates']
        else:
            logging.error("This data file does not have the firing rate data. It is probably old.")
            return df, labels, simu_data['conf'], []
    elif load_prmts:
        return df, labels, simu_data['conf']
    elif load_rates:
        if simu_data.get('rates', False):
            return df, labels, simu_data['rates']
        else:
            logging.error("This data file does not have the firing rate data. It is probably old.")
            return df, labels, []
    else:
        return df, labels


def save_plot(fig, filename, **kwargs):
    """Saves the figure ``fig`` at ``filename`` in different formats. Checks overwriting.

    :param plt.Figure fig: figure to be saved.
    :param str filename: saving path.
    :param kwargs: additional keyword arguments.
    """
    logging.info(f"Saving figure with path '{filename}'.")
    saveformat = kwargs.pop('saveformat', ['png', 'pdf', 'svg', 'fig'])
    if not isinstance(saveformat, (list, tuple)):
        saveformat = [saveformat]
    for k, fmt in enumerate(saveformat):
        print("\x1b[1A", end='')
        logging.info(f"Saving figure with path '{filename}.{fmt}'.")
        print("\x1b[1A" + "\x1b[1C" + " " * 4 + "\x1b[4D" + "▓" * (k + 1) * (8 // 4))
        filename_ext = check_overwrite(filename + f'.{fmt}', force=kwargs.get('overwrite', False),
                                       auto=kwargs.get('auto', False))
        if fmt == 'fig':
            store(fig, filename_ext)
        else:
            fig.savefig(filename_ext, format=fmt, dpi=600)
    MyLog.msg(0)


def equalbins(x, nbins):
    """ Computes the binning vector of a given data vector `x` categorized in equal proportion in `nbins`.

    :param np.ndarray x: vector of data to be binned.
    :param int nbins: number of bins.
    :return: a vector with the bins corresponding to each data point.
    :rtype: np.ndarray
    """
    quantiles = np.quantile(x, np.array(range(1, nbins)) / nbins)
    edges = np.concatenate(([-np.inf], quantiles, [np.inf]))
    return np.digitize(x, edges)


# noinspection PyTypeChecker
def get_all_files(data_dir: Union[str, pathlib.Path] = './results'):
    # Load data
    data_dir = pathlib.Path(data_dir)
    logging.info(f"Getting the file list.")
    file_list = data_dir.glob('simu*')
    dfs = []
    prop_dict = dict(bump=[], biased=[], sigma=[], i0=[], frames=[], t=[])
    filename = []
    for fp in file_list:
        logging.debug(f"Loading '{fp}'...")
        fn, ext = fp.stem, fp.suffix
        props = fn.split('_')
        logging.debug(f"... with properties '{props}'.")
        # Identify type, sigma, i0, frames and tmax
        if ext == '.csv':
            for key in prop_dict:
                if key == 'bump':
                    prop_dict[key].append(True) if 'bump' in props else prop_dict[key].append(False)
                    continue
                if key == 'biased':
                    prop_dict[key].append(True) if 'biased' in props else prop_dict[key].append(False)
                    continue
                for prop in props:
                    if prop.startswith(key):
                        prop_dict[key].append(float(prop.replace(key + '-', '')))

            filename.append(fp)
            dfs.append(pd.read_csv(fp))
            dfs[-1].attrs = dict(datafile=(fp.with_suffix('')))

    prop_dict['filename'] = filename
    attdf = pd.DataFrame(prop_dict)
    attdf.loc[attdf.t == 3.0, 't'] = 2.0
    attdf['duration'] = attdf['t'] / attdf['frames']
    attdf['frames'] = attdf['frames'].astype(int)

    return attdf


"""
Analytical methods
------------------                    
"""


def log_reg(dataframe, xlabels, ylabel='binchoice', compute_pm=True):
    """ Perform a logistic regression over the binary decision making task.

    :param pd.DataFrame dataframe: data containing the results of the simulation.
    :param list of str xlabels: labels from which extract the design matrix.
    :param str ylabel: label that points to the dependent variable.
    :param bool compute_pm: compute the Psychometric values for each trial based on the logistic model.
    :return: the logistic model, the weights and their confidence intervals, the 'modified' dataframe.
    :rtype: (:py:class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`, pd.DataFrame, pd.DataFrame)
    """
    x = np.array(dataframe[xlabels].astype(float))  # Design matrix [k trials x n samples]
    y = dataframe[ylabel].astype(np.int16).to_numpy()
    # Check that `y` is a vector of zeros and ones.
    code = np.sort(np.unique(y))
    if np.any(code != (0, 1)):
        w = code[1] - code[0]
        y = np.array((y - code[0]) / w, dtype=np.int16)
    m = sm.GLM(y, x, family=sm.families.Binomial())
    try:
        model = m.fit()
    except (sm_exceptions.PerfectSeparationError, RuntimeError):
        logging.error("There are not enough trials. Perfect separation detected.")
        return -1

    mframe = pd.DataFrame({'beta': model.params, 'errors': model.bse})

    # We can compute the 'continuous' probability of having a right choice for each trial using the above 'model'
    if compute_pm:
        dataframe['pright'] = 100.0 * logistic.cdf(np.dot(x, mframe.beta))  # (N, 8) x (8, 1)

    return model, mframe, dataframe


def log_reg2d(dataframe, bin_width=10, design_matrix_identifier='x', outcome_vector_identifier='binchoice'):
    """ 2-dimensional logistic regression over the binary decision making task. The design matrix
    takes into account not only the frame-number (regression along time) but also the spatial
    dimension (orientation of each stimulus frame). The spatial information is binned according
    to parameter ``bin_width``.

    :param pd.DataFrame dataframe: data containing the simulation results. It must contain columns with labels: *xk*
                                   where k is the frame number of the stimulus k. (Optionally: you can specify the
                                   name of those columns through ``design_matrix_identifier``.
    :param int bin_width: width of each spatial bin.
    :param str design_matrix_identifier: a string (name) that identifies the columns of the design matrix in dataframe.
    :param str outcome_vector_identifier: a string (name) that identifies the column of the outcome vector (y).
    :return: the logistic model, and the regressors with their corresponding confidence interval.
    :rtype: (:py:class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`, (np.ndarray, np.ndarray) )
    """
    # Detect number of frames
    labels = re.findall(f"({design_matrix_identifier}[0-9][0-9]?) ", ' '.join(dataframe.columns))
    frames = len(labels)

    # Build the desing matrix by binning the spatial variable
    bins = np.arange(-90 - bin_width // 2 - 1, 90 + bin_width // 2 + 1, bin_width)
    zero_bin = int(np.digitize(0, bins))  # Transform each spatial value into a integer-based mapping
    dm_raw = dataframe[labels].apply(lambda x: np.digitize(x, bins)).astype(np.int16)  # Take the relevant data
    dm_dummy = pd.get_dummies(data=dm_raw, columns=labels).astype(np.int16)  # Create the dummy matrix (0s and 1s)
    zero_labels = ['x%d_%d' % (i, zero_bin) for i in range(1, frames + 1)]  # Identify the zero (center) values
    dm_dummy.loc[:, zero_labels] = 0  # Set the dummy variable

    # Outcome values
    y = (dataframe[outcome_vector_identifier].astype(np.int8).to_numpy() + 1) // 2

    # Load and fit the model
    m = sm.GLM(y, dm_dummy.to_numpy(), family=sm.families.Binomial())
    model = m.fit()

    # Reshape regressors and errors
    betas = model.params.reshape((frames, len(np.unique(dm_raw))))
    errors = model.bse.reshape((frames, len(np.unique(dm_raw))))

    return model, (betas, errors)


def noise_ppk(dataframe, sp, **kwargs):
    """ Computes the psychophysical kernel given the estimation and the phase of the sensory stimulus. This approach
    is not correct because it does not take into account the modulus of the sensory stimulus (which may even be
    negative).

    :param pd.DataFrame dataframe: data-frame containing the outcome (estimation) of each trial.
    :param np.ndarray sp: Evolution of the phase of the noisy stimulus for each trial.
    :param kwargs: Additional keyword arguments.
    :return: An array containing the (smoothed) impact of the noisy stimulus at each time bin.
    :rtype: np.ndarray
    """
    cue = kwargs.get('cue', 0.250)
    nframes = kwargs.get('nframes', 8)
    dt = kwargs.get('dt', 2E-4)
    tmax = cue * nframes
    tpoints = np.arange(0, tmax, dt)

    sp = sp[0:len(tpoints)]
    # Define a smoothing filter
    temp_smoothing = np.ones(int(kwargs.get('resample', 100E-3) / dt))
    temp_smoothing = temp_smoothing / len(temp_smoothing)

    # Obtain a boolean array that categorizes the trials between clockwise and counter-clockwise outcomes.
    choice = kwargs.get('decision', 'estim')

    dec = dataframe[choice] > 0
    pk = np.convolve(circ_dist(circmean(sp[:, dec], axis=1), circmean(sp[:, ~dec], axis=1)),
                     temp_smoothing, 'same')
    return pk


def compute_estimation_old(dataframe, bin_width=1.0, **kwargs):
    """ Computes the estimation curve given an angular bin of width `bin_width`.

    :param pd.DataFrame dataframe: data-frame containing the circular mean of the stimulus orientation along with
                                   its estimation by the model.
    :param float bin_width: width of the binning in degrees to perform the average across trials.
    :param kwargs: additional keyword arguments.
    :return: Returns a tuple (binning, average estimation, standard error).
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """
    lim = kwargs.pop('lim', [-90, 90])
    avg_estim = []
    std_estim = []
    binning = np.arange(lim[0], lim[1], bin_width)
    for point in binning:  # Compute the mean and std of the binned data.
        binned_estim = dataframe.estim.loc[
            (dataframe.average_circ >= point) & (dataframe.average_circ < (point + bin_width))]
        avg_estim.append(binned_estim.mean())
        # TODO: compute the error bars as the std of the residuals.
        std_estim.append(binned_estim.std())
    return binning + bin_width / 2, np.array(avg_estim), np.array(std_estim)


def compute_estimation(dataframe, nbins=41, ylabel='estim', **kwargs):
    """ Computes the estimation curve given an angular binning with `nbins` bins.

    :param pd.DataFrame dataframe: data-frame containing the circular mean of the stimulus orientation along with
                                   its estimation by the model.
    :param int nbins: number of bins to divide the angular range and perform the average across trials.
    :param str ylabel: label of the outcome column in the data-frame.
    :param kwargs: additional keyword arguments.
    :return: Returns a tuple (binning: average stimulus orientation, average estimation, standard error).
    :rtype: (pd.Series, pd.Series, pd.Series)
    """
    lim = kwargs.pop('lim', 90)
    # We define the categories for the data points and the model points/line
    dataframe['bin'] = equalbins(dataframe.average_circ, nbins)

    # Group the data by the binning and compute the circular mean of the stimulus orientation averages and the circular
    # mean of the estimated average orientations
    gr1 = dataframe.groupby('bin')
    x_avg = gr1.average_circ.apply(circmean, low=-180, high=180)
    y_avg = gr1[ylabel].apply(circmean, low=-180, high=180)
    y_std = gr1[ylabel].apply(circstd, low=-180, high=180)

    # Select the data inside the desired range
    s_x = x_avg.loc[np.abs(x_avg) <= lim]
    s_y = y_avg.loc[np.abs(x_avg) <= lim]
    s_e = y_std.loc[np.abs(x_avg) <= lim]

    return s_x, s_y, s_e


def compute_performance(dataframe, perfect=False):
    """ Computes the performance using the psychophysical kernel. This approach is *incorrect* because the
    logistic regression is not a valid model for these data.

    :param pd.DataFrame dataframe: data-frame with at least the design matrix (xi, i = 1, ..., 8) and the outcome.
    :param bool perfect: whether to compute the performance of the perfect integrator.
    :return: area under the psychophysical kernel.
    :rtype: np.ndarray of float
    """
    area = []
    # df_norm, lbls = normalize_dataframe(dataframe)
    df_norm = dataframe.copy()
    for k in range(1, 9):
        xlabels = [f"x{j}" for j in range(1, k + 1)]
        if perfect:
            xlabels = [f"x{j}" for j in range(1, k + 1)]
            df_norm['binchoice_old'] = (circmean(df_norm[xlabels], high=180, low=-180, axis=1) >= 0) * 2 - 1
        else:
            df_norm['binchoice_old'] = (df_norm[f"ph{k}"] >= 0) * 2 - 1
        # lbl_fmt = f"x[0-{k}][0-{k}]?_norm"
        # labels = re.findall(f"({lbl_fmt})", ' '.join(df_norm.columns))
        _, mf_r, _ = log_reg(df_norm, xlabels, ylabel='binchoice_old')
        area.append(np.sum(mf_r.beta))
    return area


def compute_lin_reg(dataframe, xlabels, ylabel):
    """ Computes the linear regression $y = \\sum_i\\beta_i x_i + \\xi$ using OLS. This approach is *incorrect* to
    compute the psychophysical kernel of the estimation because the variables are circular.

    :param pd.DataFrame dataframe: data-frame with the design matrix (xlabels) and the outcome (ylabel).
    :param list of str xlabels: labels used to select the columns of the dataframe that compose the design matrix.
    :param str ylabel: label used to select the column indicating the outcome.
    :return: Returns fitting parameters and their errors along with the residuals.
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """
    x = np.array(dataframe[xlabels].astype(float))
    y = np.array(dataframe[ylabel].astype(float))
    m = sm.OLS(y, x)
    model = m.fit()
    return model.params, model.bse, model.resid


def compute_circ_reg_by(dataframe, xlabels, ylabel):
    """ Compute the circular regression of a circular dataset using Bayesian sampling over a Projected Normal circular
    regression model  [Cremers2020]_. This function calls R which must be installed in the system and reachable for
    the python interpreter.
     The success of the fitting algorithm is not guaranteed and its computational cost depends on the parameters passed
    to the :func:`bpnr` function.

    .. [Cremers2020] Jolien Cremers (2020). bpnreg: Bayesian Projected Normal Regression Models for Circular Data.
                     R package version 1.0.3. https://CRAN.R-project.org/package=bpnreg

    :param pd.DataFrame dataframe: dataframe from where the design matrix and the outcome is extracted.
    :param list of str xlabels: list of labels pointing to the columns of the design matrix.
    :param str ylabel: label that indicates the outcome column.
    :return: weights, their error and other information.
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """

    logging.info("Computing circular regression using Bayesian sampling, this could take a while ...")
    x = np.deg2rad(np.array(dataframe[xlabels].astype(float)))
    y = np.deg2rad(np.array(dataframe[ylabel].astype(float)))

    # Create auxiliary dataframe (design matrix and outcome)
    df = pd.DataFrame(np.column_stack((x, y)), columns=['x%d' % k for k in range(1, 9)] + ['estim'])

    # Convert data to R and fit the data using R library ("bpnreg") (R version: >= 4.0.1)
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.r.assign('df', df)
        ro.r('''
                library("bpnreg")
                fit.estim <- bpnr(pred.I = estim ~ 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8, 
                data = df, its = 2000, burn = 100, n.lag = 3, seed = 10101)
            ''')

        fit = ro.r('fit.estim$lin.coef.II')
        betas = fit[1:, 0]
        betas = betas / sum(betas)
        betas_err = fit[1:, 2]
        betas_err = betas_err / sum(betas)

    return betas, betas_err, fit


def circ_reg_model(p, x, y):
    """ Model of the estimation and its -loglikelihood. To be used along :func:`compute_circ_reg_ml` to maximize
    log-likelihood.

    :param list p: list of parameters to be fitted (betas, kappa and slope parameter k)
    :param np.ndarray x: design matrix.
    :param np.ndarray y: outcome array.
    :return: Minus loglikelihood.
    :rtype: float
    """
    n = len(x)
    p = np.array(p)
    beta, kappa, k = p[0:-2], p[-2], p[-1]
    # Compute the "estimated" estimation
    theta_hat = k * np.angle(np.dot(beta, np.exp(1j * x).T))
    # - log-likelihood
    # noinspection PyCallingNonCallable
    return -(np.sum(kappa * np.cos(y - theta_hat)) - n * np.log(2 * np.pi * besseli(kappa)))


def circ_probit_model(p, x, y):
    """ Model of the categorization and its -loglikelihood. To be used along :func:`probit_circular` to maximize
    log-likelihood.

    :param list p: list of parameters to be fitted (betas, kappa and slope parameter k)
    :param np.ndarray x: design matrix.
    :param np.ndarray y: outcome array (zeros and ones).
    :return: Minus loglikelihood.
    :rtype: float
    """

    p = np.array(p)
    beta, kappa, k = p[0:-2], p[-2], p[-1]
    # Compute the "estimated" estimation
    theta_hat = k * np.angle(np.dot(beta, np.exp(1j * x).T))
    # Compute the cdf of the von-misses distribution centered at `theta_hat` with parameter `kappa`
    cdf = vonmises.cdf(theta_hat, kappa)
    # minus log-likelihood
    return -np.sum(y * np.log(cdf) + (1 - y) * np.log(1 - cdf))


def compute_circ_reg_ml(dataframe, xlabels, ylabel, normed=1):
    """ Circular regression using maximum likelihood. The estimation is assumed to be a circular mean of the
    stimulus orientations with fluctuations that converge towards a Von Misses distribution. The model corresponds
    to :func:`circ_reg_model`. The obtained weights are normalized such that `sum(betas) = 1`. This is a constraint
    passed to the minimization algorithm which is necessary due to the ambiguity of the model (the modulus of the
    summed vector can be anything; we only fit the angle).

    :param pd.DataFrame dataframe: data-frame with the design matrix `x` and the outcome `y`.
    :param list of str xlabels: column labels of the design matrix.
    :param str ylabel: label of the outcome column in the data-frame.
    :return: Normalized weights, their errors and the minimization object that contains all fitting results.
    :rtype: (np.ndarray, np.ndarray, Any)
    """
    mylog = MyLog(10)
    logging.info("Computing circular regression using Max. likelihood, this could take a while ...")
    x = np.deg2rad(np.array(dataframe[xlabels].astype(float)))
    y = np.deg2rad(np.array(dataframe[ylabel].astype(float)))

    # Minimize log-likelihood
    min_options = dict(disp=False, maxiter=10000, eps=1E-10)
    nframes = len(xlabels)

    # Boundaries
    lbound = np.zeros(nframes + 2).tolist()
    ubound = np.concatenate((np.ones(nframes) * 10, [10000, 10])).tolist()
    min_bounds = list(np.array([lbound, ubound]).transpose())

    # Constraint
    def constraint(b):
        return normed - np.sum(b[0:nframes])

    # Starting point
    success = False
    while not success:
        starting_parameters = np.abs(np.random.rand(nframes + 2))
        fit = minimize(circ_reg_model, starting_parameters, args=(x, y), bounds=min_bounds, options=min_options,
                       constraints={"fun": constraint, "type": "eq"}, tol=1E-8)
        success = fit.success
        mylog(0) if success else mylog(-1)
        if not success:
            logging.info("Minimization failed, retrying ...")

    # noinspection PyUnboundLocalVariable
    betas = fit.x[0:nframes]
    betas_err = np.zeros_like(betas)

    return betas, betas_err, fit


def probit_circular(dataframe, xlabels, ylabel='binchoice', plabel='pright', compute_pm=True, **kwargs):
    """ Computes the probit regression for circular variables by means of max. likelihood. The design matrix is
    indicated by the columns with labels `xlabels` and the outcome with the label `ylabel`.

    :param pd.DataFrame dataframe: data-frame containing the simulation data.
    :param list of str xlabels: column labels corresponding to the design matrix.
    :param str ylabel: column label corresponding to the binary outcome.
    :param str plabel: column label corresponding to the categorical probability.
    :param bool compute_pm: compute the Psychometric values for each trial based on the circular probit model.
    :return: the fitting model, the weights and the modified data-frame.
    :rtype: (Any, pd.DataFrame, pd.DataFrame)
    """
    mylog = MyLog(10)
    logging.info("Computing binary circular regression using Max. likelihood, this could take a while ...")
    x = np.deg2rad(np.array(dataframe[xlabels].astype(float)))
    y = dataframe[ylabel].astype(np.int16).to_numpy()
    nframes = len(xlabels)
    # Check that `y` is a vector of zeros and ones.
    code = np.sort(np.unique(y))
    if np.any(code != (0, 1)):
        w = code[1] - code[0]
        y = np.array((y - code[0]) / w, dtype=np.int16)

    # Minimize log-likelihood
    # #### #### #### #### ###
    min_options = dict(disp=False, maxiter=10000, eps=1E-10)  # Minimization options
    min_options.update(**kwargs)
    tol = 1E-8
    # Boundaries
    lbound = np.zeros(nframes + 2).tolist()
    ubound = np.concatenate((np.ones(nframes) * 10, [10000, 10])).tolist()
    min_bounds = list(np.array([lbound, ubound]).transpose())

    # Constraint
    def constraint(b):
        return 1.0 - np.sum(b[0:nframes])

    # Starting point
    success = False
    max_iter = 1000
    iter = 0
    while not success and iter < max_iter:
        starting_parameters = np.abs(np.random.rand(nframes + 2))

        fit = minimize(circ_probit_model, starting_parameters, args=(x, y), bounds=min_bounds, options=min_options,
                       constraints={"fun": constraint, "type": "eq"}, tol=tol)
        success = fit.success
        mylog(0) if success else mylog(-1)
        if not success:  # Usually fails for the sigma = 0
            logging.info(f"Last fit: {fit.x}")
            logging.info("Minimization failed, retrying ...")
            min_options.update(disp=True)  # Minimization options
            tol = tol * 10

    # Prepare output
    # noinspection PyUnboundLocalVariable
    betas = fit.x[0:nframes]
    kappa = fit.x[nframes]
    k = fit.x[nframes + 1]
    # TODO: compute the error bars with bootstrapping
    betas_err = np.zeros_like(betas)
    mframe = pd.DataFrame({'beta': betas, 'errors': betas_err})

    # We can compute the 'continuous' probability of having a right choice for each trial using the above 'model'
    if compute_pm:
        theta_hat = k * np.angle(np.dot(betas, np.exp(1j * x).T))
        dataframe[plabel] = vonmises.cdf(theta_hat, kappa)

    return fit, mframe, dataframe


def compute_pm_reg(dataframe, bins=41, lim=45, compute=False, save=True, plabel='pright', ylabel='binchoice_phase',
                   fitlabel='bfit_pm', phase_label='estim', nframes=8):
    """ Computes the psychometric points by means of the binary circular regression.

    .. todo::

       Compute the errors (25% and 75% quantiles).

    :param pd.DataFrame dataframe: data-frame with the continuous estimation and the design matrix.
    :param int bins: number of bins (points) that are taken from the model to be plotted.
    :param float lim: angular limit that fixes the range of the psychometric plot.
    :param bool compute: Force the fitting computation.
    :param bool save: whether the computed fit and the modified data-frame are saved.
    :param str plabel: column label corresponding to the categorical probability.
    :param str ylabel: column label corresponding to the binary outcome.
    :param str fitlabel: keyword in which the fit is stored.
    :param str phase_label: column label corresponding to the continuous outcome.
    :param int nframes: number of stimulus frames.
    :return: Bins and Probability of clockwise choices for each stimulus orientation bin.
    :rtype: (pd.Series, pd.Series)
    """

    # Compute the binary regression if there are no previous fits of this data
    if plabel not in dataframe.columns or compute:
        dataframe[ylabel] = (dataframe[phase_label] >= 0)
        xlabels = [f"x{k}" for k in range(1, nframes + 1)]
        fit, _, dataframe = probit_circular(dataframe, xlabels, ylabel=ylabel, plabel=plabel)
        np.save(str(dataframe.attrs['datafile']) + '_pm_reg', dict(fit), allow_pickle=True)
        # Save the data for the future
        if save:
            logging.debug("Saving the modified dataframe and fit ...")
            dataframe.to_csv(dataframe.attrs['datafile'] + '.csv')
            data = np.load(dataframe.attrs['datafile'] + '.npy', allow_pickle=True)
            try:
                data[fitlabel] = dict(fit)
            except IndexError:
                data = dict(data.item())
                data[fitlabel] = dict(fit)
            np.save(dataframe.attrs['datafile'], data, allow_pickle=True)

    # We define the categories for the data points and the model points/line
    dataframe['bin'] = equalbins(dataframe.average_circ, bins)

    # Group the data by the binning and compute the circular mean of the stimulus orientation averages and the median
    # of the fitted probability for each point in the bin
    gr1 = dataframe.groupby('bin')
    x_avg = gr1.average_circ.apply(circmean, low=-180, high=180)
    y_reg = gr1[plabel].mean()

    # Select the data inside the desired range
    s_x = x_avg.loc[np.abs(x_avg) <= lim]
    s_y = y_reg.loc[np.abs(x_avg) <= lim]

    return s_x, s_y


def compute_pm_data(dataframe, bins=21, lim=45.0, sample_size=0.1, compute_errorbars=False, ylabel='estim'):
    """ Computes the psychometric data points (x, y) where x corresponds to the circular mean of the
    stimulus orientations averages (which are circular means of their corresponding stimulus frames), and
    y corresponds to the arithmetic mean of the binary choice for which the sign of the phase is taken as
    the readout.

    :param pd.DataFrame dataframe: data-frame with the continuous estimation and the design matrix.
    :param int bins: number of bins in which divided the data points.
    :param float lim: angular limit that fixes the range of the psychometric plot.
    :param float sample_size: size of the sample (proportion of the entire dataset) for bootstrapping.
    :param bool compute_errorbars: whether the errorbars are computed or not (*Incorrect*).
    :param str ylabel: column label corresponding to the binary outcome.
    :return: Binning and Probability of clockwise choices for each stimulus orientation bin.
    :rtype: (pd.Series, pd.Series, np.ndarray, np.ndarray) or (pd.Series, pd.Series)
    """

    dataframe['binchoice_phase'] = (dataframe[ylabel] >= 0).astype(float)
    # We define the categories for the data points and the model points/line
    dataframe['bin'] = equalbins(dataframe.average_circ, bins)

    # Group the data by the binning and compute the circular mean of the stimulus orientation averages and the average
    # of the binary outcomes
    gr1 = dataframe.groupby('bin')
    x_avg = gr1.average_circ.apply(circmean, low=-180, high=180)
    y_avg = gr1.binchoice_phase.mean()

    # Select the data inside the desired range
    s_x = x_avg.loc[np.abs(x_avg) <= lim]
    s_y = y_avg.loc[np.abs(x_avg) <= lim]

    if compute_errorbars:
        import bootstrapped.bootstrap as bs
        import bootstrapped.stats_functions as bs_stats
        # Compute the error bars of the mean by bootstrapping
        logging.info('Estimating confidence intervals by bootstrapping. This can take a while ...')
        lb, ub = [], []
        size = int(gr1.size()[1] * sample_size)
        for k in range(100):
            lb.append([])
            ub.append([])
            for b in gr1:
                res = bs.bootstrap(b[1].binchoice_phase.sample(size).astype(int).to_numpy(), stat_func=bs_stats.mean,
                                   is_pivotal=True)
                lb[-1].append(res.value - res.lower_bound)
                ub[-1].append(res.upper_bound - res.value)
        mlb = np.array(lb).mean(axis=0)
        mub = np.array(ub).mean(axis=0)
        s_l = mlb[np.abs(x_avg) <= lim]
        s_u = mub[np.abs(x_avg) <= lim]
        return s_x, s_y, s_l, s_u

    return s_x, s_y


def get_amp_eq_stats(i0s=(0.02, 0.05, 0.08), **kwargs):
    """ Obtain the estimates, the psychophysical kernels and the psychometric data for the amplitude
    equation simulations.

    :param tuple or list or np.ndarray i0s: values of the general excitability parameter.
    :param kwargs: key-word arguments that control if the data must be simulated, loaded and/or saved. Additional
                   options passed to the function :fun:`lib_ring.amp_eq_simulation`.
    :return: weights of the psychophysical kernels for the ring-data and the amplitude equation data.
    :rtype: ([list], [list])
    """
    logging.info(f"Computing the psychophysical kernels of the amplitude equation data ...")
    simu_opts = dict(translate=True, save_evo=False, init_time=0.0, rinit=0.2)
    simulate = kwargs.pop('simulate_amp_eq', False)
    save = kwargs.pop('save_amp_eq', False)
    bump = 'bump' if kwargs.get('bump', False) else 'nobump'
    simu_opts.update(kwargs)

    est1, est2 = ([], [])
    wrs, was = ([], [])
    pm1, pm2 = ([], [])
    pmd1, pmd2 = ([], [])
    for i0 in i0s:
        # load datafiles to extract the design matrices
        dfile = f"results/variable_frames/modified/simu_10000_{bump}_sigma-0.15_i0-{i0:0.2f}_frames-8_t-2.0_mod"
        logging.info(f"Loading data from '{dfile}.*' ...")
        df = pd.read_csv(dfile + '.csv')
        df.attrs['datafile'] = dfile
        try:
            data = np.load(dfile + '.npy', allow_pickle=True, encoding='latin1')
            # data = load_obj(dfile, extension='.npy')
            conf = data['conf']
        except IndexError:
            # noinspection PyTypeChecker
            data = dict(np.load(dfile + '.npy', allow_pickle=True).item())
            conf = data['conf']

        xlabels = ['x%d' % (k + 1) for k in range(0, conf['nframes'])]
        if simulate or ('amp_estim' not in df.columns):
            simu_opts.update(conf)
            orients = df[xlabels].to_numpy().T
            simu_opts.update(tmax=(simu_opts['nframes'] * simu_opts['cue_duration'] + simu_opts['dt']))
            logging.info(f"Simulating amplitude equation evolution ...")
            _, _, estim, _, _, _, _ = simulation(orientations=orients, **simu_opts)
            df['amp_estim'] = np.rad2deg(estim[0])
            df['amp_choice'] = (df.amp_estim >= 0)
            if save:
                logging.info(f"Saving the simulated data to '{dfile}.csv' ...")
                df.to_csv(dfile + '.csv')
        # Estimation curves
        est1.append(compute_estimation(df, ylabel='estim'))
        est2.append(compute_estimation(df, ylabel='amp_estim'))
        # PPKs
        # Try to load the weights if any or compute them if not
        if 'bfit_est' not in data.keys():
            wr, _, rfit = compute_circ_reg_ml(df, xlabels, ylabel='estim')
            if save:
                data['bfit_est'] = dict(rfit)
        else:
            wr = data['bfit_est']['x'][0:conf['nframes']]
        if simulate or ('bfit_est_amp' not in data.keys()):
            wa, _, afit = compute_circ_reg_ml(df, xlabels, ylabel='amp_estim')
            if save:
                data['bfit_est_amp'] = dict(afit)
        else:
            wa = data['bfit_est_amp']['x'][0:conf['nframes']]
        wrs.append(wr)
        was.append(wa)
        if save:
            np.save(dfile, data, allow_pickle=True)

        # Psychometric data
        # Try to load the psychometric data if any or compute it if not
        pm1.append(compute_pm_reg(df, lim=60, plabel='pright', ylabel='binchoice_phase', fitlabel='bfit_pm'))
        pm2.append(compute_pm_reg(df, compute=simulate, lim=60, plabel='amp_pright', ylabel='amp_choice',
                                  fitlabel='bfit_pm_amp', phase_label='amp_estim'))
        pmd1.append(compute_pm_data(df, bins=15, lim=60, ylabel='estim'))
        pmd2.append(compute_pm_data(df, bins=15, lim=60, ylabel='amp_estim'))

    return (est1, est2), (wrs, was), ((pm1, pmd1), (pm2, pmd2))


def measure_delta_ang(**kwargs):
    """ Measures the angular displacement in the amplitude equation due to a stimulus frame of a certain duration and
    strength. The measure is done using both the simulation of the amplitude equation (:func:`simulation`) and
    its analytical approximation: :func:`psi_evo_r_constant`.

    :param kwargs: key-word arguments that contain all the parameters of the model and the simulation.
    :return: a data-frame containing the values of some parameters, the displacement of the phase and the modulus of
             the amplitude equation. It also contains the orientation of the stimulus.
    """
    rbase = []
    rfinal = []
    psi_final = []
    psi_aprox_final = []
    i0s = kwargs.pop('i0s', np.arange(0.1, 1.1, 0.1))
    thetas = kwargs.get('thetas', [90])
    i0_dat = []
    theta_dat = []

    logging.info(f"Computing the impact of the stimulus for different I0 and theta_s:")
    for i0 in i0s:
        logging.debug(f"I0 = {i0:.2f}")
        for theta in thetas:
            kwargs.update(dict(orientations=np.array([theta]), sigmaOU=0.0,
                               correction=2 * 0.15 ** 2))
            data = simulation(mu=i0, theta_0=0, **kwargs)
            rbase.append(data[1][0])
            rfinal.append(data[1][-1])
            psi_final.append(data[2][-1])
            if psi_final[-1] == 0:
                psi_final[-1] = data[2][-2]
                rfinal[-1] = data[1][-2]

            data2 = psi_evo_r_constant((0, kwargs.get('tframe', 0.250), 100), r=rfinal[-1], i1=data[4][-2], theta=theta)
            psi_aprox_final.append(data2[1][-1])
            i0_dat.append(i0)
            theta_dat.append(theta)

    data = dict(i0=i0_dat, theta=theta_dat, r0=np.array(rbase), rf=np.array(rfinal), psi=np.rad2deg(psi_final),
                psi_aprox=psi_aprox_final)

    return pd.DataFrame(data=data)


def compute_ppk_slope(b):
    """ Method to compute the slope(s) of the weights of the given PPK(s).

    :param list b: weights corresponding to the PPK or multiple PPK.
    :return: slope(s)
    :rtype: list of float or float
    """
    # Detect whether we have a single list of weights or a list of lists.
    if isinstance(b, Iterable):
        if not isinstance(b[0], Iterable):
            b = [b]
    else:
        raise TypeError("Parameter 'b' should be an iterable ...")

    slopes = []
    for ws in b:
        frames = np.arange(1, len(ws) + 1)
        v = np.var(frames, ddof=1)
        # Fit line to weights
        frames = sm.add_constant(frames)
        m = sm.OLS(ws, frames)
        model = m.fit()
        slopes.append(2 * v * model.params[1])
    return slopes


def slopes_vs_time(i0s=(0.02, 0.05, 0.08, 0.1, 0.2), **kwargs):
    dfs = []
    for i0 in i0s:
        compute = kwargs.get('compute_weights', False)
        weights = []
        # Check if previous results exist
        logging.info(f"Loading previously computed weigths for I0 = {i0:.2f}, if there are any ...")
        try:
            weights = np.load(f"./results/variable_frames/ppk_weights_i0-{i0:.2f}_frames-16_t-4.0.npy",
                              allow_pickle=True)
        except IOError:
            logging.info(f"There are no pre-computed weights.")
            compute = True
        if compute:
            logging.info(f"Computing new weights ...")
            df = pd.read_csv(f"./results/variable_frames/simu_10000_nobump_sigma-0.15_i0-{i0:.2f}_frames-16_t-4.0.csv")
            for k in range(2, 17):
                logging.debug(f"... for {(k + 1)} frames ...")
                xlabels = ['x%d' % j for j in range(1, k + 1)]
                ylabel = 'ph%d' % k
                w, _, _ = compute_circ_reg_ml(df, xlabels, ylabel=ylabel)
                weights.append(w)
            logging.info("Saving weigths.")
            np.save(f"./results/variable_frames/ppk_weights_i0-{i0:.2f}_frames-16_t-4.0", weights, allow_pickle=True)
        logging.info("Computing slopes for this weights.")
        sl = compute_ppk_slope(weights)
        d = pd.DataFrame(sl, columns=['slope'])
        d['frames'] = np.arange(2, 17)
        d['t'] = np.arange(2, 17) * 0.25
        d['i0'] = i0
        dfs.append(d)

    logging.info("Merging datasets.")
    d_t = pd.concat(dfs)
    d_t.to_csv('./results/ppk_slopes_duration-0.25_frames-2to16.csv')
    return d_t


def compute_sqr_error(filename, betas, k=1):
    """ Compute the estimation squared error of a dataset and the errors of its corresponding weighted circular
    average (similar to a weighted perfect integrator, in which fluctuations are not taken into account).

    :param str filename: name of the file that contains the trial data.
    :param pd.DataFrame betas: a :obj:`pandas.DataFrame` with the weights of the circular regression (obtained from
                               :func:`compute_circ_reg_ml`) and column names that correspond to the column names of
                               the design matrix in `filename`.
    :param float k: the estimation bias parameter (not used). We fix this to 1 to compute the "perfect integration"
                    squared error.
    :return: the squared estimation error and the squared estimation error of the perfect integration.
    :rtype: (float, float, flaat, float)
    """
    # Load the data-frame from `filename`
    df = pd.read_csv(filename)
    try:
        xlabels = betas.columns
    except AttributeError:
        xlabels = betas.keys().to_list()

    betas = betas.astype(np.float64)

    x = np.deg2rad(df[xlabels].to_numpy())
    true = np.deg2rad(df.average_circ)
    y = np.deg2rad(df.estim)
    # We use k = 1 to obtain the estimation of the Perfect Integrator
    y_hat = np.angle(np.dot(betas, np.exp(1j * x).T))

    es_err = np.mean(np.rad2deg(circ_dist(y, true)) ** 2)
    pi_err = np.mean(np.rad2deg(circ_dist(y_hat, true)) ** 2)
    std_err = np.mean(np.rad2deg(circ_dist(y, true * k)) ** 2)

    # Compute the standard deviation
    sx, sy, se = compute_estimation(df)
    std_dev = np.mean(se)

    return es_err, pi_err, std_err, std_dev


def get_sqr_errors(**kwargs):
    """ Get the squared errors from the statistics :obj:`pandas.DataFrame` saved in disk. If the statistics for the
    chosen data do not exist, compute them and add them to the file for future uses.

    :param kwargs: key-word arguments to select the target data.
    :return: selected statistics :obj:`pandas.DataFrame`
    :rtype: pd.DataFrame
    """

    nframes = kwargs.get('nframes', 8)
    tmax = kwargs.get('tmax', 2.0)
    bump = kwargs.get('bump', False)
    biased = kwargs.get('biased', False)
    directory = pathlib.Path(kwargs.get('data_dir', './results/variable_frames/'))
    label = kwargs.get('label', 'sqr_error')
    pi_label = kwargs.get('pi_label', 'pi_error')
    xlabels = ['x%d' % i for i in range(1, nframes + 1)]
    logging.info("Loading and selecting the data that meets the criteria ...")
    # Load data-frame containing statistics
    stats_file = kwargs.get('stats_file', f"./results/integration_stats_frames-{nframes}_t-{tmax:.1f}.csv")
    stats = pd.read_csv(stats_file)
    # Load all the data from `directory`
    fdf = get_all_files(directory)

    # Select the files that meet the criteria
    if kwargs.get('i0s', None) is not None:
        sel = fdf.loc[(fdf.frames == nframes) & (fdf.t == tmax) & (fdf.bump == bump) & (fdf.biased == biased)].copy()
    else:
        sel = fdf.loc[(fdf.frames == nframes) & (fdf.t == tmax) & (fdf.bump == bump) & (fdf.biased == biased)].copy()
    # Check if the selected files are already on the `stats` data-frame
    sel_set = set(sel.filename.to_list())
    stats_set = set(stats.filename.to_list())
    not_included = sel_set.difference(stats_set)
    # Select only the files that are not in the `stats` dataframe
    new_sel = sel.loc[sel.filename.isin(not_included)].copy().reset_index(drop=True)
    logging.info('... done.')
    # Compute the PK and other things before we continue with the errors (if we have to)
    if len(new_sel) != 0:
        logging.info(f"Computing PK for {len(new_sel)} data-frames ...")
        weights = []
        pm_dat = []
        pm_reg = []
        kappas = []
        reg_slopes = []
        for n, (k, row) in enumerate(new_sel.iterrows()):
            logging.info(f"Computing PK ({n + 1}/{len(new_sel)}) for '{row.filename}' data-frame ...")
            df = pd.read_csv(row.filename)
            df.attrs['datafile'] = pathlib.Path(row.filename).with_suffix('')
            w, _, fit = compute_circ_reg_ml(df, xlabels, ylabel='estim')
            sx, sy = compute_pm_data(df, lim=90, bins=18)
            pm_dat.append([sx, sy])
            sxp, syp = compute_pm_reg(df, lim=90, bins=41, save=False, nframes=nframes)
            pm_reg.append([sxp, syp])
            weights.append(w)
            kappas.append(fit.x[-2])
            reg_slopes.append(fit.x[-1])

        logging.info("Modifying `stats` data-frame and saving ...")
        x = pd.DataFrame(weights, columns=xlabels)
        pm_dat = np.array(pm_dat)
        pm_reg = np.array(pm_reg)
        new_sel = pd.concat([new_sel, x], axis=1)  # We add the columns with the PK weights
        new_sel['ppk_slope'] = compute_ppk_slope(weights)  # We compute the PK slope
        new_sel['kappa'] = np.array(kappas)
        new_sel['k_slope'] = np.array(reg_slopes)
        for k in range(len(pm_dat[0, 0, :])):
            new_sel['pmx%d' % k] = pm_dat[:, 0, k]
            new_sel['pmy%d' % k] = pm_dat[:, 1, k]
        for k in range(len(pm_reg[0, 0, :])):
            new_sel['pmxr%d' % k] = pm_reg[:, 0, k]
            new_sel['pmyr%d' % k] = pm_reg[:, 1, k]

        # Finally we add the data to the `stats` data-frame, and take the opportunity to clean the data-frame
        for col in stats.columns:
            if col.startswith('Unnamed') or col.startswith('index'):
                stats.drop(columns=col, inplace=True)
        stats = stats.append(new_sel).reset_index(drop=True)
        # Save
        stats.to_csv(stats_file)

    # Select the data again from the stats
    logging.info("Computing missing errors...")
    sel = stats.loc[(stats.frames == nframes) & (stats.t == tmax) & (stats.bump == bump)
                    & (stats.biased == biased)].copy()
    # Compute the squared error and update the `stats` data-frame
    compute = kwargs.get('compute_errors', False)
    update = False
    for k, row in sel.iterrows():
        if label in row.keys():
            if not np.isnan(row[label]):
                if not compute:
                    continue
        update = True
        es_err, pi_err, std_err, std_dev = compute_sqr_error(row.filename, row[xlabels], row.k_slope)
        stats.loc[k, label] = es_err
        stats.loc[k, pi_label] = pi_err
        stats.loc[k, 'std_err'] = std_err
        stats.loc[k, 'std_dev'] = std_dev
        sel.loc[k, label] = es_err
        sel.loc[k, pi_label] = pi_err
        sel.loc[k, 'std_dev'] = std_dev

    # Save changes
    if update:
        logging.info("Saving changes...")
        stats.to_csv(stats_file)

    return sel



"""
Running average methods
-----------------------

Methods to analyze the dynamics of the running average and the leaky integrator.
"""


def averager_integrator(kappas=(0.1, 1, 10, 100), orientations=(0, ), size=(50, 100), **kwargs):

    logging.info("Computing data for different widths of the distribution of stimuli.")
    biased = kwargs.pop('biased', False)
    theta_0 = 0 if biased else -1
    simu_opts = dict(translate=True, save_evo=False, init_time=0.0, rinit=0.01, i1=0.005, nframes=size[0],
                     ntrials=size[1], cue_duration=0.250, tmax=(size[0] * 0.250 + 2E-4), sigmaOU=0.15, theta_0=theta_0)
    bump = 'bump' if kwargs.get('bump', False) else 'nobump'
    i0s = kwargs.pop('i0s', np.arange(0.01, 0.21, 0.01))
    simu_opts.update(kwargs)

    # Generate the stimuli
    orientations = np.deg2rad(orientations)
    thetas, rt, angt, ra, anga = [], [], [], [], []
    for x in orientations:
        logging.info(f"Computing for an average orientation: ({int(np.rad2deg(x)):d})")
        rt.append([])
        angt.append([])
        thetas.append([])
        ra.append([])
        anga.append([])
        for kappa in kappas:
            logging.info(f"Computing for width (kappa): ({kappa})")
            th = np.random.vonmises(x, kappa, size=size)
            # Running average
            t, r, a, _ = running_avg(np.exp(1j*th))
            thetas[-1].append(th)
            rt[-1].append(r)
            angt[-1].append(a)

            # Amplitude equation simulation
            orients = np.rad2deg(th)
            ra[-1].append([])
            anga[-1].append([])
            for i0 in i0s:
                logging.info(f"Simulating amplitude equation for i0: ({i0})")
                simu_opts.update(i0=i0)
                _, amp, phi, _, _, _, _ = simulation(orientations=orients, **simu_opts)
                ra[-1][-1].append(amp)
                anga[-1][-1].append(phi)

    thetas = np.array(thetas)
    rt = np.array(rt)
    angt = np.array(angt)
    ra = np.array(ra)
    anga = np.array(anga)

    results = dict(stim=thetas, mod=rt, ang=angt, amp=ra, phi=anga)
    biased = '_biased' if biased else ''
    save_obj(results, f"./supplementary_alpha_{bump}{biased}_{size[0]}_{size[1]}", extension='.npy')
    return results


"""
Plotting functions
------------------


Plots that rely on a single ``dataframe`` obtained by running an instance of the 
simulation ``ring_simulation.py``. 
"""


@apply_custom_style
def plot_pm(dataframe, labels=('norm_category', 'norm_average_circ', 'pright'), xrange=(-0.5, 0.5),
            fig_kwargs=None, **kwargs):
    """ Plot the psychometric function.

    :param pd.DataFrame dataframe: simulation results.
    :param list of str labels: labels pointing towards the columns that contain the relevant data.
    :param (float, float) xrange: plotting range.
    :param dict or None fig_kwargs: dictionary with keyword arguments passed to the figure-axes constructor.
                                    If None, ax and fig keywords will be search in kwargs.
    :param kwargs: additional keyword arguments.
                   legend: keyword to control the legend (default: whole).
                   scatter: whether to plot the scatter plot or not. (default: True)
                   label: label of the main line, if a label is not provided, default is (``Category\\naverage``)
                   The rest of the keywords are passed to the plt.plot function.
    :return: figure, ax, and main plot
    :rtype: (plt.Figure, plt.Axes, plt.Line2D)
    """
    legend = kwargs.pop('legend', True)
    ylabel = kwargs.pop('ylabel', True)
    scatter = kwargs.pop('scatter', True)
    percentile = kwargs.pop('percentile', True)

    if fig_kwargs is None:
        fig_kwargs = dict(ax=kwargs.pop('ax', False), fig=kwargs.pop('fig', False))
    ax, fg, fig_kwargs = setup_ax(fig_kwargs)

    if scatter:
        alpha = 0.3 if percentile else 0.5
        marker_size = 0.1 if percentile else 0.5
        ax.scatter(dataframe[labels[1]], dataframe[labels[2]], marker='.', color='gray', s=marker_size, alpha=alpha,
                   label='Single\ntrials')

    for c in dataframe.columns:
        logging.debug(f"Column {c}: {dataframe[c].dtype}")
    group = dataframe.groupby(labels[0])
    if not kwargs.get('label', False):
        kwargs['label'] = 'Category\naverage'
    if percentile:
        median_category = group.median()
        quantile_75 = group.quantile(0.75)
        quantile_25 = group.quantile(0.25)
        selected_data = median_category.loc[np.abs(median_category.category) <= xrange[1] * 90].pright
        upper_quantile = quantile_75.loc[np.abs(quantile_75.category) <= xrange[1] * 90].pright - selected_data
        lower_quantile = -quantile_25.loc[np.abs(quantile_25.category) <= xrange[1] * 90].pright + selected_data
        p = ax.errorbar(np.array(selected_data.index), selected_data,
                        yerr=np.array([lower_quantile, upper_quantile]), fmt='-o', clip_on=False, linewidth=0.5,
                        capsize=1.0, **kwargs)
    else:
        # Mean values of trials grouped by category
        mean_category = group.mean()
        # mean_category.binchoice = mean_category.binchoice.apply(lambda x: 50.0 * (x + 1))
        # std_category = group.std()  # Not very useful as the distributions are non symmetrical in general
        # Select data that falls into the specified range (``xrange``)
        selected_data = mean_category.loc[np.abs(mean_category.category) <= xrange[1] * 90].pright
        p, = ax.plot(np.array(selected_data.index), selected_data, '-o', clip_on=False, linewidth=0.5, **kwargs)

    ax.set_xlim(*xrange)
    ax.set_xticks(np.linspace(*xrange, 5))
    ax.set_ylim(0, 100)
    ax.set_xlabel('Stimulus category')
    if ylabel:
        ax.set_ylabel(r"Probability right (\%)")
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    if legend:
        ax.legend(fontsize=6, frameon=False, loc='upper left')

    return fg, ax, p


@apply_custom_style
def plot_estimation(dataframe, scatter=False, bins=(16, 16), fig_kwargs=None, **kwargs):
    """ Plot an 2d-histogram or a scatter plot with density color mapping of the position of the bump (phase) at the end
    of the last stimulus frame.

    :param pd.DataFrame dataframe: data containing the simulation results.
    :param bool scatter: whether to plot the scatter plot or not.
    :param (int, int) bins: number of bins used for the 2d-histogram or the density plot (x, y).
    :param dict fig_kwargs: keyword arguments passed to the figure/ax constructor. If None, then ``ax`` and ``fig``
                            parameters will be search at kwargs.
    :param kwargs: additional keyword arguments passed to the histogram or scatter plot.
    :return: figure and axes containing the estimated orientation.
    :rtype: (plt.Figure, plt.Axes)
    """

    if fig_kwargs is None:  # Search for fig/axes keywords
        fig_kwargs = dict(ax=kwargs.pop('ax', False), fig=kwargs.pop('fig', False))

    ax, fg, fig_kwargs = setup_ax(fig_kwargs)  # Get axes and figure if none provided.

    lim = kwargs.pop('lim', [-90, 90])
    ticks = kwargs.pop('ticks', [-90, -45, 0, 45, 90])

    if scatter:
        density_scatter(dataframe.average_circ, dataframe.estim, bins=bins, ax=ax, **kwargs)
    else:
        ax.hist2d(dataframe.average_circ, dataframe.estim, bins, **kwargs)
    ax.plot(lim, lim, '--', color='black', linewidth=0.5)

    ax.set_xlabel(r'Average orientation ($^\circ$)')
    ax.set_ylabel(r'Estimated orientation ($^\circ$)')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.legend(fontsize=6, frameon=False, markerscale=10.0)

    return fg, ax


@apply_custom_style
def plot_estimation_avg(dataframe, set_ylabel=False, bin_width=1, fig_kwargs=None, **kwargs):
    """ Plot the average estimated orientation across trials and its confidence interval.

    :param pd.DataFrame dataframe: data containing the simulation results.
    :param bool set_ylabel: whether to set the ylabel or not.
    :param float bin_width: bin width for the estimation binning.
    :param dict fig_kwargs: keyword arguments passed to the figure/ax constructor. If None, then ``ax`` and ``fig``
                            parameters will be search at kwargs.
    :param kwargs: additional keyword arguments passed to line plot.
    :return: figure and axes containing the estimated orientation.
    :rtype: (plt.Figure, plt.Axes)
    """
    legend = kwargs.pop('legend', True)
    lim = kwargs.pop('lim', [-90, 90])
    ticks = kwargs.pop('ticks', [-90, -45, 0, 45, 90])

    if fig_kwargs is None:  # Search for fig/axes keywords
        fig_kwargs = dict(ax=kwargs.pop('ax', False), fig=kwargs.pop('fig', False))

    ax, fg, fig_kwargs = setup_ax(fig_kwargs)  # Get axes and figure if none provided.

    ax.plot(lim, lim, '--', color='black', linewidth=0.3)

    binned_estim = []
    avg_estim = []
    std_estim = []
    binning = np.arange(lim[0], lim[1] + bin_width, bin_width)
    for point in binning:  # Compute the mean and std of the binned data.
        binned_estim.append(dataframe.estim.loc[(dataframe.average_circ >= point) &
                                                (dataframe.average_circ < (point + bin_width))])
        avg_estim.append(binned_estim[-1].mean())
        std_estim.append(binned_estim[-1].std())
    avg_estim, std_estim = np.array(avg_estim), np.array(std_estim)
    ax.plot(binning, avg_estim, color='red', label='Avg.', **kwargs)
    ax.fill_between(binning, avg_estim - std_estim, avg_estim + std_estim,
                    color='lightsalmon', alpha=0.5, label='Std.')

    ax.set_xlabel(r'Average orientation ($^\circ$)')
    if set_ylabel:
        ax.set_ylabel(r'Estimated orientation ($^\circ$)')
        ax.set_yticks(ticks)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks(ticks)
    if legend:
        ax.legend(fontsize=6, frameon=False)

    return fg, ax


@apply_custom_style
def plot_ppk(mframe, legend='whole', fig_kwargs=None, **kwargs):
    """ Plot the Psychophysical kernel (impact vs. frame).

    :param pd.DataFrame mframe: dataframe containing the regressors and their errors of the logistic regression.
    :param bool legend: whether to set the legend or not.
    :param dict fig_kwargs: keyword arguments passed to the figure/ax constructor. If None, then ``ax`` and ``fig``
                            parameters will be search at kwargs.
    :param kwargs: additional keyword arguments passed to errorbar plot.
    :return: figure and axes containing the estimated orientation.
    :rtype: (plt.Figure, plt.Axes)
    """
    set_labels = kwargs.pop('set_labels', True)
    ylabel = kwargs.pop('ylabel', True)
    if fig_kwargs is None:  # Search for fig/axes keywords
        fig_kwargs = dict(ax=kwargs.pop('ax', False), fig=kwargs.pop('fig', False))

    ax, fg, fig_kwargs = setup_ax(fig_kwargs)  # Get axes and figure if none provided.

    # TODO: modify ``plot_error_filled`` in ``plotting_lib`` and
    #  include errorbar plot as an alternative to the line plot
    eb = ax.errorbar(np.arange(1, 9), mframe.beta, mframe.errors, **kwargs)
    color = eb.lines[0].get_color()

    fb = ax.fill_between(np.arange(1, 9), mframe.beta - mframe.errors, mframe.beta + mframe.errors, color=color,
                         alpha=0.2)
    if set_labels:
        eb.set_label(r'$\beta_i$')
        fb.set_label('Std.')

    ax.set_xticks(range(1, 9))
    ax.set_xlabel('Frame number')
    if ylabel:
        ax.set_ylabel('Frame impact')
    else:
        plt.setp(ax.get_yticklabels(), visible=False)

    if legend:
        ax.legend(fontsize=6, frameon=False)
    return fg, ax


@apply_custom_style
def plot_2dppk(df, dpi=300, cmap='RdBu_r', interpolate=False, title="No-bump\ncondition", bin_width=10, **kwargs):
    """Plots a 2-dimensional psychophysical kernel along with the averages across both axis (frame number and
    spatial dimension: orientation of the stimulus).

    :param pd.DataFrame df: data as a :py:mod:`pandas` data-frame format.
    :param int dpi: dots per inch of the plot.
    :param str cmap: colormap used for the 2D color-plot.
    :param bool interpolate: whether to interpolated and smooth the values of the 2D color-plot.
    :param str title: title of the figure.
    :param int bin_width: width of the bins used for the spatial binning.
    :param kwargs: additional keyword arguments.
    :return: created figure.
    :rtype: plt.Figure
    """
    # Set some variables
    mylog = kwargs.get('log', MyLog(10))
    bins = np.arange(-90 - bin_width // 2 - 1, 90 + bin_width // 2 + 1, bin_width)
    plt.rcParams['figure.dpi'] = dpi
    # Compute logistic regression
    logging.info('Fitting data to the logistic function...')
    logmodel, (regressors, bse) = log_reg2d(df, bin_width=bin_width)
    nframes = regressors.shape[0]
    mylog(0)

    # Create figure
    fig = plt.figure(figsize=(5, 3))

    # Create plot grid
    gs1 = gridspec.GridSpec(7, 1)
    gs2 = gridspec.GridSpec(7, 1)
    gs_bar = gridspec.GridSpec(27, 1)  # May be

    left = 0.1
    right = 0.75
    hspace = 0.3
    top = 0.97
    bottom = 0.12

    gs1.update(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=0.0)
    gs2.update(left=right + hspace / 15, right=0.97, top=top, bottom=bottom, hspace=hspace, wspace=0.0)
    gs_bar.update(left=right + hspace / 15, right=0.97, top=top, bottom=bottom, hspace=hspace * 2, wspace=0.0)

    ax_c = fig.add_subplot(gs1[3:, 0])
    ax_f = fig.add_subplot(gs1[0:3, 0], sharex=ax_c)

    # Color bar ax
    ax_bar = create_colorbar_ax(fig, gs_bar[9, 0])

    # Plot data
    logging.info('Plotting the logistic regression results...')
    vmax = np.ceil(np.max(np.abs(regressors)))

    if interpolate:
        f = interp2d(np.linspace(0, nframes + 1, nframes), bins[0:-1], regressors.T, kind='linear')
        xnew = np.arange(0, nframes + 1, .1)
        ynew = np.arange(-90 - bin_width // 2 - 1, 90 + bin_width // 2 + 1, bin_width / 10)
        # noinspection PyTypeChecker
        znew = f(xnew, ynew)
        pc = ax_c.pcolormesh(xnew, ynew, znew, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        pc = ax_c.pcolormesh(np.arange(0, nframes + 1), bins, regressors.T, cmap=cmap, vmin=-vmax, vmax=vmax)

    frame_v = np.arange(0, nframes)
    ax_c.set_ylim([np.min(bins), np.max(bins)])
    ax_c.set_xlim([0, nframes])
    ax_c.set_yticks([-90, -45, 0, 45, 90])
    ax_c.set_xticks(frame_v + 0.5)
    ax_c.set_xticklabels(frame_v + 1)
    ax_c.set_xlabel('Frame number')
    ax_c.set_ylabel(r'Frame orientation ($^\circ$)')

    cmap = pc.get_cmap()
    norm = MidpointNormalize(midpoint=0.0, vmin=-vmax, vmax=vmax)

    # Color bar plot:
    cbar = plt.colorbar(pc, cax=ax_bar, fraction=1.0, orientation='horizontal', aspect=15)
    cbar.set_ticks([-vmax, 0.0, vmax])
    cbar.ax.set_xlabel('Frame impact', ha='center', va='bottom')
    cbar.ax.xaxis.set_label_coords(0.5, 1.3)
    ax_s = fig.add_subplot(gs2[3:, 0], sharey=ax_c, sharex=cbar.ax)

    # Averages and marginals
    mean_ppk_orientations = regressors.mean(axis=0)
    std_ppk_orientations = np.sqrt(np.mean(bse ** 2, axis=0))
    mean_ppk_frames = np.abs(regressors).mean(axis=1)
    std_ppk_frames = np.sqrt(np.mean(bse ** 2, axis=1))

    for orientation in regressors.T:
        ax_f.plot(frame_v + 0.5, orientation, '-', color=cmap(norm(orientation.mean())),
                  linewidth=0.8, alpha=0.5)
    ax_f, m_p, m_eb = plot_error_filled(frame_v + 0.5, mean_ppk_frames, std_ppk_frames, '-o',
                                        color='black', alpha=.1, ax=ax_f)
    m_p.set_label(r'$\beta_f$')
    m_p.set_linewidth(0.5)
    m_eb.set_label(r'Std.')

    ax_f.plot([0, nframes + 1], [0, 0], '--', linewidth=0.2, color='black')
    ax_f.set_ylabel('Frame impact')
    ax_f.legend(fontsize=6, frameon=False)

    for beta_frame in regressors:
        ax_s.plot(beta_frame, bins[0:-1] + 5, '-', color='gray', linewidth=0.8, alpha=0.5)

    ax_s, o_p, o_eb = plot_error_filled(bins[0:-1] + bin_width // 2, mean_ppk_orientations, std_ppk_orientations,
                                        '-o', color='black', alpha=.1, indp_axis='y', ax=ax_s)
    o_p.set_label(r'$\beta_c$')
    o_p.set_linewidth(0.5)
    o_eb.set_label(r'Std.')

    ax_s.legend(fontsize=6, frameon=False)
    ax_s.plot([0, 0], [-90 - bin_width // 2, 90 + bin_width // 2], '--', linewidth=0.2, color='black')

    ax_f.set_ylim(-vmax, vmax)
    ax_f.set_yticks([-vmax, 0.0, vmax])

    # Format axes
    mod_axes(ax_f, y='right')
    mod_axes(ax_s, x='bottom')
    plt.setp(ax_f.get_xticklabels(), visible=False)
    plt.setp(ax_s.get_yticklabels(), visible=False)
    plt.setp(ax_s.get_xticklabels(), visible=False)

    # Set a title
    fig.text(0.97, 0.97, title, horizontalalignment='right', fontsize=10, va='top', ha='right')
    mylog(0)
    return fig


@apply_custom_style
def plot_bifurcation(bdata, save=True, righttoleft=False):
    """ Plots the amplitude at the top of the bump (phi = psi) as a function of the
    external input, for different noise levels.

    Creates a new figure for each value of i1.

    .. note::

        Depending on the way the bifurcation was done during the simulation it may be necessary to tweak some
        aspects of this function, such as the range of values that are plotted.

    :param dict bdata: data of the bifurcation. It should consist on a nested dictionary of the form:
        {value_of_i1_0: {value_of_sigma_0: (rates, bif_points, crit_point), value_of_sigma_1: ...}, value_of_i1_1: ...}
        where (rates, bif_points, crit_point) are the arrays/values corresponding to the rate of the network,
        the points of bifurcating parameter, and the critical point, respectively.
    :param bool save: whether to save or not the created figure(s).
    :param bool righttoleft: whether the bifurcation was computed 'from right to left' or not.
    :return: a tuple with the list of figures and axes generated by the function.
    :rtype: ( list of plt.Figure, list of plt.Axes )
    """

    # TODO: modify this function to be more flexible
    plt.rcParams['figure.dpi'] = 200

    # Load the data of the bifurcation done for zero noise and zero input:
    sigma0 = np.load('obj/bifurcation/bifurcation_sigma-0.npy', allow_pickle=True)

    # Range of inputs used for the theoretically obtained firing rate amplitude
    # TODO: this step should be done in a separate function
    ith0 = np.arange(0.0, 1.0, 0.0001)
    ith1 = np.arange(1.0, 1.5, 0.0001)

    w0 = -2
    tau = 0.02

    rth0 = (1.0 / tau) * (1.0 - 2 * ith0 * w0 - np.sqrt(1 - 4 * w0 * ith0)) / (2 * w0 ** 2)  # Amplitude of SHS
    rth1 = rth0[-1] + 2 * np.sqrt(2 * (ith1 - 1.0)) / tau  # 1st order approximation of the amplitude of the bump.

    ith = np.concatenate((ith0, ith1))
    rth = np.concatenate((rth0, rth1))
    # END of the theoretical computation

    figs = []
    axes = []
    for i1 in bdata:
        f, ax = plt.subplots(1, 1, figsize=(5, 4))
        f.set_tight_layout(False)
        figs.append(f)
        axes.append(ax)

        mod_axes(ax)
        cmap = plt.cm.get_cmap('Reds')
        norm = MidpointNormalize(midpoint=0.05, vmin=-0.1, vmax=0.2)

        ax.plot(sigma0[0][:-1] - 1, sigma0[1][:-1], color='olivedrab', linewidth=0.5, label=r'$\sigma = 0,\ I_1 = 0$')
        ax.plot(sigma0[0][:-1] - 1, sigma0[2][:-1], color='olivedrab', linewidth=0.5)
        ax.plot(ith - 1, rth, '--', color='skyblue', linewidth=0.5, label='Theory (1st order)')

        for sigma in bdata[i1]:
            rates, i0points, i0critical, amp_eq = bdata[i1][sigma]  # rates:(len(i0) x n_trials x n)
            # Shapes of matrices except a_rates: (len(i0points) x n_trials)
            a_rates, phases, amp_p, amp_a = get_phases_and_amplitudes_auto(rates, aligned_profiles=True)
            amp_p_avg = amp_p.mean(axis=1)
            amp_p_std = amp_p.std(axis=1)
            amp_a_avg = amp_a.mean(axis=1)
            amp_a_std = amp_a.std(axis=1)
            avg_rate = a_rates.mean(axis=(1, 2))  # Mean across trials and spatial dimension (neurons)
            # TODO: the ranges here are not general, they depend on the way the data was generated
            ax, p_p, eb_p = plot_error_filled(i0points, amp_p_avg, amp_p_std, ax=ax, color=cmap(norm(sigma)))
            p_p.set_label(r'$\sigma = %.2f$' % sigma)
            plot_error_filled(i0points, amp_a_avg, amp_a_std, color=cmap(norm(sigma)), ax=ax)
            ax.plot(i0points, avg_rate, linewidth=0.5, linestyle='dashed', color=cmap(norm(sigma)))
            ax.set_ylim(0, np.max(amp_p_avg * 1.1))

        ax.plot([0, 0], [0, 50], '--', color='black', linewidth=0.5)

        ax.set_xlabel(r'$I - I^T$', labelpad=10)
        ax.set_ylabel(r'Firing Rate (Hz)')

        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        ax.legend(frameon=False, fontsize=8)
        ax.set_xlim(0.9 - 1, 1.35 - 1)
        ax.set_ylim(0.0, 80)

        f.suptitle(r'$I_1 = %.3f$' % i1, fontsize=10)
        if righttoleft:
            ax.arrow(1.0, 1.0, -0.15, 0, fc='black', ec='black', width=0.02, transform=ax.transAxes, alpha=0.5,
                     head_width=0.05, head_length=0.05, length_includes_head=True, head_starts_at_zero=True,
                     clip_on=False)
        else:
            ax.arrow(0.8, 1.0, 0.15, 0, fc='black', ec='black', width=0.02, transform=ax.transAxes, alpha=0.5,
                     head_width=0.05, head_length=0.05, length_includes_head=True, head_starts_at_zero=True,
                     clip_on=False)

        if save:
            for fmt in ['png', 'pdf', 'svg']:
                f.savefig('bifurcation_amplitude_i1-%.3f.%s' % (i1, fmt), format=fmt, dpi=600)

    return figs, axes


@apply_custom_style
def plot_bifurcation_aq(bdata, save=True, righttoleft=False):
    """ Plots the amplitude at the top of the bump (phi = psi) as a function of the
    external input, for different noise levels.

    Creates a new figure for each value of i1.

    .. note::

        Depending on the way the bifurcation was done during the simulation it may be necessary to tweak some
        aspects of this function, such as the range of values that are plotted.

    :param dict bdata: data of the bifurcation. It should consist on a nested dictionary of the form:
        {value_of_i1_0: {value_of_sigma_0: (rates, bif_points, crit_point), value_of_sigma_1: ...}, value_of_i1_1: ...}
        where (rates, bif_points, crit_point) are the arrays/values corresponding to the rate of the network,
        the points of bifurcating parameter, and the critical point, respectively.
    :param bool save: whether to save or not the created figure(s).
    :param bool righttoleft: whether the bifurcation was computed 'from right to left' or not.
    :return: a tuple with the list of figures and axes generated by the function.
    :rtype: ( list of plt.Figure, list of plt.Axes )
    """

    # TODO: modify this function to be more flexible
    plt.rcParams['figure.dpi'] = 200

    # Load the data of the bifurcation done for zero noise and zero input:
    sigma0 = np.load('obj/bifurcation/bifurcation_sigma-0.npy', allow_pickle=True)

    # Range of inputs used for the theoretically obtained firing rate amplitude
    # TODO: this step should be done in a separate function
    ith0 = np.arange(0.0, 1.0, 0.0001)
    ith1 = np.arange(1.0, 1.5, 0.0001)

    w0 = -2
    tau = 0.02

    rth0 = (1.0 / tau) * (1.0 - 2 * ith0 * w0 - np.sqrt(1 - 4 * w0 * ith0)) / (2 * w0 ** 2)  # Amplitude of SHS
    rth1 = rth0[-1] + 2 * np.sqrt(2 * (ith1 - 1.0)) / tau  # 1st order approximation of the amplitude of the bump.

    ith = np.concatenate((ith0, ith1))
    rth = np.concatenate((rth0, rth1))
    # END of the theoretical computation

    figs = []
    axes = []
    for i1 in bdata:
        f, ax = plt.subplots(1, 1, figsize=(5, 4))
        f.set_tight_layout(False)
        figs.append(f)
        axes.append(ax)

        mod_axes(ax)
        cmap = plt.cm.get_cmap('Reds')
        norm = MidpointNormalize(midpoint=0.05, vmin=-0.1, vmax=0.2)

        ax.plot(sigma0[0][:-1] - 1, sigma0[1][:-1], color='olivedrab', linewidth=0.5, label=r'$\sigma = 0,\ I_1 = 0$')
        ax.plot(sigma0[0][:-1] - 1, sigma0[2][:-1], color='olivedrab', linewidth=0.5)
        ax.plot(ith - 1, rth, '--', color='skyblue', linewidth=0.5, label='Theory (1st order)')
        r0 = rth0[-1]
        for sigma in bdata[i1]:
            rates, i0points, i0critical, amp_eq = bdata[i1][sigma]  # rates:(len(i0) x n_trials x n)
            # Shapes of matrices except a_rates: (len(i0points) x n_trials)
            amp = 2.0 * np.sqrt(np.real(amp_eq * np.conj(amp_eq)))
            amp_avg = amp.mean(axis=1)
            amp_std = amp.std(axis=1)
            ax, p_a, eb_a = plot_error_filled(i0points, r0 + amp_avg, amp_std, ax=ax, color=cmap(norm(sigma)))
            p_a.set_label(r'$\sigma = %.2f$' % sigma)
            ax.set_ylim(0, np.max(amp_avg * 1.1))

        ax.plot([0, 0], [0, 50], '--', color='black', linewidth=0.5)

        ax.set_xlabel(r'$I - I^T$', labelpad=10)
        ax.set_ylabel(r'$R$ (Hz)')

        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        ax.legend(frameon=False, fontsize=8)
        ax.set_xlim(0.9 - 1, 1.35 - 1)
        ax.set_ylim(0.0, 80)

        f.suptitle(r'$I_1 = %.3f$' % i1, fontsize=10)
        if righttoleft:
            ax.arrow(1.0, 1.0, -0.15, 0, fc='black', ec='black', width=0.02, transform=ax.transAxes, alpha=0.5,
                     head_width=0.05, head_length=0.05, length_includes_head=True, head_starts_at_zero=True,
                     clip_on=False)
        else:
            ax.arrow(0.8, 1.0, 0.15, 0, fc='black', ec='black', width=0.02, transform=ax.transAxes, alpha=0.5,
                     head_width=0.05, head_length=0.05, length_includes_head=True, head_starts_at_zero=True,
                     clip_on=False)

        if save:
            for fmt in ['png', 'pdf', 'svg']:
                f.savefig('bifurcation_amplitude_equation_i1-%.3f.%s' % (i1, fmt), format=fmt, dpi=600)

    return figs, axes


@apply_custom_style
def plot_bump_profiles(bdata, save=True):
    """ Function that creates a figure(s) with a set of plots of the bump for different values of the external
    current. It includes a colormap for the evolution of the bump, and several spatial profiles of the
    firing rate for different values of the external current.

    .. note::

        The function will create a figure per each value of the noise.

    :param dict bdata: data of the bifurcation. It should consist on a nested dictionary of the form:
        {value_of_i1_0: {value_of_sigma_0: (rates, bif_points, crit_point), value_of_sigma_1: ...}, value_of_i1_1: ...}
        where (rates, bif_points, crit_point) are the arrays/values corresponding to the rate of the network,
        the points of bifurcating parameter, and the critical point, respectively.    
    :param bool save: whether to save or not the created figure(s).
    :return: a list of figure(s).
    :rtype: list of plt.Figure
    """

    plt.rcParams['figure.dpi'] = 200
    figsize = np.array([4.5, 3]) * 1.3
    figs = []
    # TODO: make this more general (identify ``i0points`` and ``n`` from arguments)
    theta = np.arange(200) / 200 * 180 * 2 - 180

    # Design the grid of axes (positions, dimensions, etc.
    gs1 = gridspec.GridSpec(1, 3)
    gs2 = gridspec.GridSpec(1, 3)
    gs3 = gridspec.GridSpec(1, 3)
    gs_bar = gridspec.GridSpec(1, 1)

    left = 0.1
    rightp = 0.9
    hspace = 0.3
    wspace = 0.2

    rfactor = 0.8

    h1 = 1 / 5 * rfactor
    h2 = 1.6 / 5 * rfactor
    h3 = 1.8 / 5 * rfactor

    top1 = 0.95
    bottom1 = top1 - h1
    top2 = bottom1 - hspace / 10
    bottom2 = top2 - h2
    top3 = bottom2 - hspace / 3
    bottom3 = top3 - h3

    gs1.update(left=left, right=rightp, top=top1, bottom=bottom1, hspace=hspace, wspace=wspace)
    gs2.update(left=left, right=rightp, top=top2, bottom=bottom2, hspace=hspace, wspace=wspace)
    gs3.update(left=left, right=rightp, top=top3, bottom=bottom3, hspace=hspace, wspace=wspace)
    gs_bar.update(left=rightp + hspace / 20, right=rightp + hspace / 12, top=top1, bottom=bottom1, hspace=hspace,
                  wspace=0)

    # Compute things and plot
    for i1 in bdata:
        for sigma in bdata[i1]:
            rdata, i0points, i0critical, amp_eq = bdata[i1][sigma]  # rates: (len(i0) x n_trials x n)
            # Shapes of matrices except a_rates: (len(i0points) x n_trials)
            a_rates, phases, amp_p, amp_a = get_phases_and_amplitudes_auto(rdata, aligned_profiles=True)
            amp_p_avg = amp_p.mean(axis=1)
            amp_a_avg = amp_a.mean(axis=1)

            rsample = rdata[:, 0, :]  # Single trial (bif_points x n)

            # Align bumps to 0
            rmean = a_rates.mean(axis=1)  # Average across trials
            f = plt.figure(figsize=figsize)
            f.set_tight_layout(False)

            # Create plot grid
            ax_c = f.add_subplot(gs1[0, 0:])
            ax_amp = f.add_subplot(gs2[0, 0:], sharex=ax_c)

            ax_p1 = f.add_subplot(gs3[0, 0], sharey=ax_amp)
            ax_p2 = f.add_subplot(gs3[0, 1], sharey=ax_p1, sharex=ax_p1)
            ax_p3 = f.add_subplot(gs3[0, 2], sharey=ax_p1, sharex=ax_p1)

            mod_axes([ax_amp, ax_p1, ax_p2, ax_p3])

            ax_bar = f.add_subplot(gs_bar[0, 0])
            ax_bar.clear()
            ax_bar.set_xticks([])
            ax_bar.set_yticks([])
            for spine in ['bottom', 'right', 'top', 'left']:
                ax_bar.spines[spine].set_visible(False)

            # Plot data
            bump = ax_c.pcolormesh(i0points, theta, rsample.T, vmin=0)
            bump.set_edgecolor('face')
            label = (r' $\sigma = %.2f$' % sigma) + '\n' + (r'$I_1 = %.3f$' % i1)
            ax_amp.plot(i0points, amp_p_avg, label=label)
            ax_amp.plot(i0points, amp_a_avg)

            ax_amp.plot([0, 0], [0, np.max(amp_p_avg * 1.5)], '--', color='black', linewidth=0.5)
            ax_amp.set_ylim(0, np.max(amp_p_avg * 1.1))
            ax_amp.set_xlabel(r'$I_0 = I - I^T$', labelpad=3)
            ax_amp.set_ylabel(r'Firing Rate (Hz)')

            for i0, ax_p, color, mark in zip([0.05, 0.1, 0.3], [ax_p1, ax_p2, ax_p3],
                                             ['olivedrab', 'sandybrown', 'indigo'], ['s', 'o', '^']):
                i0_indx = np.argwhere(np.abs(i0points - i0) < 0.005).ravel()[0]
                ax_amp.plot([i0, i0], [0, 180], '--', color=color)
                ax_amp.plot([i0 + 0.01], [np.max(amp_p_avg) * 1.1], mark, color=color, markersize=5, clip_on=False)

                ax_p.plot(theta, rsample[i0_indx], label=r'Trial')
                ax_p.plot(theta, rmean[i0_indx], '--', color='black', label='Avg.', linewidth=0.5)
                ax_p.plot(0.9, 0.9, mark, color=color, transform=ax_p.transAxes, markersize=5)
                ax_p.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax_p.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax_p.set_xticks([-180, -90, 0, 90, 180])
                # ax_p.legend(frameon=False, loc='upper left', fontsize=6)
            ax_p1.legend(frameon=False, loc='upper left', fontsize=8)

            # Color bar
            cbar = plt.colorbar(bump, cax=ax_bar, fraction=1.0, orientation='vertical', aspect=15)
            cbar.set_label('FR (Hz)', fontsize=10)
            cbar.set_ticks([0.0, 75])
            cbar.solids.set_edgecolor("face")

            ax_p1.set_xlim(-180, 180)
            ax_p1.set_ylabel(r'Firing Rate (Hz)')
            ax_c.set_ylim(-180, 180)
            ax_c.set_xlim(i0points[1], i0points[-1])
            ax_c.set_yticks([-180, 0, 180])

            ax_c.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax_c.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            ax_p2.set_xlabel(r'$\theta$ ($^\circ$)')
            ax_c.set_ylabel(r'$\theta$ ($^\circ$)', rotation=0)
            ax_amp.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax_amp.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            ax_amp.legend(frameon=False, fontsize=8)

            plt.setp(ax_c.get_xticklabels(), visible=False)
            for axp in [ax_p2, ax_p3]:
                plt.setp(axp.get_yticklabels(), visible=False)

            if save:
                for fmt in ['png', 'pdf', 'svg']:
                    f.savefig('bifurcation_summary_sigma-%.2f_i1-%.3f.%s' % (sigma, i1, fmt), format=fmt, dpi=600)
            figs.append(f)

    return figs


@apply_custom_style
def plot_profiles_comp(bdata, save=True, i0=0.05):
    """ Function that creates a figure(s) with a set of plots showing the bump profile (firing rate of the network
    at a given point of the bifurcation) for different values of the amplitude of the stimuli and noise levels.

    .. note::

        The function will create a figure per each value of the noise.

    :param dict bdata: data of the bifurcation. It should consist on a nested dictionary of the form:
        {value_of_i1_0: {value_of_sigma_0: (rates, bif_points, crit_point), value_of_sigma_1: ...}, value_of_i1_1: ...}
        where (rates, bif_points, crit_point) are the arrays/values corresponding to the rate of the network,
        the points of bifurcating parameter, and the critical point, respectively.
    :param bool save: whether to save or not the created figure(s).
    :param float i0:  value of the constant external current for which the firing rate profile will be plotted.
    :return: a list of figure(s).
    :rtype: list of plt.Figure
    """

    n_trials = 10
    theta = np.arange(200) / 200 * 180 * 2 - 180
    plt.rcParams['figure.dpi'] = 200
    figs = []

    for i1 in bdata:
        f, axes = plt.subplots(1, 2, figsize=(6, 2.8), sharex='all', sharey='all',
                               gridspec_kw=dict(wspace=0.1, top=0.92, bottom=0.16, left=0.1, right=0.98))
        f.set_tight_layout(False)
        mod_axes(axes)
        ymax = 0
        for sigma in bdata[i1]:
            rdata, i0points, critical_point, amp_eq = bdata[i1][sigma]
            i0_indx = np.argwhere(np.abs(i0points - i0) < 0.005).ravel()[0]
            n_trials = rdata.shape[1]
            rsample = rdata[:, 0]  # First trial
            # Get the aligned data:
            align_rdata = align_profiles(rdata)
            rmean = align_rdata.mean(axis=1)  # Mean across trials

            axes[0].plot(theta, rsample[i0_indx], linewidth=0.5)
            axes[1].plot(theta, rmean[i0_indx], label=r'$\sigma = %.2f$' % sigma)
            aux_max = np.max(rdata[i0_indx])
            ymax = aux_max if ymax < aux_max else ymax

        axes[0].set_xlim(-180, 180)
        axes[0].set_xticks([-180, -90, 0, 90, 180])

        ymax = 24 if ymax < 24 else ymax * 1.1

        axes[0].set_ylim(0, ymax)
        axes[1].legend(frameon=False, fontsize=8, loc='upper left')
        axes[0].text(1.0, 1.0, 'Sample trial', transform=axes[0].transAxes, ha='right', va='top')
        axes[1].text(1.0, 1.0, r'Average $(n = %d)$' % n_trials, transform=axes[1].transAxes, ha='right', va='top')
        f.suptitle(r'$I_1 = %.3f$, $I_0 = %.2f$' % (i1, i0), fontsize=10)

        for k in [0, 1]:
            axes[k].set_xlabel(r'$\theta$ ($^\circ$)')

        axes[0].set_ylabel(r'Firing Rate (Hz)')

        axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        plt.setp(axes[1].get_yticklabels(), visible=False)
        if save:
            for fmt in ['png', 'pdf', 'svg']:
                f.savefig('profiles_at_i0-%.2f_i1-%.3f.%s' % (i0, i1, fmt), format=fmt, dpi=600)

        figs.append(f)

    return figs


# ##############################################################################################
@apply_custom_style
def plot_trial(r_data, dataframe, i0_data, figsize=np.array([5.0, 4.5]) * 1.2, save=False, **kwargs):
    """Plots a figure that summarizes the results of a single randomly chosen (see constrains) trial.

    :param Cython.Includes.numpy.ndarray r_data: firing rate data.
    :param pd.DataFrame dataframe: data frame that includes the simulation design and the results.
    :param Cython.Includes.numpy.ndarray figsize: Figure size (width, heigth) in inches.
    :param bool save: whether to save or not the created figure(s).
    :param kwargs: additional keyword arguments.
    :return: a list of figure(s).
    :rtype: list of plt.Figure
    """
    if not isinstance(r_data, np.ndarray):
        try:
            r_data = np.load(str(r_data), allow_pickle=True)
            dataframe = pd.read_csv(dataframe)
        except IOError:
            return -1

    # Check dataframe's format
    try:
        assert (isinstance(dataframe, pd.DataFrame))
    except AssertionError:
        logging.critical("Variable 'dataframe' must be a dataframe.")
        return -1

    # Simulation parameters
    try:
        (tsteps, ntrials, n) = r_data.shape
    except ValueError:
        (tsteps, n) = r_data.shape
        ntrials = 1
        r_data = r_data.reshape((tsteps, 1, n))

    if n <= 1:
        logging.critical('The firing rate data is dummy data or has bad format: n = %d.' % n)
        logging.info('Skipping plotting...')
        return None, None
    logging.debug('The shape of the firing rate matrix is: (%d, %d, %d).' % (tsteps, ntrials, n))
    theta = np.arange(n) / n * 180 * 2 - 180

    t0 = -2000 * kwargs.get('dt', 0.0002)
    dts = kwargs.get('save_interval', 0.01)
    tpoints = np.arange(0, tsteps) * dts + t0
    tmax = tpoints[-1]
    tmin = tpoints[0]

    cue = kwargs.get('cue_duration', 0.250)

    # Data post-processing:
    r_aligned, phases, amps, amps_m = get_phases_and_amplitudes_auto(r_data, aligned_profiles=True)

    # #### Example trial plot (colormap, amplitude, phase, stimuli, profiles)
    trial = kwargs.pop('selected_trial', None)
    if trial is None:
        bound_category = kwargs.pop('max_category', 45)  # Find a trial whose category is less than bound_category
        category = 100  # Initialization
        trials = list(range(ntrials))
        trial = 0
        data_trial = dataframe.iloc[trial]
        while np.abs(category) > bound_category and trials:
            trial = trials.pop(np.random.randint(0, len(trials)))  # Select a trial randomly
            data_trial = dataframe.iloc[trial]
            category = data_trial['category']
        logging.debug('Semi-Randomly selected trial: %d' % trial)
    else:
        data_trial = dataframe.iloc[trial]
        category = data_trial['category']

    # For debugging:
    logging.debug(data_trial)

    r = r_data[:, trial, :]
    r_a = r_aligned[:, trial, :]
    phase = np.rad2deg(phases[:, trial])
    amp, ampm = (amps[:, trial], amps_m[:, trial])

    # I0 evolution
    i0 = i0_data[:]

    # Setting the figure
    custom_plot_params(1.2, latex=True)
    plt.rcParams['figure.dpi'] = 200

    # Grid(s)
    gs1 = gridspec.GridSpec(4, 1)
    gs_bar = gridspec.GridSpec(4, 1)

    # Plot margins and grid margins
    left = 0.1
    right_1 = 0.9
    hspace = 0.3
    wspace = 0.2

    # G1
    (top1, bottom1) = (0.95, 0.1)
    gs1.update(left=left, right=right_1, top=top1, bottom=bottom1, hspace=hspace, wspace=wspace)

    # Gbar
    (top_bar, bottom_bar, left_bar, right_bar) = (top1, bottom1, right_1 + 0.01, right_1 + 0.03)
    gs_bar.update(left=left_bar, right=right_bar, top=top_bar, bottom=bottom_bar, hspace=hspace, wspace=wspace)

    # Figure and Axes
    fig = plt.figure(figsize=figsize)
    fig.set_tight_layout(False)

    ax_amp = fig.add_subplot(gs1[3, 0])
    ax_ph = fig.add_subplot(gs1[0, 0], sharex=ax_amp)
    ax_c = fig.add_subplot(gs1[2, 0], sharex=ax_ph)
    ax_i0 = fig.add_subplot(gs1[1, 0], sharex=ax_ph)

    mod_axes([ax_amp, ax_ph, ax_i0])
    ax_ph.spines['bottom'].set_position('center')

    ax_bar = create_colorbar_ax(fig, gs_bar[2, 0])  # Ax for the color-bar of the color-plot

    # Actual plotting starts here
    # Amplitude of the bump
    ax_amp.plot(tpoints[:-1], amp[:-1], label=r'$r(\psi, t)$')
    ax_amp.plot(tpoints[:-1], ampm[:-1], label=r'$r(\psi+\pi, t)$')

    ax_amp.legend(frameon=False, fontsize=8)
    ax_amp.set_xlim(tmin, tmax)
    ax_amp.set_ylim(0, np.max(r) * 1.01)
    ax_amp.set_ylabel('Firing Rate (Hz)', labelpad=10)
    ax_amp.set_xlabel('Time (s)')

    # Phase evolution and stimuli
    nframes = kwargs.get('nframes', 8)
    labels = ['x%d' % k for k in range(1, nframes + 1)]
    stim_phases = data_trial[labels].to_numpy(int)
    stim_times = np.arange(0, nframes) * cue
    logging.debug('Orientations of the frames are: %s' % stim_phases)
    deltat = nframes * cue
    stim_t = int(deltat / dts)
    i1s = np.full(stim_t, 0.005 * dts / 2E-4)
    thetas = np.zeros(stim_t)
    for m, th in enumerate(stim_phases):
        t0 = int(cue / dts) * m
        t1 = int(cue / dts) * (m + 1)
        thetas[t0:t1] = np.repeat(np.deg2rad(th), t1 - t0)

    pvi = np.rad2deg(np.angle(np.cumsum(i1s * np.exp(1j * thetas))))

    stim_color = mcolors.to_rgba('cornflowerblue', 0.3)
    ax_ph.bar(stim_times, stim_phases, align='edge', width=cue * 1.0, lw=1.0, ec='cornflowerblue',
              fc=stim_color, label=r'$\theta_i^{\text{stim}}$')
    ax_ph.plot(tpoints[:-1], phase[:-1], color='black', linewidth=1.5, label=r'$\psi(t)$')
    ax_ph.plot(np.arange(0, len(pvi)) * dts, pvi, color='red', linewidth=1.5, label=r'PVI')
    ax_ph.legend(frameon=False, fontsize=8, ncol=2)
    ax_ph.set_ylim(-100, 100)
    ax_ph.set_yticks([-90, -45, 0, 45, 90])
    ax_ph.set_ylabel(r'$\theta\ (^\circ)$')
    plt.setp(ax_ph.get_xticklabels(), visible=False)

    # Colorplot of the bump
    bumpc = ax_c.pcolormesh(tpoints, theta, r.T, vmin=0)
    bumpc.set_edgecolor('face')
    ax_c.set_ylabel(r'$\theta\ (^\circ)$', labelpad=-1)
    ax_c.set_ylim(-180, 180)
    ax_c.set_yticks([-180, -90, 0, 90, 180])
    # ax_c.plot(tpoints[:-1], phase[:-1], color='black', linewidth=1.5, label=r'$\psi(t)$')
    plt.setp(ax_c.get_xticklabels(), visible=False)

    # Colorbar
    cbar = plt.colorbar(bumpc, cax=ax_bar, fraction=1.0, orientation='vertical', aspect=15)
    cbar.set_label('Firing Rate (Hz)', fontsize=8)
    cbar.solids.set_edgecolor("face")

    # I0 evolution
    ax_i0.plot(tpoints[:-1], i0[:-1])
    ax_i0.plot(tpoints[:-1], np.full_like(tpoints[:-1], fill_value=(1.0 + 2.0 * kwargs['sigmaOU']**2)), '--', color='black', lw=0.5)
    ax_i0.set_xlim(tmin, tmax)
    ax_i0.set_ylim(np.min(i0) * 1.2, np.max(i0) * 1.2)
    ax_i0.set_ylabel(r'$I_{\text{exc}}$', labelpad=10)
    plt.setp(ax_i0.get_xticklabels(), visible=False)
    ax_i0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Change ticklabels format (from latex to plain text)
    for ax in [ax_amp, ax_ph, ax_c]:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax_amp.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # fig_grid(fig, shift=0.4)  # Plot a grid to get an idea of the dimensions of the subplot
    # fig_grid(fig, 'v', shift=0.1)  # Plot a grid to get an idea of the dimensions of the subplot
    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        fig.set_tight_layout(False)
        sigma, i0 = itemgetter('sigmaOU', 'i0')(kwargs)
        bump = 'nobump' if kwargs.get('nobump', False) else 'bump'
        if np.all([i0, sigma]):
            filename = directory + f"trial_{bump}_{trial}_summary_i0-{i0:.2f}_sigma-{sigma:.2f}"
        else:
            filename = directory + f"summary_{bump}_t-{trial}_{ntrials}_trials"

        # Save the vectorial part only
        bumpc.set_visible(False)
        ccbar = ax_bar.collections[0]
        ccbar.set_visible(False)
        save_plot(fig, filename + "_vector", saveformat=['svg'], **kwargs)

        # Restore colorplots
        bumpc.set_visible(True)
        ccbar.set_visible(True)

        # Remove rest of the elements
        # for patch in patches:
        #     patch.set_visible(False)
        # for artist in artists:
        #     artist.set_visible(False)
        axes = [ax_ph, ax_i0, ax_amp]
        for ax in axes:
            ax.remove()

        for ax in [ax_c, ax_bar]:
            plt.setp(ax.get_children()[1:], visible=False)
        save_plot(fig, filename + "_raster", saveformat=['png'], **kwargs)

    return fig, ax_ph


@apply_custom_style
def plot_stats_summ(df, sigma, i0, lbl_fmt="x[0-9][0-9]?_norm", save=False, **kwargs):
    """Plots summary statistics of the simulation. PM function, PPk, and estimation plots.

    :param df: data-frame containing simulation results.
    :param sigma: value of :math:`\sigma_{\text{OU}}`.
    :param i0: value of :math:`I_0` where :math:`I = I_{\text{crit}} + I_0`.
    :param lbl_fmt: format of the label pointing to the design matrix as regexp.
    :param bool save: whether to save or not the created figure(s).
    :return: figure and axes.
    :rtype: (plt.Figure, np.ndarray of plt.Axes)
    """
    mylog = kwargs.pop('mylog', MyLog(10))
    # Get labels from the data-frame
    labels = re.findall(f"({lbl_fmt})", ' '.join(df.columns))
    logging.info('Fitting data to the logistic function ...')
    if kwargs.get('old_ppk', False):
        df['binchoice_old'] = (df.estim >= 0) * 2 - 1
        results = log_reg(df, labels, ylabel='binchoice_old')
    else:
        results = log_reg(df, labels)
    if isinstance(results, (tuple, list)):
        mylog(0)
        do_plot_ppk = True
        m_r, mf_r, df = results
    else:
        mylog.msg(1, up_lines=2)
        do_plot_ppk = False
        m_r, mf_r, df = (None, None, df)

    fig, axs, kw = set_plot_environment(nrows=2, ncols=2, figsize=(5, 4), dpi=200,
                                        gridspec_kw=dict(top=0.92, hspace=0.5, wspace=0.3, right=0.97, left=0.1))
    if do_plot_ppk:
        logging.info('Plotting psychometric function ...')
        fig, ax_pm, p = plot_pm(df, ax=axs[0, 0], fig=fig)
        mylog(0)
        logging.info('Plotting psychophysical kernel ...')
        fig, ax_ppk = plot_ppk(mf_r, linewidth=0.5, marker='o', fig=fig, ax=axs[0, 1])
        ax_ppk.set_ylim(0, np.max(mf_r.beta) * 1.1)
        mylog(0)
        if kwargs.get('old_ppk', False):
            logging.info('Plotting the psychophysical kernel without the readout circuit ...')
            logging.info('Fitting estimation-based choice data to the logistic function ...')
            df['binchoice_old'] = (df.estim >= 0) * 2 - 1
            # Check if this 'readout' is different from that of the decision network
            errors = np.sum(not np.alltrue(df['binchoice_old'] == df['binchoice']))
            if errors > 0:
                logging.warning(f"Categorical readout is not optimal: {errors / float(len(df))}\% of errors.")
            results2 = log_reg(df, labels, ylabel='binchoice_old', compute_pm=False)
            if isinstance(results2, (tuple, list)):
                mylog(0)
                m_r, mf_r, df2 = results
                fig, ax_ppk = plot_ppk(mf_r, linewidth=0.5, marker='o', fig=fig, ax=axs[0, 1], legend=False,
                                       set_labels=False, zorder=0)
                last_plot = ax_ppk.lines[-1]
                last_plot.set_label("Estimation-based")
                ax_ppk.legend(frameon=False, fontsize=5)
                mylog.msg(0, up_lines=2)
            else:
                mylog.msg(1, up_lines=2)

    logging.info('Plotting orientation vs. estimation figures.')
    fig, ax_estim = plot_estimation(df, fig=fig, ax=axs[1, 0], bins=(16, 16), scatter=True, marker='.',
                                    s=0.1, label='Single\ntrials')
    fig, ax_estim_avg_r = plot_estimation_avg(df, fig=fig, ax=axs[1, 1])
    mylog(0)

    title = "Bump" if kwargs.get('bump', True) else "No-bump"
    fig.suptitle(title + r' condition: $\sigma = %.2f$, $I_0 = %.3f$' % (sigma, i0), fontsize=10)
    # noinspection PyTypeChecker
    fig.set_tight_layout(False)

    if save:
        ntrials = kwargs.get('ntrials', 0)
        directory = kwargs.get('fig_dir', 'figs/')
        if np.all([i0, sigma]):
            filename = directory + f'stats_summary_{title}_i0-{i0:.2f}_sigma-{sigma:.2f}'
        else:
            filename = directory + f'stats_summary_{title}_{ntrials}_trials'
        save_plot(fig, filename, **kwargs)

    return fig, axs


@apply_custom_style
def plot_rp(dfs, labels, prmt_list, fix_prmt, midpoint=0.05, prmt_labels=(r"\sigma", r"I_0"), bump=False,
            ctitle='', colormap='coolwarm', override_max=False, show=True):
    """Generate a figure that shows the psychometric function next to the psychophysical kernel for a set of
    different parameter values to show the transition from primacy to recency integration.

    :param list of pd.DataFrames dfs:
    :param list of str labels: labels that determine the design matrix.
    :param list or np.ndarray prmt_list: list of values of the moving parameter.
    :param tuple of str prmt_labels: a tuple with the labels of the (fix, moving) parameters.
    :param float fix_prmt: value of the fix parameter.
    :param float midpoint: midpoint of the normalized colorbar.
    :param int or bool bump: whether we are in the bump or no-bump condition.
    :param str ctitle: title of colorbar for the the moving parameter.
    :param colormap: colormap scheme.
    :param bool override_max: whether to override the maximum value of the colormap.
    :param bool show: show the plot (default=True).
    :return: figure and axes.
    :rtype: (plt.Figure, list of plt.Axes)
    """
    # Set the plot environment
    gridspec_kw = dict(bottom=0.2, top=0.8, hspace=0.5, wspace=0.4, right=0.97, left=0.1)
    fig, axs, kw = set_plot_environment(nrows=1, ncols=2, figsize=(5, 2.5), dpi=200, gridspec_kw=gridspec_kw)

    # Custom colors from a normalized colormap
    mov_prmt_list_norm = (prmt_list - np.min(prmt_list)) / np.ptp(prmt_list)
    norm = MidpointNormalize(midpoint=midpoint, vmin=np.min(prmt_list), vmax=np.max(prmt_list))
    if override_max:
        mov_prmt_list_norm = (prmt_list - np.min(prmt_list)) / (
                1.2 * np.max(prmt_list) - np.min(prmt_list))
        norm = MidpointNormalize(midpoint=midpoint, vmin=np.min(prmt_list), vmax=1.2 * np.max(prmt_list))

    fix_label, mov_label = prmt_labels
    if mov_label == r"I_0":
        cmap = plt.cm.get_cmap(colormap + '_r')
    else:
        cmap = plt.cm.get_cmap(colormap)

    # Color bar
    gs_bar = gridspec.GridSpec(1, 1)  #
    gs_bar.update(left=0.68, right=0.96, top=0.83, bottom=0.8, hspace=0.0, wspace=0.0)
    cax = fig.add_subplot(gs_bar[0, 0])
    cax.clear()
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
    cbar.solids.set_edgecolor("face")

    pm_lines = []
    for df, prmt, prmt_norm in zip(dfs, prmt_list, mov_prmt_list_norm):
        color = cmap(norm(prmt))
        m_r, mf_r, df = log_reg(df, labels)
        fig, ax_pm, p = plot_pm(df, ax=axs[0], fig=fig, scatter=False, label=r"$" + mov_label + r' = %.2f$' % prmt,
                                color=color, legend=False)
        pm_lines.append(p)
        fig, ax_ppk = plot_ppk(mf_r, linewidth=0.5, marker='o', fig=fig, ax=axs[1], legend=False, color=color)

        cax.plot([prmt_norm], [-0.5], '^', clip_on=False, color=color, markersize=4.0)

    if ctitle == '':
        ctitle = r'External input' if fix_label == r"I_0" else r'Noise level'

    axs[0].legend((pm_lines[0], pm_lines[-1]), (pm_lines[0].get_label(), pm_lines[-1].get_label()), frameon=False,
                  title=ctitle, fontsize=8, title_fontsize=8)

    axs[1].plot([0, 8.2], [0, 0], '--', linewidth=0.5, color='black')
    axs[1].set_xlim(0, 8.2)
    figtitle = r'Bump condition: ' if bump else r'No-bump condition: '
    figtitle = figtitle + "$" + fix_label + r' = %.2f$' % fix_prmt
    fig.suptitle(figtitle, fontsize=10)

    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cax.set_xlabel(ctitle + r' ($' + fix_label + '$)', fontsize=8, ha='right', va='bottom')
    cax.xaxis.set_label_coords(1.0, 1.3)

    if show:
        plt.show()

    return fig, axs


@apply_custom_style
def plot_decision_circ(sdata, rdata, ddata, opts, figsize=(5, 6), save=False, **kwargs):
    """Plots a figure that summarizes the dynamics of the decision circuit.

    :param pd.DataFrame sdata: dataframe with the results of the simulation.
    :param np.ndarray rdata: firing rate data of the bump.
    :param dict ddata: firing rate and input signal of the decision signal stored in a dictionary.
    :param dict opts: dictionary of parameters and simulation options.
    :param Cython.Includes.numpy.ndarray figsize: Figure size (width, heigth) in inches.
    :param bool save: whether to save or not the created figure(s).
    :param kwargs: additional keyword arguments.
    :return: a list of figure(s).
    :rtype: list of plt.Figure
    """

    # Extract firing rate of the populations of the decision circuit and the input arriving from the ring-network
    d1 = ddata['d12'][:-1, 0]
    d2 = ddata['d12'][:-1, 1]
    li = ddata['lri'][:-1, 0]
    ri = ddata['lri'][:-1, 1]

    # Select all the left-trials (trials that ended with the bump on the left).
    left_trials = (sdata.binchoice < 0)
    # Arrange trials by their preferred category
    pr = np.concatenate((d1[:, left_trials], d2[:, ~left_trials]), axis=1)
    apr = np.concatenate((d1[:, ~left_trials], d2[:, left_trials]), axis=1)
    pi = np.concatenate((li[:, left_trials], ri[:, ~left_trials]), axis=1)
    api = np.concatenate((li[:, ~left_trials], ri[:, left_trials]), axis=1)

    # Compute mean and standard deviation
    prm = pr.mean(axis=1)
    prs = pr.std(axis=1)
    aprm = apr.mean(axis=1)
    aprs = apr.std(axis=1)

    pim = pi.mean(axis=1)
    pis = pi.std(axis=1)
    apim = api.mean(axis=1)
    apis = api.std(axis=1)

    # Select a sample of 10 trials
    tsteps, ntrials = d1.shape
    if ntrials > 10:
        random_choice = np.random.choice(ntrials, 10, replace=False)
    else:
        random_choice = np.arange(ntrials)

    dts = kwargs.get('save_interval', 0.01)
    tpoints = np.arange(0, tsteps) * dts
    tmax = tpoints[-1]

    # Stimulus
    n_frames = opts.get('nframes', 8)
    cue_duration = opts.get('cue_duration', 0.250)
    total_stimulus_duration = n_frames * cue_duration
    go_int = opts.get('go_int', 0.5)

    # Getting the phase of the bump
    r_aligned, phases, amps, amps_m = get_phases_and_amplitudes_auto(rdata, aligned_profiles=True)
    phases = phases[:-1]

    # Figure and Axes
    custom_plot_params(1.2, latex=True)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = False
    fig, axes, kw = set_plot_environment(ncols=1, nrows=3, figsize=figsize, dpi=200, sharex=False)
    ax1, ax2, ax3 = axes

    red = "#E24A33"
    blue = "#348ABD"

    # Actual plotting starts here
    for ym, ys, lbl, ls in zip([prm, aprm], [prs, aprs], ['Preferred', 'Non-preferred'], ['-', '--']):
        ax1, p, b = plot_error_filled(tpoints, ym, ys, ax=ax1, color='black', linestyle=ls, alpha=0.2)
        p.set_label(lbl)

    left_plot, right_plot = None, None
    for trial in random_choice:
        color = blue if sdata.binchoice[trial] >= 0 else red
        p1, = ax1.plot(tpoints, d1[:, trial], color=color, alpha=0.4, lw=0.5, ls='-')
        ax1.plot(tpoints, d2[:, trial], color=color, alpha=0.4, lw=0.5, ls='--')
        ax2.plot(tpoints, li[:, trial], color=color, alpha=0.4, lw=0.5, ls='-')
        ax2.plot(tpoints, ri[:, trial], color=color, alpha=0.4, lw=0.5, ls='--')
        ax3.plot(tpoints, np.rad2deg(phases[:, trial]), color=color, alpha=0.8, lw=0.5)
        if color == red:
            left_plot = p1
        else:
            right_plot = p1

    if left_plot is not None:
        left_plot.set_label('Left')
    if right_plot is not None:
        right_plot.set_label('Right')

    ax1.plot(tpoints, np.ones_like(tpoints) * 50, '--', color='black', lw=0.3, alpha=0.7)
    ax1.set_ylim(0, 75)
    ax1.set_xlim(1.8, 3.2)
    ax1.legend(frameon=False)
    ax1.set_ylabel("Firing Rate (Hz)")
    # plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xticks(np.linspace(total_stimulus_duration, total_stimulus_duration + go_int, 5))

    for ym, ys, ls in zip([pim, apim], [pis, apis], ['-', '--']):
        ax2, p, b = plot_error_filled(tpoints, ym, ys, ax=ax2, color='black', linestyle=ls, alpha=0.2)

    ax2.set_ylabel("Input (a.u.)")
    ax2.set_xlim(0, tmax)
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3.set_ylabel(r"Phase ($^\circ$)")
    ax3.set_ylim(-90, 90)
    ax3.set_yticks([-90, -45, 0, 45, 90])
    ax3.set_xlim(0, tmax)
    ax3.set_xlabel('Time (s)')

    rectspan(total_stimulus_duration, total_stimulus_duration + go_int, axes, color="gray", linewidth=0.5, alpha=0.2)
    fig.tight_layout()

    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        sigma, i0 = itemgetter('sigmaOU', 'i0')(kwargs)
        bump = 'nobump' if kwargs.get('nobump', False) else 'bump'
        if np.all([i0, sigma]):
            filename = directory + f"decision_ciruit_summary_{bump}_i0-{i0:.2f}_sigma-{sigma:.2f}"
        else:
            filename = directory + f"decision_circuit_summary_{bump}_{ntrials}_trials"
        save_plot(fig, filename, **kwargs)

    return fig, axes


@apply_custom_style
def plot_response_time(sdata, opts, figsize=(6, 3), save=False, **kwargs):
    """Plots a figure that summarizes the response times statistics of the simulation.

    :param pd.DataFrame sdata: dataframe with the results of the simulation.
    :param dict opts: dictionary of parameters and simulation options.
    :param Cython.Includes.numpy.ndarray figsize: Figure size (width, heigth) in inches.
    :param bool save: whether to save or not the created figure(s).
    :param kwargs: additional keyword arguments.
    :return: a list of figure(s).
    :rtype: list of plt.Figure
    """
    ntrials = opts.get('ntrials', 0)

    # Separate trials between correct and incorrect
    sdata = sdata.loc[sdata.binrt != -1]
    sdata.loc[:, 'binrt'] = sdata.loc[:, 'binrt'].apply(lambda x: (x - sdata.binrt.mean()) * 1000)
    correct_trials = (np.sign(sdata.binchoice) == np.sign(sdata.average_circ))
    scorrect = sdata.loc[correct_trials]
    sincorrect = sdata.loc[~correct_trials]

    # Mean values of trials grouped by category
    group = sdata.groupby('category')
    correct_group = scorrect.groupby('category')
    incorrect_group = sincorrect.groupby('category')
    mean = group.mean()
    mean_correct = correct_group.mean()
    mean_incorrect = incorrect_group.mean()
    std = group.std()
    std_correct = correct_group.sem()
    std_incorrect = incorrect_group.sem()

    # Figure and Axes
    custom_plot_params(1.2, latex=True)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = False
    fig, axes, kw = set_plot_environment(ncols=2, nrows=1, figsize=figsize, dpi=200, sharex=False)
    ax1, ax2 = axes

    red = "#E24A33"
    # blue = "#348ABD"
    blue = 'green'

    # Histogram of response times (total, incorrect, correct)
    bins = np.arange(np.min(sdata.binrt), np.max(sdata.binrt), 30)
    # ax1.hist(sdata.binrt, bins=bins, label='All trials', ec='black', fc='black', alpha=0.3, density=False)
    ax1.hist(scorrect.binrt, bins=bins, label='Correct trials', fc=blue, ec=blue, alpha=0.3, density=False)
    ax1.hist(sincorrect.binrt, bins=bins, label='Incorrect trials', fc=red, ec=red, alpha=0.3, density=False)

    ax1.legend(frameon=False, fontsize=6)
    ax1.set_xlabel('Response time (ms)')
    ax1.set_xlim(0, np.max(sdata.binrt) * 1.1)
    ax1.set_ylabel('Trial density')
    ax1.set_yticks([])

    # Response time as a function of the average orientation, for total, incorrect and correct ax2.scatter(
    # sdata.average, sdata.binrt, marker='.', color='black', s=1.0, alpha=0.3) ax2, pa, ba = plot_error_filled(
    # np.array(mean.index), mean['binrt'], std['binrt'], color='black', alpha=0.3, ax=ax2) pa.set_label('All trials')
    ax2.scatter(sdata[correct_trials].average, sdata[correct_trials].binrt, marker='.', color=blue, s=1.0, alpha=0.7)
    ax2, pc, bc = plot_error_filled(np.array(mean_correct.index), mean_correct.binrt, std_correct.binrt, color=blue,
                                    alpha=0.3, ax=ax2)
    pc.set_label('Correct trials')
    ax2.scatter(sdata[~correct_trials].average, sdata[~correct_trials].binrt, marker='.', color=red, s=1.0, alpha=0.7)
    ax2, pi, bi = plot_error_filled(np.array(mean_incorrect.index), mean_incorrect.binrt, std_incorrect.binrt,
                                    color=red, alpha=0.3, ax=ax2)
    pi.set_label('Incorrect trials')
    ax2.legend(frameon=False, fontsize=6)
    ax2.set_ylabel('Response time (ms)')
    ax2.set_xlim(-90, 90)
    ax2.set_xlabel(r"Avg. orientation ($^\circ$)")

    fig.tight_layout()

    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        sigma, i0 = itemgetter('sigmaOU', 'i0')(kwargs)
        bump = 'nobump' if kwargs.get('no_bump', False) else 'bump'
        if np.all([i0, sigma]):
            filename = directory + f"response_times_{bump}_i0-{i0:.2f}_sigma-{sigma:.2f}"
        else:
            filename = directory + f"response_times_{bump}_{ntrials}_trials"
        save_plot(fig, filename, **kwargs)

    return fig, axes


@apply_custom_style
def plot_rt_summ(i0s, sigma, save=False, **kwargs):
    """Plot response times for different values of the input current.

    :param list i0s: values of :math:`I_0` where :math:`I = I_{\text{crit}} + I_0`.
    :param sigma: value of :math:`\sigma_{\text{OU}}`.
    :param bool save: whether to save or not the created figure(s).
    :return: figure and axes.
    :rtype: (plt.Figure, np.ndarray of plt.Axes)
    """
    ntrials = kwargs.get('ntrials', 16000)
    bump = 'bump' if kwargs.pop('bump', False) else 'nobump'

    # Prepare figure
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = False
    fig, axs, kw = set_plot_environment(nrows=2, ncols=3, figsize=(6, 4), dpi=200, sharex='row', sharey='row',
                                        gridspec_kw=dict(top=0.92, hspace=0.5, wspace=0.3, right=0.97, left=0.1))

    # Load the data and compute log regression
    for k, i0 in enumerate(i0s):
        title = r"$(I_0 = %.2f$)" % i0
        legend = True if k == 0 else False
        ylabel = True if k == 0 else False
        axes = axs[:, k]
        ax1, ax2 = axes
        filename = f"simu_{ntrials}_{bump}_sigma-{sigma:.2f}_i0-{i0:.2f}.csv"
        sdata, lbls = load_data(filename, './results')

        # Separate trials between correct and incorrect
        sdata = sdata.loc[sdata.binrt != -1]
        sdata.loc[:, 'binrt'] = sdata.loc[:, 'binrt'].apply(lambda x: x / 2)
        sdata.loc[:, 'binrt'] = sdata.loc[:, 'binrt'].apply(lambda x: (x - sdata.binrt.mean()) * 1000)
        correct_trials = (np.sign(sdata.binchoice) == np.sign(sdata.average))
        scorrect = sdata.loc[correct_trials]
        sincorrect = sdata.loc[~correct_trials]

        # Mean values of trials grouped by category
        group = sdata.groupby('norm_category')
        correct_group = scorrect.groupby('norm_category')
        incorrect_group = sincorrect.groupby('norm_category')
        mean = group.mean()
        mean_correct = correct_group.mean()
        mean_incorrect = incorrect_group.mean()
        std = group.std()
        std_correct = correct_group.std()
        std_incorrect = incorrect_group.std()

        red = "#E24A33"
        green = 'green'

        # Histogram of response times (total, incorrect, correct)
        bins = np.arange(np.min(sdata.binrt), np.max(sdata.binrt), 30)
        if kwargs.get('plot_mean', False):
            ax1.hist(sdata.binrt, bins=bins, label='All trials', ec='black', fc='black', alpha=0.3, density=False)
        ax1.hist(scorrect.binrt, bins=bins, label='Correct trials', fc=green, ec=green, alpha=0.3, density=False)
        ax1.hist(sincorrect.binrt, bins=bins, label='Incorrect trials', fc=red, ec=red, alpha=0.3, density=False)

        if legend:
            ax1.legend(frameon=False, fontsize=6)
        ax1.set_xlabel(r'Response time - $\left<R_T\right>$ (ms)')
        # ax1.set_xlim(np.min(sdata.binrt) * 1.1, np.max(sdata.binrt) * 1.1)
        if ylabel:
            ax1.set_ylabel('Trial density')
        ax1.set_yticks([])

        # Response time as a function of the average orientation, for total, incorrect and correct
        if kwargs.get('plot_mean', False):
            if kwargs.get('scatter', False):
                ax2.scatter(sdata.average, sdata.binrt, marker='.', color='black', s=1.0, alpha=0.2)
            ax2, pa, ba = plot_error_filled(np.array(mean.index), mean['binrt'], std['binrt'], color='black', alpha=0.1,
                                            ax=ax2)
            pa.set_label('All trials')
        if kwargs.get('scatter', False):
            ax2.scatter(sdata[correct_trials].norm_average, sdata[correct_trials].binrt, marker='.', color=green, s=1.0,
                        alpha=0.1)
            ax2.scatter(sdata[~correct_trials].norm_average, sdata[~correct_trials].binrt, marker='.', color=red, s=1.0,
                        alpha=0.1)
        ax2, pc, bc = plot_error_filled(np.array(mean_correct.index), mean_correct.binrt, std_correct.binrt,
                                        color=green, alpha=0.3, ax=ax2)
        pc.set_label('Correct trials')
        ax2, pi, bi = plot_error_filled(np.array(mean_incorrect.index), mean_incorrect.binrt, std_incorrect.binrt,
                                        color=red, alpha=0.3, ax=ax2)
        pi.set_label('Incorrect trials')
        if legend:
            ax2.legend(frameon=False, fontsize=6)
        if ylabel:
            ax2.set_ylabel(r'Response time - $\left<R_T\right>$ (ms)')
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_xlabel(r"Category-level average")
        ax1.set_title(title)

        # Change ticklabels format (from latex to plain text)
        if ylabel:
            for ax in [ax1, ax2]:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if save:
        ntrials = kwargs.get('ntrials', 0)
        directory = kwargs.get('fig_dir', 'figs/')
        filename = directory + f'rt_summary_{ntrials}_i0_{bump}_sigma-{sigma:.2f}'
        save_plot(fig, filename, **kwargs)


@apply_custom_style
def plot_temp_integ(i0s, sigma, lbl_fmt="x[0-9][0-9]?_norm", save=False, **kwargs):
    """Plot of temporal integration behavior for different values of the parameters: PM function, PPk, and estimation.

    :param list i0s: values of :math:`I_0` where :math:`I = I_{\text{crit}} + I_0`.
    :param sigma: value of :math:`\sigma_{\text{OU}}`.
    :param lbl_fmt: format of the label pointing to the design matrix as regexp.
    :param bool save: whether to save or not the created figure(s).
    :return: figure and axes.
    :rtype: (plt.Figure, np.ndarray of plt.Axes)
    """
    mylog = kwargs.pop('mylog', MyLog(10))
    ntrials = kwargs.get('ntrials', 16000)
    bump = 'bump' if kwargs.pop('bump', False) else 'nobump'
    if bump == 'bump':
        titles = [r"\textbf{Recency}", r"\textbf{Recency}", r"\textbf{Uniform}"]
    else:
        titles = [r"\textbf{Recency}", r"\textbf{Uniform}", r"\textbf{Primacy}"]

    # Prepare figure
    # custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = False
    fig, axs, kw = set_plot_environment(nrows=3, ncols=3, figsize=(6, 5), dpi=200, sharex='row', sharey='row',
                                        gridspec_kw=dict(top=0.92, hspace=0.5, wspace=0.3, right=0.97, left=0.1))

    # Load the data and compute log regression
    dfs = []
    log_results = []
    ppk_max = 0
    for k, (i0, title) in enumerate(zip(i0s, titles)):
        title = title + r" $(I_0 = %.2f$)" % i0
        legend = True if k == 1 else False
        ylabel = True if k == 0 else False
        axes = axs[:, k]
        ax1, ax2, ax3 = axes
        filename = f"simu_{ntrials}_{bump}_sigma-{sigma:.2f}_i0-{i0:.2f}.csv"
        df, lbls = load_data(filename, './results')
        # Get labels from the data-frame
        labels = re.findall(f"({lbl_fmt})", ' '.join(df.columns))
        logging.info(f"Fitting data (i0: {i0}) to the logistic function ...")
        if not kwargs.get('new_ppk', True):
            df['binchoice'] = (df.estim >= 0) * 2 - 1
        results = log_reg(df, labels)

        if isinstance(results, (tuple, list)):
            mylog(0)
            do_plot_ppk = True
            m_r, mf_r, df = results
        else:
            mylog.msg(1, up_lines=2)
            do_plot_ppk = False
            m_r, mf_r, df = (None, None, df)
        dfs.append(df)
        log_results.append([m_r, mf_r])

        if do_plot_ppk:
            logging.info('Plotting psychometric function ...')
            fig, ax_pm, p = plot_pm(df, ax=ax1, fig=fig, legend=legend, ylabel=ylabel)
            mylog(0)
            logging.info('Plotting psychophysical kernel ...')
            if kwargs.get('new_ppk', True):
                fig, ax_ppk = plot_ppk(mf_r, linewidth=0.5, marker='o', fig=fig, ax=ax2, legend=legend, ylabel=ylabel)
                if ppk_max < np.max(mf_r.beta) * 1.1:
                    ppk_max = np.max(mf_r.beta) * 1.1
                ax_ppk.set_ylim(0, ppk_max)
                mylog(0)
            if kwargs.get('old_ppk', False):
                logging.info('Plotting the psychophysical kernel without the readout circuit ...')
                logging.info('Fitting estimation-based choice data to the logistic function ...')
                df['binchoice_old'] = (df.estim >= 0) * 2 - 1
                # Check if this 'readout' is different from that of the decision network
                errors = np.sum(not np.alltrue(df['binchoice_old'] == df['binchoice']))
                if errors > 0:
                    logging.warning(f"Categorical readout is not optimal: {errors / float(len(df))}\% of errors.")
                results2 = log_reg(df, labels, ylabel='binchoice_old', compute_pm=False)
                if isinstance(results2, (tuple, list)):
                    mylog(0)
                    m_r, mf_r, df2 = results2
                    fig, ax_ppk = plot_ppk(mf_r, linewidth=0.5, marker='s', fig=fig, ax=ax2, legend=False,
                                           set_labels=False, zorder=1, ylabel=ylabel)
                    last_plot = ax_ppk.lines[-1]
                    last_plot.set_label(r"$\beta_i$ (Phase readout)")
                    if legend:
                        ax_ppk.legend(frameon=False, fontsize=5, loc='upper left')
                    mylog.msg(0, up_lines=2)
                else:
                    mylog.msg(1, up_lines=2)

        logging.info('Plotting orientation vs. estimation figures.')
        fig, ax_estim_avg_r = plot_estimation_avg(df, fig=fig, ax=ax3, set_ylabel=ylabel, legend=legend)
        ax1.set_title(title)
        mylog(0)

        # Change ticklabels format (from latex to plain text)
        if ylabel:
            for ax in [ax1, ax2, ax3]:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    if save:
        ntrials = kwargs.get('ntrials', 0)
        directory = kwargs.get('fig_dir', 'figs/')
        filename = directory + f'stats_summary_{ntrials}_i0_{bump}_sigma-{sigma:.2f}'
        save_plot(fig, filename, **kwargs)

    return fig, axs


@apply_custom_style
def plot_bump_vs_nobump(i0s, sigma, lbl_fmt="x[0-9][0-9]?_norm", save=False, **kwargs):
    """Plot that compares the integration behavior of the network when for different initial conditions.

    :param list i0s: values of :math:`I_0` where :math:`I = I_{\text{crit}} + I_0`.
    :param sigma: value of :math:`\sigma_{\text{OU}}`.
    :param lbl_fmt: format of the label pointing to the design matrix as regexp.
    :param bool save: whether to save or not the created figure(s).
    :return: figure and axes.
    :rtype: (plt.Figure, np.ndarray of plt.Axes)
    """
    mylog = kwargs.pop('mylog', MyLog(10))
    ntrials = kwargs.get('ntrials', 16000)
    titles = [r"\noindent\textbf{No-bump condition}\\\textbf{Uniform}",
              r"\noindent\textbf{Bump condition}\\\textbf{Uniform}"]

    # Prepare figure
    # custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = False
    fig, axs, kw = set_plot_environment(nrows=3, ncols=2, figsize=(4, 5), dpi=200, sharex='row', sharey='row',
                                        gridspec_kw=dict(top=0.92, hspace=0.4, wspace=0.3, right=0.97, left=0.12))

    # Load the data and compute log regression
    dfs = []
    log_results = []
    ppk_max = 0
    for k, (i0, title) in enumerate(zip(i0s, titles)):
        title = title + r" $(I_0 = %.2f$)" % i0
        legend = True if k == 0 else False
        ylabel = True if k == 0 else False
        axes = axs[:, k]
        ax1, ax2, ax3 = axes
        bump = 'nobump' if k == 0 else 'bump'
        filename = f"simu_{ntrials}_{bump}_sigma-{sigma:.2f}_i0-{i0:.2f}.csv"
        df, lbls = load_data(filename, './results')
        # Get labels from the data-frame
        labels = re.findall(f"({lbl_fmt})", ' '.join(df.columns))
        logging.info(f"Fitting data (i0: {i0}) to the logistic function ...")
        results = log_reg(df, labels)

        if isinstance(results, (tuple, list)):
            mylog(0)
            do_plot_ppk = True
            m_r, mf_r, df = results
        else:
            mylog.msg(1, up_lines=2)
            do_plot_ppk = False
            m_r, mf_r, df = (None, None, df)
        dfs.append(df)
        log_results.append([m_r, mf_r])

        if do_plot_ppk:
            logging.info('Plotting psychometric function ...')
            fig, ax_pm, p = plot_pm(df, ax=ax1, fig=fig, legend=legend, ylabel=ylabel)
            mylog(0)
            logging.info('Plotting psychophysical kernel ...')
            fig, ax_ppk = plot_ppk(mf_r, linewidth=0.5, marker='o', fig=fig, ax=ax2, legend=legend, ylabel=ylabel)
            if ppk_max < np.max(mf_r.beta) * 1.1:
                ppk_max = np.max(mf_r.beta) * 1.1
            ax_ppk.set_ylim(0, ppk_max)
            mylog(0)
            if kwargs.get('old_ppk', True):
                logging.info('Plotting the psychophysical kernel without the readout circuit ...')
                logging.info('Fitting estimation-based choice data to the logistic function ...')
                df['binchoice_old'] = (df.estim >= 0) * 2 - 1
                # Check if this 'readout' is different from that of the decision network
                errors = np.sum(not np.alltrue(df['binchoice_old'] == df['binchoice']))
                if errors > 0:
                    logging.warning(f"Categorical readout is not optimal: {errors / float(len(df))}\% of errors.")
                results2 = log_reg(df, labels, ylabel='binchoice_old', compute_pm=False)
                if isinstance(results2, (tuple, list)):
                    mylog(0)
                    m_r, mf_r, df2 = results2
                    fig, ax_ppk = plot_ppk(mf_r, linewidth=0.5, marker='s', fig=fig, ax=ax2, legend=False,
                                           set_labels=False, zorder=1, ylabel=ylabel)
                    last_plot = ax_ppk.lines[-1]
                    last_plot.set_label(r"$\beta_i$ (Phase readout)")
                    if legend:
                        ax_ppk.legend(frameon=False, fontsize=5, loc='lower left')
                    mylog.msg(0, up_lines=2)
                else:
                    mylog.msg(1, up_lines=2)

        logging.info('Plotting orientation vs. estimation figures.')
        fig, ax_estim_avg_r = plot_estimation_avg(df, fig=fig, ax=ax3, set_ylabel=ylabel, legend=legend)
        ax1.set_title(title)
        mylog(0)

        # Change ticklabels format (from latex to plain text)
        if ylabel:
            for ax in [ax1, ax2, ax3]:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        filename = directory + f'bump_vs_nobump_{ntrials}_uniform_sigma-{sigma:.2f}'
        save_plot(fig, filename, **kwargs)

    return fig, axs


@apply_custom_style
def plot_trial_ic(r_datas, dataframes, ddatas, figsize=np.array([5.0, 5.1]) * 1.2, save=False, **kwargs):
    """Single trial with and without the bump as initial condition. It contains plots of the stimulus, the
    firing rate evolution for both conditions and the evolution of the decision network.

    :param list r_datas: firing rate datas.
    :param list of pd.DataFrame dataframes: data frames that includes the simulation design and the results.
    :param list ddatas: firing rate of the decision network.
    :param Cython.Includes.numpy.ndarray figsize: Figure size (width, heigth) in inches.
    :param bool save: whether to save or not the created figure(s).
    :param kwargs: additional keyword arguments.
    :return: a list of figure(s).
    :rtype: list of plt.Figure
    """
    # Simulation parameters
    r_data1, r_data2 = r_datas
    sdata1, sdata2 = dataframes
    (tsteps, ntrials, n) = r_data1.shape

    theta = np.arange(n) / n * 180 * 2 - 180

    dts = kwargs.get('save_interval', 0.01)
    tpoints = np.arange(0, tsteps) * dts
    tmax = tpoints[-1]

    cue = kwargs.get('cue_duration', 0.250)

    # Data post-processing:
    r_aligned1, phases1, amps1, amps_m1 = get_phases_and_amplitudes_auto(r_data1, aligned_profiles=True)
    r_aligned2, phases2, amps2, amps_m2 = get_phases_and_amplitudes_auto(r_data2, aligned_profiles=True)

    # #### Example trial plot (colormap, amplitude, phase, stimuli, profiles)
    trial = kwargs.pop('selected_trial', None)
    if trial is None:
        bound_category = kwargs.pop('max_category', 45)  # Find a trial whose category is less than bound_category
        category = 100  # Initialization
        trials = list(range(ntrials))
        trial = 0
        data_trial1 = sdata1.iloc[trial]
        while np.abs(category) > bound_category and trials:
            trial = trials.pop(np.random.randint(0, len(trials)))  # Select a trial randomly
            data_trial1 = sdata1.iloc[trial]
            category = data_trial1['category']
        logging.debug('Semi-Randomly selected trial: %d' % trial)
    else:
        data_trial1 = sdata1.loc[trial]
        category = data_trial1['category']

    # For debugging:
    logging.debug(data_trial1)

    trial = sdata1.reset_index().loc[sdata1.index == trial].index[0]
    r1 = r_data1[:, trial, :]
    r2 = r_data2[:, trial, :]
    phase1 = np.rad2deg(phases1[:, trial])
    phase2 = np.rad2deg(phases2[:, trial])
    amp1, ampm1 = (amps1[:, trial], amps_m1[:, trial])
    amp2, ampm2 = (amps2[:, trial], amps_m2[:, trial])

    dleft1 = ddatas[0][:, 0, trial]
    dleft2 = ddatas[1][:, 0, trial]
    dright1 = ddatas[0][:, 1, trial]
    dright2 = ddatas[1][:, 1, trial]

    # Setting the figure
    custom_plot_params(1.2, latex=True)
    plt.rcParams['figure.dpi'] = 200

    # Grid(s)
    gs1 = gridspec.GridSpec(5, 1)
    gs_bar = gridspec.GridSpec(5, 1)

    # Plot margins and grid margins
    left = 0.1
    right_1 = 0.9
    hspace = 0.15
    wspace = 0.2

    # G1
    (top1, bottom1) = (0.95, 0.1)
    gs1.update(left=left, right=right_1, top=top1, bottom=bottom1, hspace=hspace, wspace=wspace)
    # Gbar
    (top_bar, bottom_bar, left_bar, right_bar) = (top1, bottom1, right_1 + 0.01, right_1 + 0.03)
    gs_bar.update(left=left_bar, right=right_bar, top=top_bar, bottom=bottom_bar, hspace=hspace, wspace=wspace)

    # Figure and Axes
    fig = plt.figure(figsize=figsize)
    fig.set_tight_layout(False)

    ax_ph = fig.add_subplot(gs1[0, 0])
    ax_c1 = fig.add_subplot(gs1[1, 0], sharex=ax_ph)
    ax_c2 = fig.add_subplot(gs1[2, 0], sharex=ax_c1, sharey=ax_c1)
    ax_amp = fig.add_subplot(gs1[3, 0], sharex=ax_ph)
    ax_dec = fig.add_subplot(gs1[4, 0])

    axes = [ax_ph, ax_c1, ax_c2, ax_amp, ax_dec]

    mod_axes([ax_amp, ax_ph, ax_dec])
    ax_ph.spines['bottom'].set_position('center')

    ax_bar = create_colorbar_ax(fig, gs_bar[1:3, 0])  # Ax for the color-bar of the color-plot

    # Actual plotting starts here
    # Amplitude of the bump
    camp1, = ax_amp.plot(tpoints[:-1], amp1[:-1], label=r'No-bump')
    camp2, = ax_amp.plot(tpoints[:-1], amp2[:-1], label=r'Bump')
    color1 = camp1.get_color()
    color2 = camp2.get_color()
    ax_amp.legend(frameon=False, fontsize=8)
    ax_amp.set_xlim(0, tmax)
    ax_amp.set_ylim(0, np.max(np.array([r1, r2])) * 1.01)
    ax_amp.set_ylabel('Firing Rate (Hz)', labelpad=10)
    plt.setp(ax_amp.get_xticklabels(), visible=False)

    # Decision network
    ax_dec.plot(tpoints[:-1], dleft1[:-1], color=color1)
    ax_dec.plot(tpoints[:-1], dright1[:-1], '--', color=color1)
    ax_dec.plot(tpoints[:-1], dleft2[:-1], color=color2)
    ax_dec.plot(tpoints[:-1], dright2[:-1], '--', color=color2)
    ax_dec.plot(tpoints[:-1], np.ones_like(dright2[:-1]) * 50, '--', color='k', lw=0.5)
    ax_dec.set_xlim(1.6, 3.2)
    ax_dec.set_ylabel(r"$r_L$, $r_R$ (Hz)", labelpad=10)
    custom_lines = [Line2D([0], [0], color='k', lw=1, linestyle='-'), Line2D([0], [0], color='k', lw=1, linestyle='--')]

    ax_dec.legend(custom_lines, [r"$r_L$", r"$r_R$"], frameon=False)
    ax_dec.set_xlabel('Time (s)')
    ax_dec.text(3.0, 0.05, r'Decision window', ha='right', va='bottom', transform=ax_dec.get_xaxis_transform(),
                fontsize=8)

    # Phase evolution and stimuli
    nframes = kwargs.get('nframes', 8)
    labels = ['x%d' % k for k in range(1, nframes + 1)]
    stim_phases = data_trial1[labels].to_numpy(int)
    stim_times = np.arange(0, nframes) * cue
    logging.debug('Orientations of the frames are: %s' % stim_phases)

    stim_color = mcolors.to_rgba('cornflowerblue', 0.3)
    ax_ph.bar(stim_times, stim_phases, align='edge', width=cue * 1.0, lw=1.0, ec='cornflowerblue',
              fc=stim_color, label=r'$\theta_i^{\text{stim}}$')
    ax_ph.plot(tpoints[:-1], phase1[:-1], color=color1, linewidth=1.5, label=r'No-bump')
    ax_ph.plot(tpoints[:-1], phase2[:-1], color=color2, linewidth=1.5, label=r'Bump')
    ax_ph.legend(frameon=False, fontsize=8, ncol=2)
    ax_ph.set_ylim(-100, 100)
    ax_ph.set_yticks([-90, -45, 0, 45, 90])
    ax_ph.set_ylabel(r'$\theta\ (^\circ)$')
    plt.setp(ax_ph.get_xticklabels(), visible=False)

    # Colorplot of the bump
    vmax = np.max(np.array([r1, r2]))
    cmap = plt.get_cmap('hot')
    bump1 = ax_c1.pcolormesh(tpoints[0::1], theta, r1[0::1].T, vmin=0, vmax=vmax, cmap=cmap)
    bump1.set_edgecolor('face')
    # ax_c1.set_xlabel('Time (s)')
    ax_c1.set_ylabel(r'$\theta\ (^\circ)$', labelpad=-1)
    ax_c1.set_ylim(-180, 180)
    ax_c1.set_yticks([-180, -90, 0, 90, 180])
    bump2 = ax_c2.pcolormesh(tpoints[0::1], theta, r2[0::1].T, vmin=0, vmax=vmax, cmap=cmap)
    bump2.set_edgecolor('face')
    ax_c2.set_ylabel(r'$\theta\ (^\circ)$', labelpad=-1)
    ax_c2.set_ylim(-180, 180)
    ax_c2.set_yticks([-180, -90, 0, 90, 180])
    plt.setp(ax_c1.get_xticklabels(), visible=False)
    plt.setp(ax_c2.get_xticklabels(), visible=False)

    # Colorbar
    cbar = plt.colorbar(bump1, cax=ax_bar, fraction=1.0, orientation='vertical', aspect=15)
    cbar.set_label('Firing Rate (Hz)', fontsize=8)
    cbar.solids.set_edgecolor("face")

    # Change ticklabels format (from latex to plain text)
    for ax in [ax_amp, ax_ph, ax_c1, ax_c2, ax_dec]:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax_dec.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    rectspan(2.0, 2.0 + 1.0, [ax_ph, ax_amp, ax_dec], color="gray", linewidth=0.5, alpha=0.2, alpha_lines=0.5)

    realtrial = sdata1.iloc[trial]['index']
    fig.text(right_1, 1, r'Trial %d. Category: $%d^\circ$' % (realtrial + 1, category), ha='right', va='top')
    fig.set_tight_layout(False)

    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        filename = directory + f"comparison_bump_vs_no_bump_{trial}"
        save_plot(fig, filename, **kwargs)

    return fig, axes


@apply_custom_style
def plot_trial_ic_two(r_datas, dataframes, figsize=np.array([5.0, 5.1]) * 1.2, save=False, **kwargs):
    """Plots a figure that summarizes the results of a single randomly chosen (see constrains) trial.

    :param list r_datas: firing rate datas.
    :param list of pd.DataFrame dataframes: data frames that includes the simulation design and the results.
    :param Cython.Includes.numpy.ndarray figsize: Figure size (width, heigth) in inches.
    :param bool save: whether to save or not the created figure(s).
    :param kwargs: additional keyword arguments.
    :return: a list of figure(s).
    :rtype: list of plt.Figure
    """
    # Simulation parameters
    r_data1, r_data2 = r_datas
    sdata1, sdata2 = dataframes
    (tsteps, ntrials, n) = r_data1.shape

    theta = np.arange(n) / n * 180 * 2 - 180

    dts = kwargs.get('save_interval', 0.01)
    tpoints = np.arange(0, tsteps) * dts
    cue = kwargs.get('cue_duration', 0.250)

    # Data post-processing:
    r_aligned1, phases1, amps1, amps_m1 = get_phases_and_amplitudes_auto(r_data1, aligned_profiles=True)
    r_aligned2, phases2, amps2, amps_m2 = get_phases_and_amplitudes_auto(r_data2, aligned_profiles=True)

    # #### Example trial plot (colormap, amplitude, phase, stimuli, profiles)
    trial = kwargs.pop('selected_trial', None)
    if trial is None:
        bound_category = kwargs.pop('max_category', 45)  # Find a trial whose category is less than bound_category
        category = 100  # Initialization
        trials = list(range(ntrials))
        trial = 0
        data_trial1 = sdata1.iloc[trial]
        while np.abs(category) > bound_category and trials:
            trial = trials.pop(np.random.randint(0, len(trials)))  # Select a trial randomly
            data_trial1 = sdata1.iloc[trial]
            category = data_trial1['category']
        logging.debug('Semi-Randomly selected trial: %d' % trial)
    else:
        data_trial1 = sdata1.loc[trial]

    # For debugging:
    logging.debug(data_trial1)

    trial = sdata1.reset_index().loc[sdata1.index == trial].index[0]
    r1 = r_data1[:, trial, :]
    r2 = r_data2[:, trial, :]
    phase1 = np.rad2deg(phases1[:, trial])
    phase2 = np.rad2deg(phases2[:, trial])
    amp1, ampm1 = (amps1[:, trial], amps_m1[:, trial])
    amp2, ampm2 = (amps2[:, trial], amps_m2[:, trial])

    # Setting the figure
    custom_plot_params(1.2, latex=True)
    plt.rcParams['figure.dpi'] = 200

    # Grid(s)
    gs1 = gridspec.GridSpec(4, 1)
    gs_bar = gridspec.GridSpec(4, 1)

    # Plot margins and grid margins
    left = 0.1
    right_1 = 0.9
    hspace = 0.15
    wspace = 0.2

    # G1
    (top1, bottom1) = (0.95, 0.1)
    gs1.update(left=left, right=right_1, top=top1, bottom=bottom1, hspace=hspace, wspace=wspace)
    # Gbar
    (top_bar, bottom_bar, left_bar, right_bar) = (top1, bottom1, right_1 + 0.01, right_1 + 0.03)
    gs_bar.update(left=left_bar, right=right_bar, top=top_bar, bottom=bottom_bar, hspace=hspace, wspace=wspace)

    # Figure and Axes
    fig = plt.figure(figsize=figsize)
    fig.set_tight_layout(False)

    ax_ph = fig.add_subplot(gs1[0, 0])
    ax_c1 = fig.add_subplot(gs1[1, 0], sharex=ax_ph)
    ax_c2 = fig.add_subplot(gs1[2, 0], sharex=ax_c1, sharey=ax_c1)
    ax_amp = fig.add_subplot(gs1[3, 0], sharex=ax_ph)

    axes = [ax_ph, ax_c1, ax_c2, ax_amp]

    mod_axes([ax_amp, ax_ph])
    ax_ph.spines['bottom'].set_position('center')

    ax_bar = create_colorbar_ax(fig, gs_bar[1:3, 0])  # Ax for the color-bar of the color-plot

    # Actual plotting starts here
    # Amplitude of the bump
    ax_amp.plot(tpoints[:-1], amp1[:-1], label=r'No-bump')
    ax_amp.plot(tpoints[:-1], amp2[:-1], label=r'Bump')
    ax_amp.legend(frameon=False, fontsize=8)
    ax_amp.set_xlim(0, 3.0)
    ax_amp.set_ylim(0, np.max(np.array([r1, r2])) * 1.01)
    ax_amp.set_ylabel('Firing Rate (Hz)', labelpad=10)

    # # Decision network
    ax_amp.set_xlabel('Time (s)')
    ax_amp.text(2.0, 0.05, r'Decision window', ha='left', va='bottom', transform=ax_amp.get_xaxis_transform(),
                fontsize=8)

    # Phase evolution and stimuli
    nframes = kwargs.get('nframes', 8)
    labels = ['x%d' % k for k in range(1, nframes + 1)]
    stim_phases = data_trial1[labels].to_numpy(int)
    stim_times = np.arange(0, nframes) * cue
    logging.debug('Orientations of the frames are: %s' % stim_phases)

    stim_color = mcolors.to_rgba('cornflowerblue', 0.3)
    ax_ph.bar(stim_times, stim_phases, align='edge', width=cue * 1.0, lw=1.0, ec='cornflowerblue',
              fc=stim_color, label=r'$\theta_i^{\text{stim}}$')

    ax_ph.legend(frameon=False, fontsize=8, ncol=2)
    ax_ph.set_ylim(-100, 100)
    ax_ph.set_yticks([-90, -45, 0, 45, 90])
    ax_ph.set_ylabel(r'$\theta\ (^\circ)$')
    plt.setp(ax_ph.get_xticklabels(), visible=False)

    # Colorplot of the bump
    vmax = np.max(np.array([r1, r2]))
    cmap = plt.get_cmap('hot')
    bump1 = ax_c1.pcolormesh(tpoints[0::1], theta, r1[0::1].T, vmin=0, vmax=vmax, cmap=cmap)
    bump1.set_edgecolor('face')
    # ax_c1.set_xlabel('Time (s)')
    ax_c1.set_ylabel(r'$\theta\ (^\circ)$', labelpad=-1)
    ax_c1.set_ylim(-180, 180)
    ax_c1.set_yticks([-180, -90, 0, 90, 180])
    bump2 = ax_c2.pcolormesh(tpoints[0::1], theta, r2[0::1].T, vmin=0, vmax=vmax, cmap=cmap)
    bump2.set_edgecolor('face')

    ax_c1.plot(tpoints[:-1], phase1[:-1], color=cmap(1.0), linewidth=1.5, label=r'No-bump')
    ax_c2.plot(tpoints[:-1], phase2[:-1], color=cmap(1.0), linewidth=1.5, label=r'Bump')

    ax_c2.set_ylabel(r'$\theta\ (^\circ)$', labelpad=-1)
    ax_c2.set_ylim(-180, 180)
    ax_c2.set_yticks([-180, -90, 0, 90, 180])
    plt.setp(ax_c1.get_xticklabels(), visible=False)
    plt.setp(ax_c2.get_xticklabels(), visible=False)

    # Colorbar
    cbar = plt.colorbar(bump1, cax=ax_bar, fraction=1.0, orientation='vertical', aspect=15)
    cbar.set_label('Firing Rate (Hz)', fontsize=8)
    cbar.solids.set_edgecolor("face")

    # Change ticklabels format (from latex to plain text)
    for ax in [ax_amp, ax_ph, ax_c1, ax_c2]:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax_amp.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    rectspan(2.0, 2.0 + 1.0, [ax_ph, ax_amp], color="gray", linewidth=0.5, alpha=0.2, alpha_lines=0.5)

    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        filename = directory + f"comparison_bump_vs_no_bump_v2_{trial}"
        save_plot(fig, filename, **kwargs)

    return fig, axes


@apply_custom_style
def plot_multiple_phase(r_datas, dataframes, save=False, **kwargs):
    """Plot of the phase evolution of multiple trials and their average for different initial conditions.

    :param list r_datas: firing rate datas.
    :param list of pd.DataFrame dataframes: data frames that includes the simulation design and the results.
    :param bool save: whether to save or not the created figure(s).
    :param kwargs: additional keyword arguments.
    :return: a list of figure(s).
    :rtype: list of plt.Figure
    """
    # Simulation parameters
    r_data1, r_data2 = r_datas
    sdata1, sdata2 = dataframes
    (tsteps, ntrials, n) = r_data1.shape

    dts = kwargs.get('save_interval', 0.01)
    tpoints = np.arange(0, tsteps) * dts
    tmax = tpoints[-1]

    # Data post-processing:
    r_aligned1, phases1, amps1, amps_m1 = get_phases_and_amplitudes_auto(r_data1, aligned_profiles=True)
    r_aligned2, phases2, amps2, amps_m2 = get_phases_and_amplitudes_auto(r_data2, aligned_profiles=True)

    # Select trials
    mask = ((sdata1.category == 5) | (sdata1.category == -5)).to_numpy()
    p1 = np.rad2deg(phases1[:, mask])
    p2 = np.rad2deg(phases2[:, mask])

    p1m = p1.mean(axis=1)
    p2m = p2.mean(axis=1)

    # Setting the figure
    custom_plot_params(1.2, latex=True)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = False

    fig, axs, kw = set_plot_environment(nrows=2, ncols=1, figsize=(4, 4), dpi=200, sharex=True, sharey=True,
                                        gridspec_kw=dict(top=0.92, hspace=0.4, wspace=0.3, right=0.97, left=0.12))

    ax1, ax2 = axs
    red = "#E24A33"
    blue = "#348ABD"

    for ph1, ph2 in zip(p1.T, p2.T):
        ax1.plot(tpoints[:-1], ph1[:-1], color=red, lw=0.1, alpha=0.5)
        ax2.plot(tpoints[:-1], ph2[:-1], color=blue, lw=0.1, alpha=0.5)

    ax1.plot(tpoints[:-1], p1m[:-1], color=red, lw=2.0, label='No-bump')
    ax2.plot(tpoints[:-1], p2m[:-1], color=blue, lw=2.0, label='Bump')
    ax1.plot(tpoints[:-1], np.ones_like(tpoints[:-1]) * 0.0, '--', color='k', lw=0.5)
    ax2.plot(tpoints[:-1], np.ones_like(tpoints[:-1]) * 0.0, '--', color='k', lw=0.5)
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)

    ax2.set_xlim(0, tmax)
    ax2.set_ylim(-90, 90)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r"$\theta\ (^\circ)$")
    ax1.set_ylabel(r"$\theta\ (^\circ)$")
    ax1.set_yticks([-90, -45, 0, 45, 90])
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    rectspan(2.0, 2.0 + 1.0, axs, color="gray", linewidth=0.5, alpha=0.2, alpha_lines=0.5)

    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        filename = directory + f"comparison_bump_vs_no_bump_phases"
        save_plot(fig, filename, **kwargs)

    return fig, axs


@apply_custom_style
def plot_trial_full(r_data, s_data, d_data, stimulus, dataframe, figsize=np.array([9.0, 6.0]) * 1.2, save=False,
                    **kwargs):
    """Plots a figure that summarizes the results of a single randomly chosen (see constrains) trial. Includes the
    sensory circuit.
    :param Cython.Includes.numpy.ndarray r_data: firing rate data of the integration circuit.
    :param Cython.Includes.numpy.ndarray s_data: firing rate data of the sensory circuit.
    :param dict d_data: firing rate data and inputs of the decision circuit.
    :param Cython.Includes.numpy.ndarray stimulus: stimuli to the sensory circuit and fake stimuli.
    :param pd.DataFrame dataframe: data frame that includes the simulation design and the results.
    :param Cython.Includes.numpy.ndarray figsize: Figure size (width, heigth) in inches.
    :param bool save: whether to save or not the created figure(s).
    :param kwargs: additional keyword arguments.
    :return: a list of figure(s).
    :rtype: list of plt.Figure
    """

    # Simulation parameters
    (tsteps, ntrials, n) = r_data.shape

    logging.debug('The shape of the firing rate matrix is: (%d, %d, %d).' % (tsteps, ntrials, n))
    theta = np.arange(n) / n * 180 * 2 - 180

    dts = kwargs.get('save_interval', 0.01)
    tpoints = np.arange(0, tsteps) * dts
    tmax = tpoints[-1]

    cue = kwargs.get('cue_duration', 0.750)
    dcue = kwargs.get('dcue_duration', 2.0)

    # Data post-processing:
    ra, ph, amps, amps_m = get_phases_and_amplitudes_auto(r_data, aligned_profiles=True)

    # #### Example trial plot (colormap, phase, stimuli, ...)
    if 'chosen' in dataframe.columns:
        dataframe = dataframe.loc[(dataframe.chosen != -1)]
    trial = kwargs.pop('selected_trial', None)
    consistent = kwargs.pop('consistent', True)  # Find a consistent trial
    do_choice = kwargs.pop('do_choice', 1)  # Find a choice trial
    stim_pair = kwargs.pop('stim_pair', [0, 20])  # Find a trial with this stimuli pair
    stim_1, stim_2 = stim_pair  # Initialization
    if trial is None:
        if consistent:
            valid_sample = dataframe.loc[
                (dataframe.x1 == stim_1) & (dataframe.x2 == stim_2) & (dataframe.do_choice == do_choice) & (
                        np.sign(dataframe.binchoice) == np.sign(dataframe.x2))]
        else:
            valid_sample = dataframe.loc[
                (dataframe.x1 == stim_1) & (dataframe.x2 == stim_2) & (dataframe.do_choice == do_choice) & (
                        np.sign(dataframe.binchoice) != np.sign(dataframe.x2))]
        data_trial = valid_sample.iloc[np.random.randint(len(valid_sample))]
        trial = data_trial.chosen
        logging.debug('Semi-Randomly selected trial: %d' % trial)
    else:
        data_trial = dataframe.loc[dataframe.chosen == trial].iloc[0]

    # Find inconsistent and no-choice trials with same stimuli
    trial_1 = kwargs.pop('selected_inconsistent_trial', None)
    trial_2 = kwargs.pop('selected_nochoice_trial', None)

    if trial_1 is None:
        data_trial_inconsistent = dataframe.loc[
            (dataframe.x1 == stim_1) & (dataframe.x2 == stim_2) & (dataframe.do_choice == do_choice) & (
                    np.sign(dataframe.binchoice) != np.sign(dataframe.x2))].iloc[0]
        trial_1 = data_trial_inconsistent.chosen
    if trial_2 is None:
        data_trial_nochoice = dataframe.loc[
            (dataframe.x1 == stim_1) & (dataframe.x2 == stim_2) & (dataframe.do_choice != do_choice)].iloc[0]
        trial_2 = data_trial_nochoice.chosen

    # For debugging:
    logging.debug(data_trial)

    # #################################################################
    # Time series (integration circuit and sensory)
    rt = r_data[:, trial]
    pht = np.rad2deg(ph[:, trial])
    s_et = s_data[:, trial, 0:n]

    # Decision circuit
    d1 = d_data['d12'][:-1, 0, trial]
    d2 = d_data['d12'][:-1, 1, trial]

    # Stimulus
    stim = stimulus[:, 0, trial]
    sensory_stim = stimulus[:, 1, trial]

    # Global modulation
    dec_fb = stimulus[:, 5, trial]

    # ####################################################################
    # Snapshots for consistent, inconsistent and no-choice
    idx_times = (np.abs(np.tile(tpoints, (2, 1)).T - np.array([0.5, 3.25]))).argmin(axis=0)
    idx_trials = [trial, trial_1, trial_2]

    # Selective modulation
    att_mod_e = stimulus[:, 3, trial]
    att_mod_i = stimulus[:, 4, trial]

    # ####################################################################
    # Setting the figure
    custom_plot_params(1.2, latex=True)
    plt.rcParams['figure.dpi'] = 200

    # Grid(s)
    gs1 = gridspec.GridSpec(9, 1)
    gs2 = gridspec.GridSpec(9, 3)
    gs_bar = gridspec.GridSpec(9, 1)

    # Plot margins and grid margins
    left = 0.08
    right_1 = 0.4
    hspace = 0.4
    wspace = 0.2

    # G1
    (top1, bottom1) = (0.95, 0.08)
    gs1.update(left=left, right=right_1, top=top1, bottom=bottom1, hspace=hspace, wspace=wspace)
    # G2
    gs2.update(left=0.52, right=0.96, top=top1, bottom=bottom1, hspace=hspace, wspace=0.2)
    # Gbar
    (top_bar, bottom_bar, left_bar, right_bar) = (top1, bottom1, right_1 + 0.01, right_1 + 0.02)
    gs_bar.update(left=left_bar, right=right_bar, top=top_bar, bottom=bottom_bar, hspace=hspace, wspace=wspace)

    # Figure and Axes
    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(False)

    # First column axes
    ax_dc = fig.add_subplot(gs1[0:2, 0])
    ax_gm = fig.add_subplot(gs1[2, 0], sharex=ax_dc)
    ax_in = fig.add_subplot(gs1[3:5, 0], sharex=ax_dc)
    ax_ss = fig.add_subplot(gs1[5, 0], sharex=ax_dc)
    ax_ssn = fig.add_subplot(gs1[6:8, 0], sharex=ax_dc, sharey=ax_in)
    ax_st = fig.add_subplot(gs1[8, 0], sharex=ax_dc)

    axs1 = [ax_dc, ax_gm, ax_in, ax_ss, ax_ssn, ax_st]
    color_ax = [ax_in, ax_ss, ax_ssn]
    mod_axes(axs1)

    # Bar axes
    bar_in = create_colorbar_ax(fig, gs_bar[3:5, 0])
    bar_ss = create_colorbar_ax(fig, gs_bar[5, 0])
    bar_ssn = create_colorbar_ax(fig, gs_bar[6:8, 0])

    axs_bar = [bar_in, bar_ss, bar_ssn]

    # Second column axes
    titles = ['Stimulus 1', 'Decision', 'Stimulus 2']
    pos = [(3, 5), (6, 8), (8, 9)]
    axs_in = []
    axs_ssn = []
    axs_st = []

    axs2 = [axs_in, axs_ssn, axs_st]

    for m, (axs, position, ax_share) in enumerate(zip(axs2, pos, [ax_in, ax_ssn, ax_st])):
        (row1, row2) = position
        for k in range(3):
            ax_one = fig.add_subplot(gs2[row1:row2, k])
            ax_one.spines['top'].set_visible(False)
            ax_two = ax_one.twinx()
            ax_two.spines['top'].set_visible(False)
            if k == 0:
                ax_one.spines['right'].set_visible(False)
                ax_one.yaxis.set_ticks_position('left')
                ax_two.spines['right'].set_visible(False)
                ax_two.spines['left'].set_visible(False)
                ax_two.axes.yaxis.set_ticks([])
            if k == 1:
                ax_two.spines['right'].set_visible(False)
                ax_two.spines['left'].set_visible(False)
                ax_one.spines['right'].set_visible(False)
                ax_one.spines['left'].set_visible(False)
                ax_one.axes.yaxis.set_ticks([])
                ax_two.axes.yaxis.set_ticks([])
            if k == 2:
                ax_two.spines['left'].set_visible(False)
                ax_two.yaxis.set_ticks_position('right')
                ax_one.spines['right'].set_visible(False)
                ax_one.spines['left'].set_visible(False)
                ax_one.axes.yaxis.set_ticks([])
                ax_two.grid(False)
            axs.append([ax_one, ax_two])

    # ####################
    # Plotting starts here
    red = "#E24A33"
    blue = "#348ABD"
    pink = "#cf76b2"
    green = "#6c906f"
    ax_dc.plot(tpoints[:-1], d1, label=r'$r_L$', color=green)
    ax_dc.plot(tpoints[:-1], d2, label=r'$r_R$', color=pink)
    ax_dc.set_xlim(0, tmax)
    ax_dc.set_ylim(0, np.max([np.max(d1), np.max(d2)]) * 1.2)
    ax_dc.plot(tpoints[:-1], np.ones_like(tpoints[:-1]) * 50, '--', color='black', lw=0.3, alpha=0.7)
    plt.setp(ax_dc.get_xticklabels(), visible=False)
    ax_dc.set_ylabel('Firing Rate (Hz)', fontsize=8)
    ax_dc.legend(frameon=False, fontsize=8)
    ax_dc.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # Global modulation
    ax_gm.plot(tpoints[:-1], np.max(dec_fb[:-1], axis=1), label=r'g', color='gray')
    ax_gm.set_ylim(0, np.max(dec_fb) * 1.05)
    plt.setp(ax_gm.get_xticklabels(), visible=False)
    ax_gm.set_ylabel('Glob. mod.', fontsize=8)
    # ax_gm.legend(frameon=False, fontsize=8)
    ax_gm.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    saveformat = kwargs.get('saveformat', 'png')

    # Colorplots and colorbars
    data1 = [rt, sensory_stim, s_et]
    cmaps = ['hot', 'viridis', 'hot']
    labels = ['Firing Rate (Hz)', 'Input (a.u.)', 'Firing Rate (Hz)']
    for k, (ax, axb, dat, cm) in enumerate(zip(color_ax, axs_bar, data1, cmaps)):
        vmax = np.max(dat) * 1.1
        vmin = 0.0
        ax.patch.set_alpha(0.0)

        # Colorplot of the bump
        if saveformat == 'png':
            colorplot = ax.pcolormesh(tpoints, theta, dat.T, vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cm))
            colorplot.set_edgecolor('face')
            ax.set_ylabel(r'$\theta\ (^\circ)$', labelpad=-1, fontsize=8)
            plt.setp(ax.get_xticklabels(), visible=False)

            # Colorbar
            cbar = plt.colorbar(colorplot, cax=axb, fraction=1.0, orientation='vertical', aspect=15)
            cbar.solids.set_edgecolor("face")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        if k != 1:
            axb.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        else:
            axb.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    cmap = plt.cm.get_cmap('hot')
    ax_in.plot(tpoints, np.rad2deg(ph[:, trial]), color=cmap(1.0), linewidth=1.0)

    ax_in.set_ylim(-180, 180)
    ax_in.set_yticks([-180, -90, 0, 90, 180])
    ax_ss.set_ylim(-90, 90)
    ax_ss.set_yticks([-90, 0, 90])

    # Stimulus
    nframes = kwargs.get('nframes', 8)
    labels = ['x%d' % k for k in range(1, nframes + 1)]
    stim_phases = data_trial[labels].to_numpy(int)
    stim_times = np.array([0, cue + dcue])
    logging.debug('Orientations of the frames are: %s' % stim_phases)

    stim_color = mcolors.to_rgba('cornflowerblue', 0.3)
    ax_st.bar(stim_times, stim_phases, align='edge', width=cue * 1.0, lw=1.0, ec='cornflowerblue',
              fc=stim_color, label=r'$\phi_i^{\text{stim}}$')
    ax_st.plot(tpoints, np.zeros_like(tpoints), linestyle=(0, (10, 3)), lw=0.5, color='black')
    ax_st.legend(frameon=False, fontsize=8, loc='upper left')
    ax_st.set_ylim(-25, 25)
    ax_st.set_yticks([-20, 0, 20])
    ax_st.set_ylabel(r'$\theta\ (^\circ)$', fontsize=8)
    ax_st.set_xlabel(r'Time (s)')
    ax_st.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax_st.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Snapshot plots (2nd column)
    # (integration circuit, ssn, stimulus)
    colors = [blue, blue, red]
    alphas = [0.5, 1.0]
    for row, (data, axs) in enumerate(zip([r_data, s_data[:, :, 0:n], stimulus[:, 0]], [axs_in, axs_ssn, axs_st])):
        vmax = np.max(data[:, trial]) * 1.1
        vmin = 0.0
        for k, tr in enumerate(idx_trials):
            ax1 = axs[k][0]
            ax2 = axs[k][1]
            for c, idx in enumerate(idx_times):
                ax1.plot(theta, data[idx, tr], color=colors[k], alpha=alphas[c])
                if row == 0:
                    ax2.plot(theta, stimulus[idx, 5, tr], color='gray', alpha=alphas[c] - 0.4)
                    ax_dc.plot(tpoints[idx], np.max([d1, d2]) * 1.3, 'v', markersize=6, clip_on=False, color=blue,
                               alpha=alphas[c])
                if row == 1 and c == 1:
                    ax2.plot(theta, stimulus[idx, 3, tr], color='gray', alpha=alphas[c] - 0.4)

            if row == 0:
                ax1.set_xlim([-180, 180])
                ax1.set_xticks([-180, -90, 0, 90, 180])
                ax2.set_ylim([0, np.max(dec_fb) * 1.1])
                ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                avg = dataframe.loc[dataframe.chosen == tr].average.to_numpy()
                estim = dataframe.loc[dataframe.chosen == tr].estim.to_numpy()
                axtop = ax1.twiny()
                axtop.set_xlim([-30, 30])
                axtop.spines['top'].set_bounds((-20, 20))
                axtop.spines['left'].set_visible(False)
                axtop.spines['right'].set_visible(False)
                axtop.grid(False)
                axtop.plot(np.array([avg, avg]), np.array([vmax - 5, vmax + 5]), lw=1, color='black', clip_on=False)
                axtop.plot(np.array([estim]), np.array([vmax + 10]), marker='v', color='black', markersize=8,
                           clip_on=False)
                axtop.set_xticks([-20, 0, 20])
                axtop.set_xticklabels(['-20', '', '20'])
                axtop.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                if k == 2:
                    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            else:
                ax1.set_xlim([-90, 90])
                ax1.set_xticks([-90, -45, 0, 45, 90])
            if row == 1:
                plt.setp(ax1.get_xticklabels(), visible=False)
                ax2.set_ylim([0, np.max(stimulus[:, 3]) * 2.0])
                if k == 2:
                    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            if row == 2:
                ax2.axes.yaxis.set_ticks([])
                ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax2.spines['right'].set_visible(False)
            if k == 0:
                ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))

            ax1.set_ylim([vmin, vmax])

    # cmap = plt.cm.get_cmap('Reds')
    # norm = MidpointNormalize(midpoint=0.5, vmin=-0.5, vmax=1.2)
    # idx_times = (np.abs(np.tile(tpoints, (2, 1)).T - np.array([0.5, 2.0, 3.25]))).argmin(axis=0)
    # for k, idx in enumerate(idx_times[0:2]):
    #     ax_p1.plot(theta, r_e[idx], color=cmap(norm(tpoints[idx])), label=r'$%.2f$ s' % tpoints[idx], linewidth=0.5)
    #     ax_p2.plot(theta, stim[idx] + att_mod_e[idx], color=cmap(norm(tpoints[idx])), label=r'$%.2f$ s' % tpoints[idx],
    #                linewidth=0.5)
    #     ax_p2.plot(theta, stim[idx], color=cmap(norm(tpoints[idx])), label=r'$%.2f$ s' % tpoints[idx],
    #                linewidth=0.5, linestyle='dashed')
    #     ax_p3.plot(theta, sensory_stim[idx], color=cmap(norm(tpoints[idx])), label=r'$%.2f$ s' % tpoints[idx],
    #                linewidth=0.5)
    #     ax_p3.plot(theta, fake_stim[idx], color=cmap(norm(tpoints[idx])), label=r'$%.2f$ s (fake)' % tpoints[idx],
    #                linewidth=0.5, linestyle='--')
    #     ax_p4.plot(theta, att_mod_e[idx], color=cmap(norm(tpoints[idx])), label=r'$%.2f$ s (exc)' % tpoints[idx],
    #                linewidth=0.5)
    #     ax_p4.plot(theta, att_mod_i[idx], color=cmap(norm(tpoints[idx])), label=r'$%.2f$ s (inh)' % tpoints[idx],
    #                linewidth=0.5, linestyle='--')
    #     ax_c.plot(tpoints[idx], 200, 'v', markersize=6, clip_on=False, color=cmap(norm(tpoints[idx])))
    #

    rectspan(cue, cue + dcue, [ax_dc, ax_gm], color="gray", linewidth=0.5, alpha=0.2)
    rectspan(cue, cue + dcue, [ax_st], color="gray", linewidth=0.5, alpha=0.2)

    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        fig.set_tight_layout(False)
        sigma, i0 = itemgetter('sigmaOU', 'i0')(kwargs)
        bump = 'nobump' if kwargs.get('nobump', False) else 'bump'
        if np.all([i0, sigma]):
            filename = directory + f"full_trial_{bump}_{trial}_summary_i0-{i0:.2f}_sigma-{sigma:.2f}"
        else:
            filename = directory + f"full_trial_{bump}_t-{trial}_{ntrials}_trials"
        save_plot(fig, filename, **kwargs)


def plot_ring_vs_amp(i0s=(0.05,), save=False, **kwargs):
    """ Plot that compares the dynamics of the amplitude equation and that of the ring model showing individual trials,
    average estimates, psychometric curve and psychophysical kernels.

    :param list or tuple or np.ndarray i0s: Values of the excitability of the system.
    :param bool save: Save the plot in different formats.
    :return:
    """
    (ering, eamp), (w_ring, w_amp), (pmring, pmamp) = get_amp_eq_stats(i0s, rinit=kwargs.get('pkrinit', 0.2), **kwargs)
    pmring_reg, pmring_dat = pmring
    pmamp_reg, pmamp_dat = pmamp

    custom_plot_params(1.0, latex=True)
    plt.rcParams['figure.dpi'] = 300

    # Axes location variables
    fig_options = dict(left=0.12, right=0.98, top=0.98, bottom=0.1, hspace=0.5, wspace=0.2)

    # Create figure and axes
    figsize = kwargs.pop('figsize', (6, 3.4))
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=figsize, sharex='row', sharey='row', gridspec_kw=fig_options)
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(False)

    for axb in axes.ravel():
        axb.patch.set_alpha(0.0)
        axb.grid(False)
    # Delete some axes
    mod_axes(axes)

    # Actual plotting starts here
    # #### #### #### #### #### ##
    dashed = (0, (10, 3))
    cs = ["#139fff", "#ff6929"]
    # Estimation plots
    axs = axes[0]
    xlabel = kwargs.pop('xlabel', r"Avg. orientation $(^\circ)$")
    ylabel = kwargs.pop('ylabel', r"Estimation $(^\circ)$")
    el = kwargs.pop('eslim', 90)
    eslims = (-el - 5, el + 5)
    esticks = [-el, -int(el / 2), 0, int(el / 2), el] if el % 2 == 0 else [-el, 0, el]
    for k, (er, ea) in enumerate(zip(ering, eamp)):
        axs[k].plot([-90, 90], [-90, 90], ls=dashed, color='black', lw=0.5)
        mod_ax_lims(axs[k], xlims=eslims, ylims=eslims, xlabel=xlabel, xticks=esticks, yticks=esticks,
                    xbounds=(-el, el), ybounds=(-el, el), xformatter='%d', yformatter='%d')
        axs[k].plot(er[0], er[1], color=cs[0], label='Ring')
        axs[k].fill_between(er[0], er[1] - er[2], er[1] + er[2], color=cs[0], alpha=0.3)
        axs[k].plot(ea[0], ea[1], color=cs[1], label='Amp. eq.')
        axs[k].fill_between(ea[0], ea[1] - ea[2], ea[1] + ea[2], color=cs[1], alpha=0.3)

        if k == 0:
            axs[k].set_ylabel(ylabel)
            axs[k].legend(frameon=False)
        else:
            plt.setp(axs[k].get_yticklabels(), visible=False)
    # PPK plots
    axs = axes[1]
    for k, (wr, wa) in enumerate(zip(w_ring, w_amp)):
        frames = list(range(1, len(wr) + 1))
        axs[k].plot(frames, wr, color=cs[0])
        axs[k].plot(frames, wa, color=cs[1])
        mod_ax_lims(axs[k], xlims=(0.5, frames[-1]), xticks=frames, ylims=(-0.01, 0.3), yticks=[0, 0.1, 0.2, 0.3],
                    xbounds=(frames[0], frames[-1]), ybounds=(0, 0.3), xlabel='Frame number',
                    xformatter='%d', yformatter='%.1f')
        if k == 0:
            axs[k].set_ylabel(r"Stim. Impact")
        else:
            plt.setp(axs[k].get_yticklabels(), visible=False)
        axs[k].text(1.0, 1.0, r"$I_0 = %.2f$" % i0s[k], transform=axs[k].transAxes, ha='right', va='top', clip_on=False)

    # Psychometric plots
    axs = axes[2]
    ylabel = r"Probability CW"
    for k, (rr, rd, ar, ad) in enumerate(zip(pmring_reg, pmring_dat, pmamp_reg, pmamp_dat)):
        mod_ax_lims(axs[k], xlims=(-65, 65), ylims=(-0.05, 1.0), xlabel=r"Average orientation $(^\circ)$",
                    xticks=[-60, -30, 0, 30, 60], yticks=[0, 0.5, 1],
                    ybounds=(0, 1), xbounds=(-60, 60), xformatter='%d', yformatter='%.1f')

        axs[k].plot(rr[0], rr[1], color=cs[0])
        axs[k].plot(ar[0], ar[1], color=cs[1])
        axs[k].plot(rd[0], rd[1], 'o', clip_on=False, markersize=3.0, markerfacecolor='white', markeredgewidth=1.0,
                    color=cs[0])
        axs[k].plot(ad[0], ad[1], 'o', clip_on=False, markersize=3.0, markerfacecolor='white', markeredgewidth=1.0,
                    color=cs[1])
        if k == 0:
            axs[k].set_ylabel(ylabel)
        else:
            plt.setp(axs[k].get_yticklabels(), visible=False)

    if save:
        directory = 'figs/amp_vs_ring/'
        fig.set_tight_layout(False)
        filename = directory + f"amp_stats"
        save_plot(fig, filename, overwrite=False, auto=True, saveformat=['png', 'pdf', 'svg'])

    return fig
