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
 * Any other script that produces similar data frames.

The library contains two type of methods, those intended for statistical analysis and those that
produce figures with the processed data:

Trial generators
----------------

.. autosummary::
   :toctree: generated/

   beta_random        Generate trials with their average value following a beta distribution.
   uniform_random     Generate trials by selecting the frames uniformly (average ~ n-convolution).
   uniform_targeted   Generate trials by selecting the frames uniformly and filtering to achieve a
                       target average.
   random_sample      Similar to :func:`uniform_targeted` but simpler version.

Constructor and wrapper
-----------------------

.. autosummary::
   :toctree: generated/

   generate_samples   A method that guides the generation of trials following specific criteria such as the
                       distribution.
   new_trialdata      A wrapper method that call the above function.

Auxiliary methods
-----------------

.. autosummary::
   :toctree: generated/

   stack_frames       Auxiliary function that gathers all the information of the frames in a large column.

Implementation
--------------

.. todo::

   Give a brief description.
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import circmean

logging.getLogger('trial_generator').addHandler(logging.NullHandler())

__author__ = 'Jose M. Esnaola-Acebes'
__docformat__ = 'reStructuredText'


def beta_random(alpha, beta, **kwargs):
    """ Function that generates samples of stimuli composed of a number of
    frames taken from a beta distribution with parameters 'alpha' and 'beta'.

    :param float alpha:  alpha parameter of the beta distribution.
    :param float beta: beta parameter of the beta distribution.
    :param kwargs: keyword arguments passed to np.random.beta.
    :return: an array with randomly generated beta-distributed samples.
    :rtype: np.ndarray
    """
    return np.array(180.0 * np.random.beta(alpha, beta, **kwargs) - 90.0, dtype=int)


# noinspection PyUnusedLocal
def uniform_random(*args, **kwargs):
    """ Function that generates samples of stimuli composed of a number of
    frames taken from a uniform distribution between -90 and 90.

    :param args: dummy list of variables.
    """

    size = kwargs.get('size', (10000, 8))
    return np.random.randint(-90, 90, size=size)


def uniform_targeted(target, tolerance, **kwargs):
    """ Function that generates samples of stimuli composed of a number of
    frames taken from a uniform distribution between -90 and 90, and filtered
    such that the average of the samples (per trial) meets target +/- tolerance requirement.

    :param float target: Value of the desired average value.
    :param float tolerance: Admitted deviation.
    :param kwargs: additional keyword arguments (``size`` of matrix and ``chunk``).
    :return: array with randomly generated stimulus samples.
    :rtype: np.ndarray
    """
    num_trials, num_frames = kwargs.get('size', (10000, 8))
    chunk = kwargs.get('chunk', 1000)

    ss = []
    while len(ss) < num_trials:
        samp = np.random.randint(-90, 90, size=(chunk, num_frames))
        # mean = samp.mean(axis=1)  # linear Average
        mean = circmean(samp, low=-180, high=180, axis=1)  # Circular average
        selected = samp[np.abs(mean - target) < tolerance]
        if len(selected) > 0:
            ss.extend(selected)
        else:
            continue
    return np.array(ss)[:num_trials]


def random_sample(possible_orientations, **kwargs):
    """Randomly generate samples of a given ``size`` out of ``possible_orientations``.

    :param np.ndarray or list possible_orientations: vector with the possible orientations that can have each
                                                     stimulus frame.
    :param kwargs: additional keyword arguments (``size``).
    :return: array with randomly generated stimulus samples.
    :rtype: np.ndarray
    """
    num_orientations = len(possible_orientations)
    size = kwargs.get('size', (100, 8))
    sample = np.ones(size, dtype=int)
    sample_idx = np.ones(size, dtype=int)

    for trial in range(size[0]):
        sample_idx[trial] = np.random.randint(0, num_orientations - 1, size[1])
        sample[trial] = np.take(possible_orientations, sample_idx[trial])
    return sample


def generate_samples(num_trials, num_frames, distribution=uniform_random, **kwargs):
    """Constructor function. Generates trials following a specific design of the stimulus frame distribution.

    :param int num_trials: number of trials to be generated.
    :param int num_frames: number of frames that each stimulus has.
    :param function or str distribution: distribution of the trials.
    :param kwargs: additional keyword arguments passed to the ``distribution`` function.
    :return: data-frame containing the generated trials, where each column corresponds to a frame number and each
             row to a trial.
    :rtype: pd.DataFrame
    """
    logging.debug('Generating samples using %s' % distribution.__name__)
    mu0 = kwargs.pop('mu', 0)
    sigma0 = kwargs.pop('sigma', 10)
    mu = (mu0 + 90) / 180
    sigma = sigma0 / 180
    args = (mu0, sigma0)
    kwargs['size'] = (num_trials, num_frames)

    if distribution == beta_random:
        alpha = mu * ((mu * (1 - mu)) / (num_frames * sigma ** 2) - 1)
        beta = (1 - mu) * ((mu * (1 - mu)) / (num_frames * sigma ** 2) - 1)
        args = (alpha, beta)

    if distribution == random_sample:
        po = kwargs.pop('possible_orientations')
        args = (po,)

    samples = distribution(*args, **kwargs)
    average = samples.mean(axis=1)
    average_circ = circmean(samples, low=-180, high=180, axis=1)
    choice = (average > 0)
    choice_circ = np.array((average_circ > 0))
    choice[average == 0] = np.nan
    choice_circ[average_circ == 0] = np.nan

    sample_dict = {}
    columns = []
    for frame in range(num_frames):
        label = 'x%d' % (frame + 1)
        sample_dict[label] = samples[:, frame]
        columns.append(label)

    sample_dict['average'] = average
    sample_dict['average_circ'] = average_circ
    sample_dict['bincorrect'] = choice
    sample_dict['bincorrect_circ'] = choice_circ
    columns.extend(['average', 'average_circ', 'bincorrect', 'bincorrect_circ'])
    df = pd.DataFrame(sample_dict, columns=columns)
    return df


def new_trialdata(num_trials, num_frames, categories=(0,), **kwargs):
    """Function that generates stimuli and returns a dataframe containing the design matrix.

    :param int num_trials: number of trials to be generated.
    :param int num_frames: number of frames each trial has.
    :param list or tuple or np.ndarray of float categories: trial generation is constrained to these categories only.
    :param kwargs: additional keyword arguments used to tweak the distribution of trials.
    :return: a fully formatted data-frame with all the information about the stimuli.
    :rtype: pd.DataFrame
    """
    stim_data = []

    if list(categories) == list((0,)):
        categories = np.linspace(-75, 75, 16)  # Corresponds to a histogram with bins=np.linspace(-80, 80, 17)

    if kwargs.get('distribution', uniform_targeted) == 'random_sample':
        df = generate_samples(num_trials, num_frames, possible_orientations=np.array(categories),
                              distribution=random_sample)
        # df['category'] = df['average'].copy()
        df['category'] = df['average_circ'].copy()
    else:
        for cat in categories:  # Generate a dataframe for each category
            num_cat = num_trials // len(categories)
            logging.debug('Generating %d trials for category: %s.' % (num_cat, cat))
            if cat == categories[len(categories) // 2]:
                num_cat += num_trials % len(categories)
            stim_data.append(generate_samples(num_cat, num_frames,
                                              distribution=kwargs.get('distribution', uniform_targeted), mu=cat,
                                              sigma=kwargs.get('sigma', 5)))

        # Concatenate categories
        if len(set(categories)) != len(categories):
            df = pd.concat(stim_data, keys=set(categories))
        else:
            df = pd.concat(stim_data, keys=categories)

    # Tidy up the dataframe
    df = df.reset_index()
    df.index.name = 'Trial'
    df = df.rename(columns=dict(level_0='category', level_1='sub_trial', index='sub_trial'))
    logging.debug(f"Columns of the dataframe: {df.columns}")

    if kwargs.get('save_stim', False):
        df.to_csv('./obj/Stim_df_%d_%d.csv' % (num_trials, num_frames))
        df.to_pickle('./obj/Stim_df_%d_%d.npy' % (num_trials, num_frames))

    return df


def stack_frames(df):
    """Auxiliary function to transform a dataframe ``df`` stacking the data of all stimulus frames into a
    single column.

    :param pd.DataFrame df: data-frame containing the stimuli information.
    :return: a transformed data-frame.
    :rtype: pd.DataFrame
    """
    df = df.set_index(['category', 'sub_trial', 'average', 'bincorrect'], append=True)
    df = df.stack().reset_index().set_index(['Trial', 'category', 'sub_trial', 'average', 'bincorrect'])
    df = df.rename(columns={'level_5': 'frame', 0: 'orientation'})

    return df
