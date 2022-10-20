"""
Simulation of a bump attractor network for stimulus integration
===============================================================

.. module:: ring_simulation
   :platform: Linux
   :synopsis: simulation of the ring attractor network for stimulus integration tasks.

.. moduleauthor:: Jose M. Esnaola-Acebes <josemesnaola@gmail.com>

This script runs a simulation of a bump attractor network (**cite**) to model stimulus integration in visually
guided perceptual tasks.


Methods included in :mod:`ring_simulation`
------------------------------------------

.. autosummary::
   :toctree: generated/

   set_up_simulation       Set up the simulated task environment.
   process_results         Pre-Process the results of the simulation.
   simulate_task_parallel  Prepare the simulation variables to be run in parallel.
   run_simulation          Simulation's main body: run the simulation.
   do_bifurcation          Automatically guided bifurcation generator.

Default parameters
------------------

Table.

Implementation
--------------

Description needed.
"""

import logging
import os
import sys
import timeit

import numpy as np
import pandas as pd

sys.path.insert(0, './lib/')
from lib_sconf import log_conf, create_dir, path_exists, get_paths, save_obj, check_overwrite, Parser

file_paths = get_paths(__file__)
sys.path.insert(0, file_paths['s_dir'] + '/lib/')
from lib_NeuroDyn import sigmoid

from lib_plotting import plt
import gc

from lib_ring import connectivity, ou_process, icritical, coefficient_a, coefficient_c, load_ic, save_ic, \
    compute_phase, r_roots, sigmoid_pw_v
from lib_parallel import ParallelSimulation
from trial_generator import new_trialdata
import multiprocessing as mp

logging.getLogger('ring_simulation').addHandler(logging.NullHandler())

__author__ = 'Jose M. Esnaola-Acebes'

init_options = {'tau': 0.02, 'dt': 2E-4, 'tmax': 2.05, 'n': 200, 'm': [-2.0, 1.0, 0.5], 'dcue': [0.75, 1.0],
                'tauOU': 1E-3, 'sigmaOU': 0.0, 'nframes': 8, 'cue_duration': 0.250, 'save_interval': 0.01}

logger = None


def kernel(theta, choice='left', maxv=1.0, shape='triangle'):
    k = np.ones_like(theta) * 0.0
    n = len(k)

    if shape == 'linear':
        k = np.ones_like(theta) * maxv
        if choice == 'left':
            k[n // 2:] = 0
        else:
            k[:n // 2] = 0
    elif shape == 'triangle':
        if choice == 'left':
            k = maxv * (-theta / np.pi)
        elif choice == 'right':
            k = maxv * (theta / np.pi)
    return k


def run_simulation(samples, init_ph, inputs, opts, **kwargs):
    """Simulation's main function. Runs the integration time-loop.

    :param np.ndarray samples: Matrix that includes the information of the stimuli.
    :param np.ndarray init_ph: Array with the initial positions of the bump.
    :param np.ndarray inputs: A matrix containing all possible stimuli inputs.
    :param dict opts: Dictionary that includes all necessary parameters and options.
    :param kwargs: Additional keyword arguments passed by the :mod:`ParallelSimulation`.
    :return: Simulation's results.
    """
    save = opts.get('save_fr', False)
    is_bump = not opts.get('no_bump', False)  # Default is True
    tmax = opts.get('tmax', 2.05)  # Maximum time
    dt = opts.get('dt', 2E-4)  # Time step
    tau = opts.get('tau', 20E-3)  # Neurons' time constant
    n = opts['n']
    # Number of trials is determined by the samples table
    ntrials = len(samples[:, 0])

    # Stimulus
    n_frames = opts.get('nframes', 8)
    cue_duration = opts.get('cue_duration', 0.250)
    tcue = [0.0, cue_duration]
    tcues = np.array(np.array(tcue) / dt, dtype=int)
    lencue = tcues[1] - tcues[0]
    frame = 0  # Frame counter
    total_stimulus_duration = n_frames * cue_duration
    if tmax < total_stimulus_duration:
        tmax = total_stimulus_duration + 0.05

    # Decision period
    urgency_duration = opts.get('urgency_duration', 0.5)

    # Variables (time, space and firing rate)
    tpoints = np.arange(0, tmax, dt)
    theta = np.arange(n) / n * (2 * np.pi) - np.pi
    nsteps = len(tpoints)

    rsteps = nsteps if save == 'all' else 2
    block = kwargs.get('block', 0)
    logger.debug(f'Simulating block {block}...')

    # Saving the firing rate profile
    saving_counter = 0
    if save == 'partial':
        # 2.05 seconds are simulated => with a 50 ms interval 41 points are saved
        saving_interval = opts.get('save_interval', 0.05)
        tpoints_save = np.arange(0, tmax - dt, saving_interval)
        save_steps = len(tpoints_save)
        save_steps = save_steps if save_steps % 2 == 0 else save_steps - 1
        r_profiles = np.ones((save_steps, ntrials, n)) * .0
        saving_sampling = nsteps // save_steps
        logging.debug('Saving firing rate every %d ms. Number of points: %d. Sampling freq: %d' %
                      (saving_interval * 1E3, save_steps, saving_sampling))
        # Saving decision network activity
        d12 = np.ones((save_steps, 2, ntrials)) * 0.0
        lri = np.ones((save_steps, 2, ntrials)) * 0.0
        # Saving the perfectly integrated stimulus
        total_stim = np.zeros((ntrials, n))
        perfect_phase = np.ones((save_steps, ntrials))
        A_save = np.ones((save_steps, ntrials), dtype=np.complex) * .0
        i0_save = np.ones(save_steps) * .0
    else:  # Dummy variables to avoid annoying warnings
        r_profiles = np.array([[[0]]])  # 3-dim array for compatibility reasons
        saving_sampling = 0
        d12 = np.array([[[0]]])
        lri = np.array([[[0]]])
        total_stim = np.array([[0]])
        perfect_phase = np.array([[0]])
        A_save = np.array([[0]], dtype=np.complex)
        i0_save = np.array([0])

    # Firing rate matrix
    r = np.ones((rsteps, ntrials, n)) * 0.01
    phases = np.zeros((n_frames, ntrials))

    # Integration times
    response_times = np.ones(ntrials) * -1.0

    # Connectivity
    modes = opts.get('m', [-1.0, 2.0, 0.2])
    cnt = connectivity(n, modes)
    # OU process (be careful with the RAM!!!)
    sigmaou = opts.get('sigmaOU', 0.6)
    tau_ou = opts.get('tauOU', 1E-3)
    worker = ParallelSimulation.worker(real=True)
    seed = np.random.seed((int(timeit.default_timer() * 10000 + worker) // worker) % (2 ** 32 - 1))
    ou = ou_process(dt, nsteps + 1, 0.0, sigmaou, tau_ou, ntrials, n, seed=seed)  # (stim_steps, ntrials, n)

    # Sensory inputs to the integration circuit
    icrit = icritical(modes[0], modes[1], tau=tau)
    i0_over = opts.get('i0', 0.01)
    i1 = opts.get('i1', 0.01)
    # if opts.get('init_save', False):  # This is no longer needed as it is better to save the initial conditions taking
    #     i1 = 0.0                      # into account the effect of the input on the bump
    input1 = i1 * np.take(inputs, samples[:, frame], axis=0)  # (ntrials x n)

    # Phase computation
    cosine = np.cos(theta)
    sine = np.sin(theta)

    # Amplitude equation
    A = np.ones((rsteps, ntrials), dtype=np.complex) * 0.0
    a = coefficient_a(modes[0], i0_over - 2.0 * sigmaou ** 2, icrit, tau)
    c = coefficient_c(modes[0], modes[2], icrit, tau)
    if is_bump:
        A[-1] = A[-1] * 0 + np.sqrt(0j + a / -c) * np.exp(1j * 0.0)
        A[0] = A[-1]
    pre_d = 1.0 / 2.0 * 1.0 / tau / modes[1]
    d = pre_d * i1
    n1 = np.sqrt(4) * pre_d * np.exp(-1j * np.pi) * np.fft.fft(ou, axis=-1)[:, :, 1] / n
    ts_v = np.zeros((nsteps, ntrials))
    orientations = samples - 90  # recover the original angles
    for nframe, orientation in enumerate(orientations.T):  # orientations: nframes x ntrials
        t0 = int((cue_duration * nframe) / dt)
        t1 = int((cue_duration * (nframe + 1)) / dt)
        ts_v[t0:t1] = np.deg2rad(orientation)
    amp_phases = np.zeros((n_frames, ntrials))

    # Initial condition of the bump (load the profile from dictionary)
    if opts.get('init_load', False) or is_bump:
        effe_i0_over = opts.get('i0_init', i0_over - 2.0 * sigmaou ** 2)
        r0 = load_ic('./obj/r0_initial_conditions.npy', critical=('n',), n=n, i0_over=effe_i0_over,
                     w0=modes[0], w1=modes[1], w2=modes[2])
        if r0 is not False:
            r[-1] = np.repeat([r0], ntrials, axis=0)  # (ntrials x n)

        # Set the initial phase of the bump to the phase of the nobump simulation after the first stimulus frame
        if opts.get('nobias', False):
            logging.debug("Simulating an unbiased system.")
            for k, ph in enumerate(init_ph):
                r[-1, k] = np.roll(r[-1, k], ph)

        # Set the initial condition of the amplitude equation
        psi0 = compute_phase(r[-1], n, cosine, sine)
        R0, _ = r_roots(mu=a, i1=d, c=c, theta=0.0)
        A[-1] = A[-1] * 0 + R0[0] * np.exp(1j * psi0)
        A[0] = A[-1]

    """
    Readout network set-up
    """
    dtaue = opts.get('dtaue', 20E-3)  # 20 ms
    dtaui = opts.get('dtaui', 20E-3)  # 20 ms
    selff = opts.get('js', 1.9)
    cross = opts.get('jc', 1.0)
    g = opts.get('g', 1.0)
    i_i = opts.get('i_i', 0.2)
    d1 = np.ones((rsteps, ntrials)) * 10.6
    d2 = np.ones_like(d1) * 10.6
    ri = np.ones_like(d1) * 21.0

    # Urgency signal (sub-critical pitchfork)
    i_rest = opts.get('i_rest', 0.33)  # Bistable regime
    i_urge = opts.get('i_urge', 0.50)  # pure WTA regime
    go_interval = opts.get('go_int', 0.5)  # Urgency signal time interval (seconds)
    go_signal = total_stimulus_duration * 1.0
    go_tstep = int(go_signal / dt)
    urgency_signal = np.ones(nsteps) * i_rest
    urgency_signal[go_tstep:go_tstep + int(go_interval / dt)] = i_urge

    # Left and right inputs to the decision network
    max_signal = opts.get('max', 1.0)
    kernel_l = kernel(theta, 'left', maxv=max_signal, shape='linear')
    kernel_r = kernel(theta, 'right', maxv=max_signal, shape='linear')
    left_input = 0.0
    right_input = 0.0

    # Decision-making vectors
    decided = np.ones(ntrials, dtype=bool) * 0
    decision_threshold = opts.get('threshold', 50)
    decision_phase = np.ones(ntrials) * 0.0

    # #############################################################################################
    logging.debug("New simulation launched.")

    time_init = timeit.default_timer()
    tstep = 000

    temps = 0.0
    np.seterr(all='raise')
    ttau = dt / tau
    stim_input = input1 * 0.0  # (ntrials, n)
    stim_noise = 0.0
    amp_noise = 0.0
    s1 = 0
    i0_over_tmp = i0_over * 1.0
    i0_over = -icrit * 0.25

    while temps < (tmax - dt):
        # Time step variables
        kp = tstep % rsteps
        k = (tstep + rsteps - 1) % rsteps

        # 2. Presynaptic inputs
        s = 1.0 / n * np.dot(r[k], cnt)  # (ntrials, n) x (n, n) = (ntrials, n)
        # s = 1.0 / n * np.einsum('ij,ijk->ij', r[k], cnt)  # (ntrials, n) x (n, n) = (ntrials, n)
        # s = 1.0 / n * np.array([r[k, l] @ cnt[l] for l in range(ntrials)])  # (ntrials, n) x (n, n) = (ntrials, n)
        # Monitor changes in i0
        if tstep == 0:
            i0_over = i0_over_tmp * 1.0
            stim_input = input1 * 1.0  # (ntrials, n)
        i0 = icrit + i0_over

        # Stimulus i is ON
        if tcues[0] <= tstep:
            stim_noise = ou[s1]  # (ntrials, n)
            amp_noise = n1[s1]
            s1 += 1
        if tstep == tcues[1]:  # We jump to the next frame or end the stimulus
            # We store the phase of the bump at the end of the stimulus frame
            phases[frame] = compute_phase(r[k], n, cosine, sine)
            amp_phases[frame] = np.angle(A[k])

            frame += 1  # Add one to the counter
            if frame < n_frames:
                tcues += lencue  # Change the interval of the frame to the next period of the stimulus
                stim_input = i1 * np.take(inputs, samples[:, frame], axis=0)  # (ntrials x n) Change the orientation
            else:  # End of the stimulus
                stim_input = 0.0
                stim_noise = 0.0
                amp_noise = 0.0
                # i0_over = -icrit * 0.25
        elif not np.alltrue(decided):  # Check if the decision has been made
            decided_now = ((d1[k] >= decision_threshold) | (d2[k] >= decision_threshold))
            new_decided = (decided_now & ~decided)
            if len(np.argwhere(new_decided)) > 0:
                response_times[np.argwhere(new_decided)] = temps - go_signal
                decision_phase[np.argwhere(new_decided)] = compute_phase(r[k, np.argwhere(new_decided)],
                                                                         n, cosine, sine)
                decided = (decided | new_decided)

        # 3. Integration
        """
        Integration circuit
        """
        try:
            r[kp] = r[k] + ttau * (
                    -r[k] + sigmoid_pw_v(i0 + stim_input + stim_noise + tau * s, tau=tau))  # (ntrials, n)
        except FloatingPointError:  # Prevent UnderFlow errors by setting the firing rate to 0.0
            r[r < 1E-12] = 0.0
        """
        Amplitude equation
        """
        A[kp] = A[k] + ttau * (a * A[k] + d * np.exp(1j * ts_v[tstep]) + amp_noise + c * (A[k] * np.conj(A[k])) * A[k])

        """
        Decision circuit: the decision readout is performed by a second neural network consisting on a pair of 
        excitatory neural populations, both recurrently connected to a pool of interneurons. 
        This network [Wong and Wang (2006), Roxin and Ledberg (2008)] can be described with two identical recurrently 
        coupled neural populations, with local recurrent excitation and cross-coupled inhibition.
        
        The decision readout process starts with "an urgency signal" consisting on a non-specific external input
        targeting both populations, which will set the readout network in a WTA dynamical regime, where the activity 
        of one of the two populations will dominate over the other. Once a firing-rate threshold is reached the 
        decision (or the motor planning) is taken.
        """

        left_input = 1 / n / 2.0 * np.dot(r[k], kernel_l)
        right_input = 1 / n / 2.0 * np.dot(r[k], kernel_r)
        try:
            input_1 = dtaue * (selff * d1[k] - cross * ri[k]) + left_input + urgency_signal[tstep]
            input_2 = dtaue * (selff * d2[k] - cross * ri[k]) + right_input + urgency_signal[tstep]
            input_i = dtaui * g * (d1[k] + d2[k]) + i_i

            d1[kp] = d1[k] + dt / dtaue * (-d1[k] + sigmoid(input_1) / dtaue)
            d2[kp] = d2[k] + dt / dtaue * (-d2[k] + sigmoid(input_2) / dtaue)
            ri[kp] = ri[k] + dt / dtaui * (-ri[k] + sigmoid(input_i) / dtaui)
        except FloatingPointError:
            d1[d1 < 1E-12] = 0.0
            d2[d2 < 1E-12] = 0.0
            ri[ri < 1E-12] = 0.0

        if save == 'partial' and tstep % saving_sampling == 0 and tstep != 0:
            r_profiles[saving_counter] = r[kp]
            d12[saving_counter] = np.array([d1[kp], d2[kp]])
            lri[saving_counter] = np.array([left_input, right_input])
            total_stim += (stim_input + stim_noise)
            perfect_phase[saving_counter] = compute_phase(total_stim, n, cosine, sine)
            A_save[saving_counter] = A[kp]
            i0_save[saving_counter] = i0
            saving_counter += 1

        temps += dt
        tstep += 1

    del ou
    gc.collect()

    temps -= dt
    tstep -= 1
    frame -= 1
    # Stop the timer
    logging.debug('Total time: {}.'.format(timeit.default_timer() - time_init))

    if opts.get('init_save', False):
        effe_i0_over = opts.get('i0_init', i0_over - sigmaou ** 2)
        save_ic('./obj/r0_initial_conditions.npy', r[-1, 0], n=n, i0_over=effe_i0_over,
                w0=modes[0], w1=modes[1], w2=modes[2])

    logging.debug('Preprocessing block data...')
    estimation = phases[frame]
    last_phase = compute_phase(r[(tstep + rsteps - 1) % rsteps], n, cosine, sine)
    bin_choice = np.array(((d12[-2, 1] - d12[-2, 0]) >= 0) * 2 - 1, dtype=int)
    bin_choice[bin_choice == 0] = 1

    if save == 'all':
        return estimation, bin_choice, response_times, phases, decision_phase, last_phase, r, np.array(
            [d1, d2]).swapaxes(0, 1), np.array([left_input, right_input]), perfect_phase, A, amp_phases, i0_save
    else:
        return estimation, bin_choice, response_times, phases, decision_phase, last_phase, r_profiles, d12, lri, \
               perfect_phase, A_save, amp_phases, i0_save


def simulate_task_parallel(opts, n_trials=10000, chunk=1000, **kwargs):
    n = opts.get('n', 200)  # Number of neurons (spatial dimension)
    theta = np.arange(n) / n * (2 * np.pi) - np.pi

    # Set-up simulation environment: stimuli.
    inputs, samples_idx, simu_data = set_up_simulation(theta, n_trials, opts, **kwargs)

    # Run simulated trials in parallel
    sys.stdout.flush()
    processes = opts.get('cpus', mp.cpu_count() // 2)  # Default: Take all cpus (assuming hyperthreading)
    # noinspection PyTypeChecker
    parallel_simu = ParallelSimulation(run_simulation, n_trials, chunk, processes,
                                       show_progress=not opts.get('quiet', False))
    results = parallel_simu((samples_idx, simu_data['init_ph_n'].to_numpy(dtype=int),), (inputs, opts))

    # Collect results
    logging.info('Merging chunked data...')
    return process_results(results, parallel_simu.n_b, opts.get('nframes', 9), chunk, simu_data, **kwargs)


def set_up_simulation(theta, n_trials, opts, **kwargs):
    n_frames = opts.get('nframes', 8)
    angle_resolution = opts.get('angle_resol', 1)

    num_categories = 16 if n_trials >= 16 else n_trials
    orient_categories = opts.get('stim', np.linspace(-75, 75, num_categories))  # Input categories
    sigma_stim = 5
    # num_categories = 31 if n_trials >= 31 else n_trials
    # orient_categories = opts.get('stim', np.linspace(-75, 75, num_categories))  # Input categories
    # sigma_stim = 2.5
    if np.all(orient_categories == 0) and not opts.get('init_save'):
        orient_categories = np.linspace(-75, 75, num_categories)

    # All possible orientations
    logging.info('Setting possible spatial inputs...')
    theta_m, orientations_m = np.meshgrid(theta, np.arange(-90, 90 + 1, angle_resolution))
    inputs = np.cos(theta_m - np.deg2rad(orientations_m))  # (num_stim x n)
    del theta_m, orientations_m
    gc.collect()

    # Setup dataframe
    stim_labels = ['x%d' % i for i in range(1, n_frames + 1)]
    phase_labels = ['ph%d' % i for i in range(1, n_frames + 1)]
    amp_phase_labels = ['amp_ph%d' % i for i in range(1, n_frames + 1)]
    cols = ['estim', 'binchoice', 'binrt'] + phase_labels + amp_phase_labels

    # Create random stimuli sequences:  (n_trials x num_frames)
    logger.info('Setting up stimuli dataframe.')
    stim_file = opts.get('stim_file', 'Stim_df_%d_%d.npy' % (n_trials, n_frames))
    if stim_file == 'default':
        stim_file = 'Stim_df_%d_%d.npy' % (n_trials, n_frames)
    if not path_exists('./obj/' + stim_file):
        logger.warning(f"Custom stim. file './obj/{stim_file}' not found.")

    if opts.get('custom', False) or not path_exists('./obj/' + stim_file):
        logger.debug('Generating new stimuli.')
        dist = opts.pop('distribution', 'uniform')
        if dist == 'random_sample':
            simu_data = new_trialdata(n_trials, n_frames, orient_categories, distribution='random_sample', **opts)
        else:
            simu_data = new_trialdata(n_trials, n_frames, orient_categories, sigma=sigma_stim, **opts)
        opts['distribution'] = dist
        simu_data['init_ph_n'] = np.ones(len(simu_data), dtype=int)
    else:
        logger.debug('Loading stimuli data from %s' % ('./obj/' + stim_file))
        simu_data = pd.read_pickle('./obj/' + stim_file)
        load_init_ph = True
        nobump_data = dict()
        try:
            nobump_data = pd.read_csv(f"./results/simu_10000_nobump_sigma-0.15_i0"
                                      f"-{opts.get('i0'):.2f}_frames-8_t-{opts.get('tmax'):.1f}.csv")
        except FileNotFoundError:
            try:
                nobump_data = pd.read_csv(f"./results/simu_10000_nobump_sigma-0.15_i0"
                                          f"-{opts.get('i0'):.2f}_frames-8_t-3.0.csv")
            except FileNotFoundError:
                simu_data['init_ph_n'] = np.ones(len(simu_data), dtype=int)
                load_init_ph = False
        if load_init_ph:
            simu_data['init_ph_n'] = ((nobump_data['ph1'] + 180) / 360 * len(theta) - len(theta) / 2).astype(int)

    simu_data = pd.concat([simu_data, pd.DataFrame(columns=cols)], sort=False).reset_index()
    simu_data[['estim', 'binrt'] + phase_labels + amp_phase_labels] = simu_data[['estim', 'binrt'] + phase_labels
                                                                                + amp_phase_labels].astype(float)
    simu_data[['binchoice', 'bincorrect', 'bincorrect_circ']] = simu_data[
        ['binchoice', 'bincorrect', 'bincorrect_circ']].astype(float)
    simu_data.rename(columns=dict(level_0='Trial'))
    simu_data.index.name = 'Trial'
    logging.debug('Stimuli categories: %s' % simu_data.category.unique())
    samples_idx = np.array(simu_data[stim_labels].to_numpy() + 90, dtype=int)  # type: np.ndarray
    mylog(0)

    return inputs, samples_idx, simu_data


def process_results(results, n_blocks, n_frames, chunksize, dataframe, **kwargs):
    """

    :param results: List of results from :func:`run_simulation`, arranged in blocks.
    :param int n_blocks: Number of blocks in which the results are distributed.
    :param int n_frames: Number of frames of the task stimulus.
    :param int chunksize: Size of each simulation block (trials per block).
    :param pd.DataFrame dataframe: Data-frame containing the design of the simulated task.
    :param kwargs: Additional keyword arguments.
    :return: Rearranged results.
    """

    phase_labels = ['ph%d' % i for i in range(1, n_frames + 1)]
    amp_phase_labels = ['amp_ph%d' % i for i in range(1, n_frames + 1)]
    r_profiles = []
    d12 = []
    lri = []
    perfect_phase = []
    A = []
    i0 = []

    for k in range(n_blocks):
        res = results[k]
        i1, i2 = k * chunksize, (k + 1) * chunksize - 1
        dataframe.loc[i1:i2, 'estim'] = np.rad2deg(res[0])
        dataframe.loc[i1:i2, 'binchoice'] = res[1]
        dataframe.loc[i1:i2, 'binrt'] = res[2]
        dataframe.loc[i1:i2, phase_labels] = np.rad2deg(res[3].T)
        dataframe.loc[i1:i2, 'phd'] = np.rad2deg(res[4].T)
        dataframe.loc[i1:i2, 'ph_last'] = np.rad2deg(res[5].T)
        r_profiles.append(res[6])
        d12.append(res[7])
        lri.append(res[8])
        perfect_phase.append(res[9])
        A.append(res[10])
        dataframe.loc[i1:i2, amp_phase_labels] = np.rad2deg(res[11].T)
        i0 = res[12]

    # reshaping from (blocks, rsteps/saving_counter, block_trials, n) to (rsteps/saving_counter, total_trials, n):
    r_profiles = np.concatenate(tuple(r_profiles), axis=1)
    d12 = np.concatenate(tuple(d12), axis=2)
    lri = np.concatenate(tuple(lri), axis=2)
    perfect_phase = np.concatenate(tuple(perfect_phase), axis=1)
    A = np.concatenate(tuple(A), axis=1)

    mylog(0)
    return dataframe, r_profiles, dict(d12=d12, lri=lri), perfect_phase, A, i0


if __name__ == '__main__':
    # -- Simulation configuration: parsing, debugging.
    pars = Parser(desc='Simulation of the bump attractor network.',
                  groups=('Parameters', 'Network', 'Stimulus', 'Decision_circuit'))
    conf_options = pars.opts
    logger, mylog = log_conf(pars.debug_level)

    try:
        conf_options.update(get_paths(__file__))
    except NameError:
        pass
    init_options.update(conf_options)
    num_trials = init_options.get('ntrials', 1000)
    overwrite = init_options.get('overwrite', False)
    results_dir = str(init_options.get('res_dir', './results/'))
    create_dir(results_dir)

    data, rdata, ddata, p_ph, amp, i0 = simulate_task_parallel(init_options, num_trials,
                                                               chunk=init_options.get('chunk', 100))
    # Create a file name for data saving
    bump = not init_options.get('no_bump', False)  # Default is True
    file_ref = ('bump_' if bump else 'nobump_')
    if bump:
        file_ref += ('' if init_options.get('nobias', False) else 'biased_')
    file_ref += ('sigma-%.2f_i0-%.2f' % (init_options['sigmaOU'], init_options['i0']))
    file_ref += ('_frames-%d_t-%.1f' % (init_options['nframes'], init_options['tmax']))

    # Save data
    if not init_options.get('no_save', False):
        sample_size = init_options.get('sample', 500)
        if sample_size > len(data):
            sample_size = len(data)
        random_choice = np.sort(np.random.choice(len(data), sample_size, replace=False))
        data['chosen'] = -1
        data.loc[random_choice, 'chosen'] = random_choice
        sampled_rdata = rdata[:, random_choice]
        sampled_ddata = ddata.copy()
        sampled_ddata['d12'] = sampled_ddata['d12'][:, :, random_choice]
        sampled_ddata['lri'] = sampled_ddata['lri'][:, :, random_choice]
        sampled_p_ph = p_ph[:, random_choice]
        sampled_amp = amp[:, random_choice]
        sampled_i0 = i0[:]

        results_dict = dict(conf=init_options, rates=sampled_rdata, data=data, ddata=sampled_ddata,
                            p_ph=sampled_p_ph, A=sampled_amp, i0=sampled_i0)
        filename = check_overwrite(results_dir + 'simu_%d_%s' % (num_trials, file_ref) + '.npy',
                                   force=overwrite, auto=True)
        filename, extension = os.path.splitext(filename)
        logger.info(f'Saving data to {filename}.npy ...')
        if save_obj(results_dict, filename, extension='.npy'):
            mylog(0)
        else:
            mylog.msg(1, up_lines=2)
        logger.info(f'Saving data-frame to {filename}.csv...')
        try:
            data.to_csv(filename + '.csv')  # Save the data-frame in csv format
            mylog(0)
        except FileNotFoundError:
            mylog(1)

    # Plotting
    if init_options.get('plot', False):
        logger.info('Plotting some preliminary results...')
        from lib_analysis import plot_trial

        f, ax = plot_trial(rdata, data, i0, save=not init_options.get('no_save', False), **init_options)

        if init_options.get('show_plots', True):
            logger.info('Showing plot now...')
            plt.show()
