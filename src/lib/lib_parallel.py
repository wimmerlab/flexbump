"""
Parallelization library for trial-based simulations
===================================================

.. currentmodule:: lib_parallel
   :platform: Linux
   :synopsis: module for handling parallel computation of simulations

.. moduleauthor:: Jose M. Esnaola-Acebes <josemesnaola@gmail.com>

This library contains a parallelization class (:class:`ParallelSimulation`) which is thought to
run trial-based simulations in parallel using the CPU multi-core capability. The class
distributes trials among the CPU cores using :mod:`multiprocessing` module in chunks of trials (``chunk_size``)
defined by the user.

.. note::

   This library is not designed to run spiking network simulations in parallel.


Methods included in :class:`ParallelSimulation`
-----------------------------------------------

.. autosummary::
   :toctree: generated/

   __init__           Initialization of the parallel framework.
   __call__           Execution: call to trial distributor and simulation.
   distribute_trials  Set up the iterator that is passed to :func:`pool.starmap_async`
   run_task_unpack    Intermediary function that unpacks arguments, separating
                       simulation-specific arguments and those related to the
                       parallelization process.

Static methods included in :class:`ParallelSimulation`
------------------------------------------------------

.. autosummary::
   :toctree: generated/

   poolcontext        Auxiliary function that modifies the behavior of a *Terminate* signal.
   worker             Function that extracts the first integer value out of a string:
                       useful to get the worker's *id*.
   print_table        Print a table with a given number of rows and columns, and their
                       associated titles and headers.

Implementation
--------------

In order to use this library, the main function that runs the simulation (``func``) must have a given format.
Specifically, input and output data must be formatted as n-dimensional :mod:`numpy` arrays where the last
axis of those matrices stores the trial indices.

Format of simulation input and output variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Input data must be passed to :class:`ParallelSimulation` when called in two separated sets of data:

 1. ``local_args``: Input data that is **distributed** among trials, e.g. different stimuli data.
 2. ``global_args``: Input data which is **common** (global) to all trials.

The distributed data, ``local_args`` must be passed as a :py:mod:`numpy` ``np.ndarray`` with shape
:math:`\\left(d_1, d_2, \\dots, d_n\\right)`, where :math:`d_i,\\ i = 1, 2, \\dots, n` are the dimensions
of the axis of the matrix, and **most importantly**, :math:`d_n` is the number of trials that we want to simulate.

.. admonition:: Example

    If we are simulating the firing rate response of a network to a time-varying stimuli of the form :math:`s(t)`
    which we could code as a :mod:`numpy` array, ::

        stimuli = np.random.randn(tpoints)

    then, if we assume that we want different stimuli at each trial, we would have a matrix containing all stimuli
    data of the form ::

        n_trials = 1000
        stimuli = np.random.randn(tpoints, n_trials)

On the other hand, common input data can have any format and will be passed to the simulation function ``fun``
as it is. Refer to :ref:`_parallel-execution` for more information on how to pass this data.

Output data is **collected** by the :py:mod:`multiprocessing` ``pool`` object, and its format will depend on the
``return`` clause used in the simulation function ``fun``. In any case, output data will be collected by the
multi-thread manager in a ``list`` where each element contains a block of the output data. Each block will contain
a number of trials equal to the selected ``chunk_size``. Therefore the number of blocks is determined by the
number of trials and the choice of ``chunk_size``: ``n_b = n_trials // chunk_size``.

.. _parallel-setup:

Set up the parallel framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parallel framework is initialize by creating a new :class:`ParallelSimulation` instance as follows,

.. code-block:: python

    parallel_simu = ParallelSimulation(simulation_func, n_trials, chunk_size, n_processes)

Where ``simulation_func`` is the main function that performs the simulation, ``n_trials`` is the number of trials,
``chunk_size`` is the number of trials that is simulated at each block, and ``n_processes`` is the number of
processes/workers that the pool will have. Typically, best performance is achieve with a number of workers equal to
the number of available threads, or alternatively to the number of cores that CPU has.

.. _parallel-execution:

Running the simulation: execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the initialization (:ref:`_parallel-setup`) is done we are ready to launch the simulation by calling
the just created parallelization object ``parallel_simu``

.. code-block:: python

    results = parallel_simu(local_args, global_args)

Where both ``local_args`` and ``global_args`` are passed as ``tuple``s.

.. admonition:: Example

    Taking the previously shown example of the input data ``stimuli``, and assuming that our simulation also needs
    some additional **global** arguments, such as an array of labels ``global_labels`` and a dictionary of
    parameters ``prmts_dict``, the execution should be written

    .. code-block:: python

        results = parallel_simu((stimuli, ), (global_labels, prmts_dict))

    .. note::

        If the distributed arguments consists in a unique object, as shown in the example above, then we may just
        pass the argument without the tuple format: ``parallel_simu(stimuli, (global_labels, prmts_dict))``.


"""

import ctypes
import logging
import multiprocessing as mp
import re
import signal
import time
import timeit
from contextlib import contextmanager
from multiprocessing.managers import SyncManager
import numpy as np

from lib_sconf import print_at

logging.getLogger('lib_parallel').addHandler(logging.NullHandler())

__author__ = 'Jose M. Esnaola-Acebes'
__docformat__ = 'reStructuredText'


class ParallelSimulation:
    """ Create a new pool-based multiprocess framework to run simulations in parallel. """
    def __init__(self, fun, n_trials, chunk_size=50, processes=2, show_progress=True):
        """ Initialization of :class:`ParallelSimulation`.

        :param function fun: function to run in parallel.
        :param int n_trials: number of trials that the simulation must be run.
        :param int chunk_size: number of trials that will be simulated in each function call.
        :param int processes: number of threads (workers) to be employed in the pool.
        :param bool show_progress: whether to show the progress of the execution through stdout.
        """
        self.__log = logging.getLogger('ParallelSimulation')
        self.fun = fun
        self.n = n_trials
        self.cs = chunk_size
        self.n_b = n_trials // chunk_size  # Number of blocks
        if self.n_b == 0:
            self.__log.error(f"Chunk size ({chunk_size}) should be smaller than the number of trials ({n_trials}).")
            raise RuntimeError
        self.processes = processes
        self.show_progress = show_progress
        # noinspection PyTypeChecker
        self.__computed_blocks = mp.Value(ctypes.c_int)  # Variable that will monitor started blocks

    def __call__(self, local_args, global_args):
        """Run the simulations in parallel.

        :param tuple local_args: arguments that will be distributed across the workers (trials).
        :param tuple global_args: arguments that will be passed to all trials.
        :return: output of function ``fun`` arranged in a list of blocks of size ``chunk_size``. If a TimeOut
                 occurred due to a SIGINT signal, then returns -1.
        :rtype: list
        """
        time1 = timeit.default_timer()
        # Manager that is able to communicate with the workers in the pool
        manager = SyncManager()  # using SyncManager() directly instead of letting mp.Manager() do it for us
        manager.start(self.mgr_init)  # start the child manager process
        self.__computed_blocks = manager.Value(int, 0)  # 'Share' this variable with all the workers
        self.__n_block = manager.Array('i', np.zeros(self.processes, dtype=int))
        # Start the pool and run the distributed simulation
        with self.poolcontext(processes=self.processes) as pool:
            self.__log.info('Distributing trials in %d pools:' % self.processes)
            if self.show_progress:
                self.print_table(self.processes)  # Print a table for monitoring the progress
            try:
                results = self.distribute_trials(pool, local_args, global_args)  # Distribute trials among workers
                while not results.ready() and self.show_progress:  # Monitor the progress
                    time.sleep(0.1)
                    for w in range(self.processes):
                        print_at("%10d" % (self.__n_block[w] * self.cs), 12 * w + 12, 2, cursor=False)
                    if self.__computed_blocks.value > self.processes:
                        progress = 100 * (self.__computed_blocks.value * self.cs) / self.n
                        print_at("T:   %3d%%" % progress, 1, 2, cursor=False)
                    else:
                        print_at("%4d" % (self.__computed_blocks.value * self.cs), 6, 2, cursor=False)
                results.get()  # Check if process has finished
            except KeyboardInterrupt:  # Listen to Interrupt signals (Ctrl + C)
                self.__log.warning("Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
            else:
                self.__log.info("Computation finished in {} s.".format(timeit.default_timer() - time1))
                pool.close()
            manager.shutdown()
            pool.join()

        try:
            return results.get(2)
        except mp.context.TimeoutError:
            return -1

    def distribute_trials(self, pool, local_args, glob_args):
        """ Distributes input data ``local_args`` among workers in ``pool``.

        :param mp.Pool pool: Pool of workers.
        :param tuple local_args: Local arguments to be distributed among workers.
        :param tuple glob_args: Global arguments to be passed to all workers.
        :return: a list with whatever ``fun`` returns.
        """
        # Check that local_args has been packed appropriately, i.e. as a tuple
        local_args = (local_args,) if not isinstance(local_args, (tuple, list)) else local_args
        # Build the iterable
        pool_task_args = [(self.fun, (k + 1),  # Pool specific variables
                           tuple([arg[k * self.cs:(k + 1) * self.cs] for arg in local_args]  # Task distributed vars
                                 + list(glob_args)))  # Task global variables
                          for k in range(self.n_b)]
        return pool.starmap_async(self.run_task_unpack, iterable=pool_task_args)

    def run_task_unpack(self, fun, block_n, task_args):
        """ Prepare the simulation function ``fun`` to be called by the worker. This function
        receives arguments which are specific of a worker (distribution has already been done).
        Prints the progress of each ``worker`` and pass the input arguments to the simulation
        function.

        :param function fun: simulation function.
        :param int block_n: the block number that is currently being processed by this worker.
        :param tuple task_args: task specific arguments arranged in a tuple that is expanded when calling the
                                simulation function ``fun``.
        :return: results of the simulation ``fun``.
        """

        # Variables to update progress monitoring
        if self.show_progress:
            cpr = mp.current_process()
            worker = self.worker(cpr, self.processes)
            self.__computed_blocks.value += 1
            self.__n_block[worker] += 1
        return fun(*task_args, block=block_n)

    @staticmethod
    @contextmanager
    def poolcontext(*args, **kwargs):
        """Modify the default signal handling behavior of :py:class:`multiprocessing.Pool` to ignore the
        ``SIGINT`` (Ctrl + C) signal.

        .. warning::

           In order to be able to stop the process it is necessary to catch the ``SIGINT`` signal, with a
           ``try``-``except`` clause for example.
        """
        # Setup parallel framework
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        pool = mp.Pool(*args, **kwargs)
        signal.signal(signal.SIGINT, original_sigint_handler)
        yield pool
        pool.terminate()

    @staticmethod
    def mgr_init():
        """Auxiliary function to modify the default behavior of :py:class:`multiprocessing.Manager` when
        receiving a ``SIGING`` (Ctrl + C) signal.
        """

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logging.debug('Initializing multiprocessing manager ...')

    @staticmethod
    def worker(cpr=None, processes=None, real=False):
        """Return the worker's id associated to the process ``cpr`` in a the range [1, processes].

        :param mp.Process or None cpr: Multiprocessing current process (equivalent to :py:class:`threading.Thread`).
        :param int or None processes: number of processes that are being run.
        :param bool real: whether to return the worker's complete id or not.
        :return: id of the worker.
        :rtype: int
        """
        if cpr is None:
            cpr = mp.current_process()
        if processes is None:
            processes = mp.cpu_count()
        wid = int(re.findall("\d+", cpr.name)[0])
        if real:
            return wid
        return ((wid % processes) + (processes - 1)) % processes

    @staticmethod
    def print_table(n_cols, row_titles=('Trial',), cfmt=r" Thread %2d", wcol0=12, wcoln=11):
        """ Prints a fancy table to stdout. It uses extended ASCII characters, therefore, not
        all terminals will display the table correctly.

        :param int n_cols: number of columns.
        :param tuple of str row_titles: Titles of the rows. Length of the provided iterable will determine the
                                        number of rows.
        :param str cfmt: format of the title of columns.
        :param int wcol0: width of first column, where the title of rows is printed.
        :param int wcoln: width of the rest of the columns.
        """
        top_ln = ("╔".rjust(wcol0) + ("═" * wcoln + "╤") * n_cols)[:-1] + "╗"
        header = ("║".rjust(wcol0) + "".join([(cfmt % (k + 1)).ljust(wcoln) + "│" for k in range(n_cols)]))[:-1] + "║"
        sup_ln = ("╔" + (wcol0 - 2) * "═" + "╬" + ("═" * wcoln + "╪") * n_cols)[:-1] + "╣"
        mid_ln = ("╟" + (wcol0 - 2) * "─" + "╫" + ("─" * wcoln + "┼") * n_cols)[:-1] + "╢"
        bot_ln = ("╚" + (wcol0 - 2) * "═" + "╩" + ("═" * wcoln + "╧") * n_cols)[:-1] + "╝"
        if not isinstance(row_titles, (list, tuple)):
            row_titles = [row_titles]
        rows = []
        for title in list(row_titles):
            rows.append(("║" + str(title).ljust(wcol0 - 2) + "║" + "│".rjust(wcoln + 1) * n_cols)[:-1] + '║')

        # Print table
        for element in [top_ln, header, sup_ln]:
            print(element)
        while rows:
            row = rows.pop()
            print(row)
            print(mid_ln) if rows else print(bot_ln)
