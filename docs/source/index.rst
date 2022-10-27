.. Flexible integration of continuous sensory evidence in perceptual estimation tasks documentation master file, created by
   sphinx-quickstart on Thu Oct 20 15:34:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Flexible integration of continuous sensory evidence in perceptual estimation tasks
==================================================================================

**Abstract**

.. raw:: html

   <div class="sidebar-box">
      <p>Temporal accumulation of evidence is crucial for making accurate judgements
         based on noisy or ambiguous sensory input. The integration process leading to
         categorical decisions is thought to rely on competition between neural
         populations, each encoding a discrete categorical choice. How recurrent neural
         circuits integrate evidence for continuous perceptual judgments is unknown.
         Here, we show that a continuous bump attractor network can integrate a circular
         feature such as stimulus direction
         nearly optimally.
         As required by optimal integration, the population activity of the network
         unfolds on a two-dimensional manifold, in which the position of the network’s
         activity bump tracks the stimulus average and, simultaneously, the bump amplitude
         tracks stimulus uncertainty. Moreover, the temporal weighting of sensory
         evidence by the network depends on the relative strength of the stimulus compared to the
         internally generated bump dynamics,
         yielding either early (primacy), uniform or late (recency) weighting. The model
         can flexibly switch between these regimes by changing a single control parameter,
         the global excitatory drive. We show that this mechanism can quantitatively
         explain individual temporal weighting profiles of human observers, and we validate
         the model prediction that temporal weighting impacts reaction times. Our findings
         point to continuous attractor dynamics as a plausible neural mechanism underlying
         stimulus integration in perceptual estimation tasks.</p>
     <p class="read-more"><a href="#" class="sphinx-bs btn text-wrap btn-secondary stretched-link reference internal">Read More</a></p>
   </div>
   
**This repository**

Here you will find the code and the data that supports our work. In order to get a general idea on how to
simulate the ring model navigate towards the :ref:`howto` page. A more detailed documentation is given in the
:ref:`user` or directly at :ref:`code`. Finally, the documentation of the experimental
data is located at :ref:`data`.

.. grid:: 2

    .. grid-item-card::
       :img-top: ./_static/index-images/getting_started.svg

       Getting started
       ^^^^^^^^^^^^^^^

       How to use the scripts, generate the figures, etc.

       +++

       .. button-ref:: howto/index
           :click-parent:
           :expand:
           :color: secondary

           Start here


    .. grid-item-card::
       :img-top: ./_static/index-images/user_guide.svg

       User guide
       ^^^^^^^^^^

       Detailed overview of the simulation framework.

       +++

       .. button-ref:: user_guide/index
           :click-parent:
           :expand:
           :color: secondary

           To the user guide

    .. grid-item-card::
       :img-top: ./_static/index-images/data.svg

       Data
       ^^^^

       Sources of data.

       Extraction of data.

       +++

       .. button-ref:: data/index
           :click-parent:
           :expand:
           :color: secondary

           Data documentation

    .. grid-item-card::
       :img-top: ./_static/index-images/api.svg

       API reference
       ^^^^^^^^^^^^^

       The reference guide contains a description of the functions, modules and objects
       included in these repository.

       +++

       .. button-ref:: code/index
           :click-parent:
           :expand:
           :color: secondary

           To the api guide


.. toctree::
   :maxdepth: 3

   howto/index
   user_guide/index
   data/index
   code/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
