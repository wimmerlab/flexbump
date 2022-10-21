# Psychophysical data from the the category-level averaging task of Valentin Wyart

Behavioral data from three experiments using the cardinal/diagonal paradigm:
- dataset 1: Wyart et al., 2012, Neuron
- dataset 2: Cheadle, Wyart et al., 2014, Neuron - Exp. 2 (fMRI experiment)
- dataset 3: Wyart et al., 2015, Journal of Neuroscience - focused attention condition
- dataset 4: Cheadle, Wyart et al., 2014, Neuron - Exp. 1 (pupillometry experiment; continuous response)
- dataset 5: Wyart et al., 2015, Journal of Neuroscience - divided attention condition

![](doc/figs/cateogry-level_averaging_task.png)

![](doc/figs/decision_mapping_rule.png)

Note that the average sensory evidence in dataset 1 only takes 5 discrete values (strong/weak evidence for either cardinal or diagonal categories or zero evidence). In dataset 2 and 3 the avg. sensory evidence has a continuous distribution (no zero evidence trials).

Relevant fields in the data structure (n being the total number of trials) include:
data.sub  (n x 1) = the subject number
data.dv   (n x 8) =  the cardinal/diagonal evidence (from -1=diagonal to +1=cardinal)
                    for each sample of each trial (8 samples per trial)
data.resp (n x 1) = the subject response (+1=cardinal or -1:diagonal)
data.rt   (n x 1) = the subject response time (in secs)

The live script make_all.mlx generates all experimental data figures shown in the paper.
