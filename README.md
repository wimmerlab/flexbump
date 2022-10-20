# Flexible integration of continuous sensory evidence in perceptual estimation tasks
Code of the computational models and psychophysical data of the paper "Flexible integration of continuous sensory evidence in perceptual estimation tasks"

## Requirements

The scripts have been tested in Python 3.9 running on Ubuntu 22.04 and Debian 11.

Create a python environment and install the `requirements.txt`

```shell
pip install -r requirements.txt
```

## Running the neural network simulations

All the necessary code to generate the simulated data is at `./src`. In order to generate the theoretical data use 
the script `ring_simulation.py` from `./src`:
```shell
cd ./src
python ring_simulation.py -W ignore [-options_flags options]
```
The configuration files are at `./src/conf/`. You would normally run the script in the following way:
```shell
python ring_simulation.py -W ignore -f conf/conf.txt [-option_flags options]
```
Once the `conf.txt` file is loaded, the script will load the options listed in the file, which can be then
modified via the command line optional arguments `[-option_flags options]`

You can see a list of the possible options by running the following command:
```shell
python ring_simulation.py -W ignore -f conf/conf.txt -h
```

## Generating the figures

In order to generate the figures of the paper (or a minimal version of them), one should run `figs_paper.py` 
with the options indicated at the bottom of the script. For example, to generate the graphs of Figure 3, execute
the script as follows:
```shell
python figs_paper.py -figsize 4.2 3.6  -eslim 90 -i0s 0.02 0.05 0.08
```
Some of the paths that point to the data should be changed before running this script. In addition, some
of the data is not included and should be generated first by running simulations.

