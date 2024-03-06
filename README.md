# Multi-fidelity ensemble framework for inverse design of photonic surfaces
Code and data for the manuscript "Inverse design of photonic surfaces on Inconel via multi-fidelity ensemble framework and high throughput femtosecond laser processing" by Luka Grbcic (LBNL),Minok Park (LBNL), Mahmoud Elzouka (LBNL), Ravi Prasher (LBNL, UCB), Juliane Müller (NREL), Costas P. Grigoropoulos (UCB), Sean D. Lubner (LBNL, BU), Vassilia Zorba (LBNL, UCB), and Wibe Albert de Jong (LBNL), 2024.

LBNL - Lawrence Berkeley National Laboratory

UCB - University of California, Berkeley

BU - Boston University

NREL - National Renewable Energy Laboratory


_______
**DATA DESCRIPTION**
_______
The dataset (in .npy format) used to train the models and for inverse design is labeled as:

**input_train_data.npy** - the input train data that contains 8,500 laser processing parameters (power, speed, spacing) (3D) 
**output_train_data.npy** - the output train data that contains 8,500 emissivity curves (822D)

**input_test_data.npy** - the input test data that contains 3,259 laser processing parameters (power, speed, spacing) (3D) 
**output_test_data.npy** - the output test data that contains 3,259 emissivity curves (822D)

The data was previously shuffled and split so these four train and test data files can be used in the same way as it is described in the manuscript.
For the inverse models, the inputs and outputs are switched.

_______
**CODE DESCRIPTION**
_______



_______
**MODELS DESCRIPTION**
_______
