# Multi-fidelity ensemble framework for inverse design of photonic surfaces
Code and data for the manuscript "Inverse design of photonic surfaces on Inconel via multi-fidelity ensemble framework and high throughput femtosecond laser processing" by Grbcic, L. et al., 2024.

The dataset (in .npy format) used to train the models and for inverse design is labeled as:

**input_train_data.npy** - the input train data that contains 8,500 laser processing parameters (power, speed, spacing) (3D) 
**output_train_data.npy** - the output train data that contains 8,500 emissivity curves (822D)

**input_test_data.npy** - the input test data that contains 3,259 laser processing parameters (power, speed, spacing) (3D) 
**output_test_data.npy** - the output test data that contains 3,259 emissivity curves (822D)

The data was previously shuffled and split so these four train and test data files can be used in the same way as it is described in the manuscript.
