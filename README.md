# Multi-fidelity ensemble framework for inverse design of photonic surfaces
Code and data for the manuscript "Inverse design of photonic surfaces on Inconel via multi-fidelity ensemble framework and high throughput femtosecond laser processing" by Luka Grbcic (LBNL), Minok Park (LBNL), Mahmoud Elzouka (LBNL), Ravi Prasher (LBNL, UCB), Juliane Müller (NREL), Costas P. Grigoropoulos (UCB), Sean D. Lubner (LBNL, BU), Vassilia Zorba (LBNL, UCB), and Wibe Albert de Jong (LBNL), 2024.

LBNL - Lawrence Berkeley National Laboratory

UCB - University of California, Berkeley

BU - Boston University

NREL - National Renewable Energy Laboratory

Models and data can be downloaded from: https://osf.io/dwgtf/

The two folders should be put in the same directory as the rest of the code.
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
**MODELS DESCRIPTION**
_______
The **pretrained_models** folder contains (in .pkl format) both the forward and inverse-trained RF scikit-learn models as well as the PCA compression models.

The forward models trained with varied train dataset sizes (2,500, 5,000, 8,500):

**forwardModel_2500_RF.pkl**

**forwardModel_5000_RF.pkl**

**forwardModel_8500_RF.pkl**

The inverse models (N=20 estimators) trained with varied train dataset sizes (2,500, 5,000, 8,500):

**inverseModel_2500_RF_20.pkl**

**inverseModel_5000_RF_20.pkl**

**inverseModel_8500_RF_20.pkl**

The PCA models (50 principal components) trained with varied train dataset sizes (2,500, 5,000, 8,500):

**pca_50_2500.pkl**

**pca_50_5000.pkl**

**pca_50_8500.pkl**

Models with the same dataset sizes should be used together (i.e. forwardModel_2500_RF with inverseModel_2500_RF and pca_50_2500)
_______
**CODE DESCRIPTION**
_______

The src folder contains the main code in the file **MFEnsembleFramework.py**.

The HF_LF_MF_scripts folder contains the scripts that can be used to reproduce the HF vs LF vs MF models comparison in Figure 3 of the manuscript.

The evaluation_scripts folder contains the scripts that can be used to reproduce the HF vs MF models in Figure 3, 10 runs per every maximum evaluation number.

The script in MF_5sets_predictions_scripts shows how to obtain the top 5 prediction sets for a given test set size.

The **target_example.py** script shows how to do the MF ensemble inference for a specific spectral emissivity target.

Finally, an example on how to run the framework to generate predictions for the whole test_set:

```python

#load all modules
import numpy as np
import joblib

import sys
import os

sys.path.insert(0, '../src')

#inverseDesign is the main class we use to create an object
from MFEnsembleFramework import InverseDesign


#load the input_data which are the spectral emissivity values (inverse relationship is what we are after),
#this is the test set and we are going to do inference for each instance
input_data = '../dataset/output_test_data.npy'

#this will be removed but we load the output_data as a placeholder
#(this is from a previous version when postprocessing was a part of this class)
output_data = '../dataset/input_test_data.npy'

m = np.load(input_data).shape[0] #get the size of the whole test set

size = 5000 #define the size of the pretrained model (5000 tells us it will use the forward, inverse and pca models trained with 5000 dataset
algorithm = 'PSO' #choose the algorithm (other option is DE)
forward_model = f'../pretrained_models/forwardModel_{size}_RF.pkl' #location of the forward model
pca_model = f'../pretrained_models/pca_50_{size}.pkl' #location of the pca model
ensemble_path = f'../pretrained_models/inverseModel_{size}_RF_20.pkl' #location of the inverse model

n_evals = 25 #maximum number of evaluations we want to use in the HF optimization cycle

dir_name = f'results_test_set' #create the result folder
os.mkdir(dir_name)

#definition of the inverse_design object where we define all the previous variables as arguments,
#the top_rank variable tells us the amount of top predicted values we want to save/consider
#the f_cut value is set to define the termination criterion (i.e. if the value yields <2% error stop optimizing)
inverse_design = InverseDesign(input_data, output_data, 
                                forward_model, pca_model, 
                                evals=n_evals, top_rank=5, 
                                optimizer=algorithm, f_cut = 0.02, test_size=m)


#we get the best predictions and prediction sets if we set multiple top_rank values
#we call the run_RF_ensemble method to do inverse desgin with the LF inverse model
predictions, prediction_sets, _ = inverse_design.run_RF_ensemble(ensemble_path) #we need to define the ensemble path here (inverse model)
```











