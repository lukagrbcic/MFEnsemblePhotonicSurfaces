import numpy as np
import joblib

import sys
import os

sys.path.insert(0, '../src')

from MFEnsembleFramework import InverseDesign



input_data = '../dataset/output_test_data.npy'
output_data = '../dataset/input_test_data.npy'
m = np.load(input_data).shape[0]

size = 5000
algorithm = 'PSO'
forward_model = f'../pretrained_models/forwardModel_{size}_RF.pkl'
pca_model = f'../pretrained_models/pca_50_{size}.pkl'
ensemble_path = f'../pretrained_models/inverseModel_{size}_RF_20.pkl'

n_evals = 50


dir_name = f'results_test_set'
os.mkdir(dir_name)

inverse_design = InverseDesign(input_data, output_data, 
                                forward_model, pca_model, 
                                evals=n_evals, top_rank=5,
                                optimizer=algorithm, f_cut = 0.02, test_size=m)

predictions, _, _ = inverse_design.run()
np.savetxt(f'results_test_set/HF_predictions_{n_evals}_evals_{algorithm}_test_set.txt', predictions)
        
