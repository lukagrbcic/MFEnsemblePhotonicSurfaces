import numpy as np
import joblib

import sys
import os

sys.path.insert(0, '../src')

from MFEnsembleFramework import InverseDesign


input_data = '../dataset/output_test_data.npy'
output_data = '../dataset/input_test_data.npy'
m = np.load(input_data).shape[0]

size = 8500
algorithm = 'PSO'
forward_model = f'../pretrained_models/forwardModel_{size}_RF.pkl'
pca_model = f'../pretrained_models/pca_50_{size}.pkl'
ensemble_path = f'../pretrained_models/inverseModel_{size}_RF_20.pkl'

n_evals = 25


dir_name = f'result_ensemble_sets'
os.mkdir(dir_name)


inverse_design = InverseDesign(input_data, output_data, 
                                forward_model, pca_model, 
                                evals=n_evals, top_rank=5, 
                                optimizer=algorithm, f_cut = 0.02, test_size=m)

_, prediction_sets, _ = inverse_design.run_RF_ensemble(ensemble_path)
         
prediction_sets_filtered = []

for s in range(len(prediction_sets[0])):
    temp = []
    for i in range(len(prediction_sets)):
        temp.append(prediction_sets[i][s])
    prediction_sets_filtered.append(temp)
    
    
for i in range(len(prediction_sets_filtered)):
    np.savetxt(f'{dir}/RF20_{i}_full_test_set.txt', prediction_sets_filtered[i])
