import numpy as np
import joblib

import sys
import os

sys.path.insert(0, '../src')

from MFEnsembleFramework import InverseDesign



input_data = '../dataset/output_test_data.npy'
output_data = '../dataset/input_test_data.npy'

size = 5000
algorithm = 'PSO'
forward_model = f'../pretrained_models/forwardModel_{size}_RF.pkl'
pca_model = f'../pretrained_models/pca_50_{size}.pkl'
ensemble_path = f'../pretrained_models/inverseModel_{size}_RF_20.pkl'

n_evals = [25, 50, 100]
n_repeats = 5

dir_name = f'results_HF_{size}_{algorithm}'
os.mkdir(dir_name)



for evals in n_evals:
    print ('Evals:', evals)
    rmse_ = []
    for runs in range(n_repeats):

        inverse_design = InverseDesign(input_data, output_data, 
                                       forward_model, pca_model, 
                                       evals=evals, top_rank=5, 
                                       optimizer='DE', f_cut = None, test_size=50)
        
        predictions, _, time = inverse_design.run()
        print ('Saving results')
        np.savetxt(f'./{dir_name}/Vanilla{runs}_predictions_{evals}.txt', predictions)
        
        
        
        
        
