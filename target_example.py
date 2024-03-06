import numpy as np
import joblib
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

sys.path.insert(0, './src')

from MFEnsembleFramework import InverseDesign


plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})



input_data = './dataset/output_test_data.npy'
output_data = './dataset/input_test_data.npy'
m = np.load(input_data).shape[0]

size = 5000
algorithm = 'DE'
forward_model = f'./pretrained_models/forwardModel_{size}_RF.pkl'
pca_model = f'./pretrained_models/pca_50_{size}.pkl'
ensemble_path = f'./pretrained_models/inverseModel_{size}_RF_20.pkl'

n_evals = 100


w = np.loadtxt('wavelength_plot.txt')
target = '100emissivity'


if target =='step':
    f_ = np.where(w < 4.6, 1, 0).reshape(1, -1)
    
elif target =='100emissivity':
    f_ = np.ones((1, 822))*1

inverse_design = InverseDesign(input_data, output_data, 
                                forward_model, pca_model, 
                                evals=n_evals, top_rank=10, 
                                optimizer=algorithm, f_cut = 0.02, test_size=m)


predictions, prediction_sets = inverse_design.inference_RF(ensemble_path, f_)

prediction_sets = np.array(prediction_sets[0])
prediction_sets_ = np.unique(prediction_sets, axis=0)


model = joblib.load(forward_model)
pca = joblib.load(pca_model)
plt.figure(figsize=(6,5))
plt.plot(w, f_[0], 'k', label='Target')
predicted_emissivities = []
for pred in prediction_sets_:
    em = model.predict([pred])
    em = pca.inverse_transform(em)
    predicted_emissivities.append(em[0])
    
    

for e in predicted_emissivities:
    plt.plot(w, e, 'g-', alpha=0.3)
    
plt.ylim(-0.05, 1.05)
plt.xlim(2.5, 12)
plt.xlabel(r'Wavelength ($\upmu$m)')
plt.ylabel('Emissivity')

patch1 = mpatches.Patch(color='black',  label='Target')
patch2 = mpatches.Patch(color='green',  label='Predicted')


ax = plt.gca()
ax.legend(handles=[patch1, patch2])

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)