import pylabfea as FE
import numpy as np
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt


def calculate_stress_from_strain(strain, C):
    """Calculates the predicted stress for a given strain using the material's stiffness
        matrix. This function multiplies the stiffness matrix C by the strain vector to
        predict the resulting stress, according to Hooke's Law for linear elastic materials.

        Parameters
        ----------
        strain: list or np.ndarray
            The strain vector, representing the deformation in the material. This should be a
            1D array or list of length 6, corresponding to the three normal and three shear
            components of strain.
        C: np.ndarray
            The stiffness matrix of the material. This should be a 6x6 symmetric
            matrix representing the material's resistance to deformation. The matrix correlates
            the strain to the stress in the material.

        Returns
        -------
        _: np.ndarray
            The predicted stress vector, calculated as the dot product of the stiffness
            matrix C and the strain vector. This vector has the same dimension as the strain
            input, containing the three normal and three shear components of stress.
    """
    strain_vector = np.array(strain)
    stress_predicted = np.dot(C, strain_vector)
    return stress_predicted


# main code block
db = FE.Data("Data_Random_Texture.json",
             epl_crit=2.e-3, epl_start=1.e-3, epl_max=0.03,
             wh_data=True)
stress = db.mat_data['elstress']
strain = db.mat_data['elstrain']
assert len(stress) == len(strain), "Stress and strain data must have the same length"
data_pairs = list(zip(strain, stress))

C = db.mat_data['elast_const']

# Visualize the results. Using the calculated C matrix to predict stresses from strains for 10% of the data points.
predicted_stresses = np.array([calculate_stress_from_strain(s, C) for s in strain])
stresses = np.array(stress)
predicted_stresses = np.array(predicted_stresses)
num_data_points = stresses.shape[0]
subset_size = int(num_data_points * 0.1)
np.random.seed(42)
selected_indices = np.random.choice(num_data_points, subset_size, replace=False)
selected_actual_stresses = stresses[selected_indices]
selected_predicted_stresses = predicted_stresses[selected_indices]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
component_names = ['Stress_11', 'Stress_22', 'Stress_33', 'Stress_12', 'Stress_13', 'Stress_23']
for i, ax in enumerate(axes):
    ax.scatter(selected_actual_stresses[:, i], selected_predicted_stresses[:, i], alpha=0.5, label='Data Points')
    max_stress = max(selected_actual_stresses[:, i].max(), selected_predicted_stresses[:, i].max())
    min_stress = min(selected_actual_stresses[:, i].min(), selected_predicted_stresses[:, i].min())
    ax.plot([min_stress, max_stress], [min_stress, max_stress], 'k--', label='Perfect Fit')
    ax.set_xlabel(f'Actual {component_names[i]} Stress')
    ax.set_ylabel(f'Predicted {component_names[i]} Stress')
    ax.set_title(f'Component: {component_names[i]}')
    ax.grid(True)
    ax.legend()
plt.tight_layout()
plt.show()
