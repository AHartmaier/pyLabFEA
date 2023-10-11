"""
Use trained ML yield function tos calculate the elastic-plastic behavior of the material in a strain-controlled loading.
Authors:  Ronak Shoghi, Alexander Hartmaier
ICAMS/Ruhr University Bochum, Germany
October 2023
"""

import pylabfea as FE
import numpy as np
import matplotlib.pyplot as plt

def construct_CV(C11, C12, C44):
    """
    Construct the elastic stiffness matrix in Voigt notation for a cubic crystal.

    Parameters:
    - C11, C12, C44: Material's elastic constants

    Returns:
    - CV: Elastic stiffness matrix
    """
    return np.array([
        [C11, C12, C12, 0, 0, 0],
        [C12, C11, C12, 0, 0, 0],
        [C12, C12, C11, 0, 0, 0],
        [0, 0, 0, C44, 0, 0],
        [0, 0, 0, 0, C44, 0],
        [0, 0, 0, 0, 0, C44]
    ])

# Import Data
db = FE.Data("Data_Base_Updated_Final_Rotated_Train.JSON", wh_data=True)
mat_ref = FE.Material(name="reference")  # define reference material, J2 plasticity, linear w.h.
mat_ref.elasticity(E=db.mat_data['E_av'], nu=db.mat_data['nu_av'])
mat_ref.plasticity(sy=db.mat_data['sy_av'], khard=4.5e3)
mat_ref.calc_properties(verb=False, eps=0.03, sigeps=True)

# db.plot_yield_locus(db =db, mat_data= db.mat_data, active ='flow_stress')
print(f'Successfully imported data for {db.mat_data["Nlc"]} load cases')
mat_ml = FE.Material(db.mat_data['Name'], num=1)  # define material
mat_ml.from_data(db.mat_data)  # data-based definition of material

# Train SVC with data from all microstructures
mat_ml.train_SVC(C=1, gamma=0.4, Fe=0.7, Ce=0.9, Nseq=1, gridsearch=False, plot=False)

# Define elastic stiffness tensor
CV = construct_CV(170000, 124000, 75000)
sig = np.zeros(6)
epl = np.zeros(6)
stresses = []
strains = [0]
seqq = [0]

# Strain increments to reach 3% strain in x-direction
total_strain = 0.01
n_increments = 50
deps_increment = np.array([total_strain/n_increments, 0, 0, 0, 0, 0])

for i in range(n_increments):
    # calculate the material response for the given strain increment
    fy1, sig, depl, grad_stiff = mat_ml.response(sig, epl, deps_increment, CV)
    epl += depl
    stresses.append(np.array(sig))
    strains.append(np.array(strains[-1]) + np.array(deps_increment[0]))

for item in stresses:
    seq = FE.sig_eq_j2(np.array(item))
    seqq.append(seq)

plt.scatter(strains, seqq)
plt.xlabel("Strain")
plt.ylabel("Stress")
plt.show()