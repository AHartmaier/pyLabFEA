"""
This file shows the steps to train the texture-dependent SVC.
The main differences to the texture-independent approach are:
- create multiple pylabFEA Data objects each from a different Data_Base.json
- understand data generated with the CYL approach as it just contains the yield onset and no SE-curve
- assign different Data objects to one pylabFEA Material object
- normalize SE data over different data sets
- include standardization objects into the material.SVC object

Author: Jan Schmidt
Email: jan.schmidt-p2d@rub.de
Date: 05.11.2024
"""

import numpy as np
import glob
import os

try: 
    from sklearnex import patch_sklearn
    patch_sklearn()
except ModuleNotFoundError:
    pass
import pylabfea as FE
import csv
from joblib import Parallel, delayed
import logging
import time
from scipy.optimize import fsolve
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 0) Set variables
path_db = "Data_CPFFT"  # JS: Steven has all data at "~/Desktop/scratch/KDEApproach5deg/KDEApproach5deg/"  #
texture_key_file = "Data_CPFFT/texture_keys_success.csv"
wh_data = False
testing = True
n_cpu = 12
verbose = 0
plot = True
plot_cyl_only = True

if verbose == 2:
    logger = logging.getLogger('sklearnex')
    logger.setLevel(logging.INFO)

def read_data(tx_key):
    """
    Creates an FE.Data object. this function is a helper function iin order to speed up things. For a few textures,
    we can follow the standard protocols and initialize the objects serial. However for > 7000 texturse that takes more
    than two hours.

    Parameters
    ----------
    tx_key : list
        Texture Keys to check

    Returns
    -------
    db_dict : dict Object
        Dictionary containing the FE.Data Object as values for each tx_key.
    """
    db_dict = {}
    for res_dir in res_dirs_list:
        # if any(tx_key in res_dir for tx_key in tx_keys):
        if tx_key in res_dir:
            try:
                db = FE.Data("Data_Base.json", path_data=res_dir, wh_data=wh_data, mode='JS', tx_data=True)
                db_dict[db.mat_data['tx_key']] = db
            except FileNotFoundError:
                print(f"{res_dir} contains no Data_Base.json")

    return db_dict

# 1) Import Data from Micromechanical Simulations
res_dirs_list = glob.glob(os.path.join(path_db, "*"))

# 1.1) Define a list of textures that should be used for training
with open(texture_key_file) as f:
    texture_keys = f.read().splitlines()

# 1.2) Create FE Data objects in a loop
start = time.time()
db_dict = Parallel(n_jobs=n_cpu, backend="loky")(delayed(read_data)(tx_key) for tx_key in texture_keys)
end = time.time()
print(f'Took me {end-start} s with {n_cpu} CPUs.')
db_dict_full = {}
for d in db_dict:
    db_dict_full.update(d)

# 2) Create Material from the list of DB
db_list = list(db_dict_full.values())
mat_ml = FE.Material(db_list[0].mat_data['tx_name'], num=1)
mat_ml.from_data([data_obj.mat_data for data_obj in db_list[:-1]])  # JS: take first texture for testing

# 3) Train SVC
train_sc, test_sc = mat_ml.train_SVC(C=10, gamma=1, Fe=0.8, Ce=0.95, Nseq=2, gridsearch=False, plot=False)

# 4) Testing
if testing:
    # 3.2) Create test material
    mat_test = FE.Material("Testium", num=1)
    mat_test.from_data(db_list[-1].mat_data)
    x_test_list = []
    y_test_list = []

    # 3.3) Create test arrays
    Nlc, N0, x_train, y_train = mat_test._create_data_for_ms(Ce=0.95, Fe=0.8, Nseq=2, idx_ms=0, extend=False)

    x_test_list.append(x_train)
    y_test_list.append(y_train)
    x_test = np.concatenate(x_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # 3.4) Transform feature vector
    x_test = mat_ml.transform_input(x_test)

    # 3.5) Evaluate Metrics
    test_sc = mat_ml.svm_yf.score(x_test, y_test)

# 4) Plot Yield Loci
fig = plt.figure(figsize=(12, 9))
fig.set_constrained_layout(True)
ax = fig.add_subplot(projection='polar')
colors = plt.cm.viridis(np.linspace(0, 1, len(db_list)))
if plot:
    for idx_ms, ms in enumerate(mat_ml.msparam):
        tex = ms['texture']
        syld = ms['sig_ideal']
        if plot_cyl_only:
            syld_cyl = []
            for stress in syld:
                if all(np.abs(comp) < 0.01 for comp in stress[-3:]):
                    syld_cyl.append(stress)
            syld = np.array(syld_cyl)
        syld_true = FE.sig_eq_j2(syld)
        syld = FE.sig_princ2cyl(syld)

        # Define stress directions in which zeros of yield function are determined
        theta = np.linspace(-np.pi, np.pi, 72)
        snorm = FE.sig_cyl2princ(np.array([np.ones(72), theta]).T)
        snorm = np.hstack((snorm, np.zeros_like(snorm)))

        # Find zeros of yield function
        x1 = fsolve(mat_ml.find_yloc, 0.8 * np.mean(syld_true) * np.ones(len(snorm)),
                    args=(snorm, None, 0., 0., 0., tex), xtol=1.e-5)
        sig_ml = snorm * x1[:, None]
        syld_ml = FE.sig_princ2cyl(sig_ml)

        # Order CPFFT data according to angle on pi-plane
        theta = syld[:, 1]
        for idx_ang, ang in enumerate(theta):
            if ang < 0.:
                theta[idx_ang] = ang + 2 * np.pi
        idx_theta = np.argsort(theta)

        # Plot CPFFT data
        ax.scatter(syld[:, 1][idx_theta], syld[:, 0][idx_theta], # / np.mean(syld_cyl[:, 0]),
                label=f"{ms['tx_key']} - CPFFT", color=colors[idx_ms])

        # Plot zeros of yield function
        ax.plot(syld_ml[:, 1], syld_ml[:, 0], #/ np.mean(syld_cyl_ml[:, 0]),
                color=colors[idx_ms], label=f"{ms['tx_key']} - ML YS", linestyle='--')

    if testing:
        for idx_ms, ms in enumerate(mat_test.msparam):
            tex = ms['texture']
            syld = ms['sig_ideal']
            if plot_cyl_only:
                syld_cyl = []
                for stress in syld:
                    if all(np.abs(comp) < 0.01 for comp in stress[-3:]):
                        syld_cyl.append(stress)
                syld = np.array(syld_cyl)
            syld_true = FE.sig_eq_j2(syld)
            syld = FE.sig_princ2cyl(syld)

            # Define stress directions in which zeros of yield function are determined
            theta = np.linspace(-np.pi, np.pi, 72)
            snorm = FE.sig_cyl2princ(np.array([np.ones(72), theta]).T)
            snorm = np.hstack((snorm, np.zeros_like(snorm)))

            # Find zeros of yield function
            x1 = fsolve(mat_ml.find_yloc, 0.8 * np.mean(syld_true) * np.ones(len(snorm)),
                        args=(snorm, None, 0., 0., 0., tex), xtol=1.e-5)
            sig_ml = snorm * x1[:, None]
            syld_ml = FE.sig_princ2cyl(sig_ml)

            # Order CPFFT data according to angle on pi-plane
            theta = syld[:, 1]
            for idx_ang, ang in enumerate(theta):
                if ang < 0.:
                    theta[idx_ang] = ang + 2 * np.pi
            idx_theta = np.argsort(theta)

            # Plot CPFFT data
            ax.scatter(syld[:, 1][idx_theta], syld[:, 0][idx_theta],  # / np.mean(syld_cyl[:, 0]),
                       label=f"{ms['tx_key']} - CPFFT - Holdout Texture", color=colors[idx_ms+3])

            # Plot zeros of yield function
            ax.plot(syld_ml[:, 1], syld_ml[:, 0],  # / np.mean(syld_cyl_ml[:, 0]),
                    color=colors[idx_ms+3], label=f"{ms['tx_key']} - ML YS - Holdout Texture", linestyle='--')

    dr = np.mean(syld[:, 0])  # / np.mean(syld_cyl[:, 0]))
    drh = 0.08 * dr
    ax.arrow(0, 0, 0, dr, head_width=0.05, width=0.004,
             head_length=drh, color='r', length_includes_head=True)
    ax.text(-0.12, dr * 0.89, '$\sigma_1$', color='r', fontsize=12)
    ax.arrow(2.0944, 0, 0, dr, head_width=0.05,
             width=0.004, head_length=drh, color='r', length_includes_head=True)
    ax.text(2.24, dr * 0.94, '$\sigma_2$', color='r', fontsize=12)
    ax.arrow(-2.0944, 0, 0, dr, head_width=0.05,
             width=0.004, head_length=drh, color='r', length_includes_head=True)
    ax.text(-2.04, dr * 0.97, '$\sigma_3$', color='r', fontsize=12)
    ax.set_title(f'Yield Loci on $\pi$-plane')
    ax.legend(loc=(.53, 0.85), ncols=2)  # [plt_cut, plt_rem], ['cut-off textures', 'remaining textures'],
    plt.show()
