#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script introduces an Active Learning method for SVM to enhance data selection
in training Machine Learning yield functions. Using the Query-By-Committee algorithm
we prioritize data points where model predictions show high disagreement,
leading to reduced variance in predictions.

Authors: Ronak Shoghi (1), Lukas Morand (2), Alexander Hartmaier (1)
1: [ICAMS/Ruhr University Bochum, Germany]
2: [Fraunhofer Institute for Mechanics of Materials IWM, Freiburg, Germany]
July 2023

Published as part of pyLabFEA package under GNU GPL v3 license
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import pylabfea as FE
from scipy.optimize import differential_evolution
from scipy.optimize import fsolve
from matplotlib.lines import Line2D


def creator_rnd(npoints, precision=8):
    """
    Generate random points in a 6D space, normalize them, and then round to the specified precision.
    Parameters
    ----------
    npoints : int
        Number of points to generate.
    precision : int, optional (default=8)
        Decimal precision to round the normalized points.
    Returns
    -------
    points : (N, 6) array
        Array of generated points with desired precision.
    """
    points = []
    for i in range(npoints):
        point = []
        while True:
            for j in range(6):
                value = np.random.uniform(-1, 1)
                point.append(value)
            norm = np.linalg.norm(point)
            if norm != 0:
                break
            else:
                point = []
        point_normalized = np.array(point) / norm
        point_rounded = np.around(point_normalized, decimals=precision)
        points.append(point_rounded)
    return np.vstack(points)


def eval_variance(angles, committee):
    """
    Evaluate the maximum disagreement among the committee based on the given angles.
    The function converts the angles to Cartesian coordinates and calculates the
    variance of the yield function outputs from the committee.
    Parameters
    ----------
    angles : (N,)-array
        List of spherical angles.

    committee : list of objects
        List of committee members where each member has a 'calc_yf' method
        to compute the yield function.
    Returns
    -------
    negative_variance : float
        Negative variance of the yield function outputs for maximization purpose.
    """
    # Convert from spherical to Cartesian coordinates
    x = FE.sig_spherical_to_cartesian(angles)
    y = np.zeros(len(committee))
    for i, member in enumerate(committee):
        y[i] = member.calc_yf(x * member.sy * 0.5)
    variance = np.var(y)
    return -variance


def comp_score(mat1, mat2, mat_ref, npoint=100, scale=10, offset=5):
    # analyze training result
    global nsamples_init, nsamples_to_generate
    loc = mat_ref.sy
    X1 = np.random.normal(loc=loc, scale=scale, size=int(npoint / 2))
    X2 = np.random.normal(loc=(loc - offset), scale=scale, size=int(npoint / 4))
    X3 = np.random.normal(loc=(loc + offset), scale=scale, size=int(npoint / 4))
    X = np.concatenate((X1, X2, X3))
    sunittest = FE.load_cases(number_3d=0, number_6d=len(X))
    sig_test = sunittest * X[:, None]
    yf_ml1 = mat1.calc_yf(sig_test)
    yf_ml2 = mat2.calc_yf(sig_test)
    yf_ref = mat_ref.calc_yf(sig_test)
    print(f'\n*** Training scores of active learning model with {nsamples_init} initial '
          f'and {nsamples_to_generate} training points:')
    FE.training_score(yf_ref, yf_ml1)
    print(f'\n*** Training scores of conventional learning model with {nsamples_init + nsamples_to_generate} '
          f'training points:')
    FE.training_score(yf_ref, yf_ml2)


def plot_variances(var_list):
    plt.plot(var_list, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Variance')
    plt.title('Variance vs Iteration')
    plt.grid()
    plt.savefig('variances_vs_iterations_weight=999.png', dpi=300)
    plt.show()


def plot_yield_locus(mat_ml, mat_h, niter, mat3=None):
    # plot yield locus
    ngrid = 50
    xx, yy = np.meshgrid(np.linspace(-1, 1, ngrid), np.linspace(0, 2, ngrid))
    yy *= mat_ml.scale_seq
    xx *= np.pi
    hh = np.c_[yy.ravel(), xx.ravel()]
    Z = mat_ml.calc_yf(FE.sig_cyl2princ(hh))  # value of yield function for every grid point
    Z2 = mat_h.calc_yf(FE.sig_cyl2princ(hh))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    contour = mat_ml.plot_data(Z, ax, xx, yy, c='black')
    legend_elements = [Line2D([0], [0], color='black', lw=2, label='ML active learning')]
    if mat3 is not None:
        Z3 = mat3.calc_yf(FE.sig_cyl2princ(hh))
        contour = mat3.plot_data(Z3, ax, xx, yy)
        legend_elements.append(Line2D([0], [0], color=contour.colors, lw=2, label='ML conventional'))
    contour = mat_h.plot_data(Z2, ax, xx, yy, c='blue')
    legend_elements.append(Line2D([0], [0], color='blue', lw=2, label='Reference'))

    ax.set_xlabel(r'$\theta$ (rad)', fontsize=22)
    ax.set_ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=22)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    plt.savefig('PLOTS_equiv_yield_stresses_iter_{}.png'.format(niter))
    plt.legend(handles=legend_elements)
    plt.show()
    # plt.close('all')


def read_vectors(file_name):
    # Load the vectors from the file
    vectors = np.loadtxt(file_name)
    return vectors


# Query by committee parameters
# max disagreement for yf-predictions,
# for classifiers generally possible: vote_entropy, consensus_entropy or
# maximum_disagreement, cf. https://modal-python.readthedocs.io/en/latest/content/query_strategies/Disagreement-sampling.html#disagreement-sampling

nmembers = 5  # Number of committee members
nsamples_init = 42  # Number of initial samples
nsamples_to_generate = 30  # Number of iterations
subset_percentage = 0.8  # Percent of data used for each committee member
init_rnd = False  # Train with random initial data points
file_init = None  # 'DATA_sunit_iter_80.txt'

# setup reference material with Hill-like anisotropy
sy = 50.
E = 200000.
nu = 0.3
hill = [1.4, 1.0, 0.7, 1.3, 0.8, 1.0]
mat_h = FE.Material(name='Hill-reference')
mat_h.elasticity(E=E, nu=nu)
mat_h.plasticity(sy=sy, hill=hill)

if isinstance(file_init, str):
    sunit = read_vectors(file_init)
elif init_rnd:
    # alternative for training with random initial data points
    sunit = creator_rnd(nsamples_init, 8)
else:
    # alternative for training with equally spaced initial data points
    c = int(nsamples_init / 3)
    d = nsamples_init - c
    sunit = FE.load_cases(number_3d=c, number_6d=d)
np.savetxt('Test_Cases.txt', sunit)

# create set of unit stresses
print('Created {0} unit stresses (6d Voigt tensor).'.format(nsamples_init))
x1 = fsolve(mat_h.find_yloc, np.ones(nsamples_init) * mat_h.sy, args=(sunit,), xtol=1.e-5)
sig = sunit * x1[:, None]
print('Calculated {} yield stresses.'.format(nsamples_init))

# train SVC with yield stress data generated from Hill flow rule
C = 3.0
gamma = 0.5
Ce = 0.99
Fe = 0.1
Nseq = 25
vlevel = 0
gsearch = True
cvals = [1, 2, 3, 4, 5]
gvals = [0.7, 1.0, 1.5, 2.0, 2.5]

mat_ml = FE.Material(name='ML-Hill')  # define material
mat_ml.train_SVC(C=C, gamma=gamma, Fe=Fe, Ce=Ce, Nseq=Nseq, sdata=sig, extend=False,
                 gridsearch=gsearch, cvals=cvals, gvals=gvals,
                 verbose=vlevel)
plot_yield_locus(mat_ml=mat_ml, mat_h=mat_h, niter=0)
np.savetxt('DATA_sig_iter_0.txt', sig)
np.savetxt('DATA_sunit_iter_0.txt', sunit)

var = []
hyp_C_list = []
hyp_g_list = []
bounds = [(0, np.pi)] + [(0, 2 * np.pi)] * 4
for i in range(nsamples_to_generate):
    # train SVC committee with yield stress data generated from Hill flow rule
    committee = []
    tstart = time.time()
    for j in range(nmembers):
        idx = np.random.choice(np.arange(sig.shape[0]),
                               int(sig.shape[0] * subset_percentage),
                               replace=False)
        mat_ml = FE.Material(name='ML-Hill_{}'.format(j))
        mat_ml.train_SVC(C=C, gamma=gamma, Fe=Fe, Ce=Ce, Nseq=Nseq,
                         sdata=sig[idx, :], extend=False,
                         gridsearch=gsearch, cvals=cvals, gvals=gvals,
                         verbose=vlevel)
        hyp_C_list.append(mat_ml.C_yf)
        hyp_g_list.append(mat_ml.gam_yf)
        committee.append(mat_ml)
    tend = time.time()
    print(f'***Iteration {i}:\n     Time for training committee: {tend - tstart}')

    # Search for next unit vector to query
    tstart = time.time()
    res = differential_evolution(eval_variance, bounds, args=(committee,),
                                 popsize=90, polish=True,
                                 updating='immediate')
    tend = time.time()
    print(f'     Time for differential evolution: {tend - tstart}')
    sunit_neww = res.x
    sunit_new = FE.sig_spherical_to_cartesian(sunit_neww)
    variance = res.fun
    var.append(-variance)

    # Calculate corresponding stress state and update data set
    x1 = fsolve(mat_h.find_yloc, mat_h.sy, args=(sunit_new,), xtol=1.e-5)
    sig_new = sunit_new * x1[:, None]
    sig = np.vstack([sig, sig_new])
    sunit = np.vstack([sunit, sunit_new])

# Train final model with all data sets
mat_ml = FE.Material(name='ML-Hill')  # define material
mat_ml.train_SVC(C=C, gamma=gamma, Fe=Fe, Ce=Ce, Nseq=Nseq,
                 sdata=sig, extend=False,
                 gridsearch=True, cvals=cvals, gvals=gvals,
                 verbose=vlevel)

# Create ML model with conventional training approach
Ntot = nsamples_init + nsamples_to_generate
c = int(Ntot / 3)
d = Ntot - c
sunit_r = FE.load_cases(number_3d=c, number_6d=d)
x1 = fsolve(mat_h.find_yloc, np.ones(Ntot) * mat_h.sy, args=(sunit_r,), xtol=1.e-5)
sig_r = sunit_r * x1[:, None]
mat_ml_r = FE.Material(name='ML-Hill')  # define material
mat_ml_r.train_SVC(C=C, gamma=gamma, Fe=Fe, Ce=Ce, Nseq=Nseq,
                   sdata=sig_r,
                   gridsearch=True, cvals=cvals, gvals=gvals, verbose=vlevel)

# Evaluate results
comp_score(mat_ml, mat_ml_r, mat_ref=mat_h, npoint=300)

# Plot results
plot_yield_locus(mat_ml, mat_h, nsamples_to_generate, mat3=mat_ml_r)
plot_variances(var)

if gsearch:
    fig = plt.figure()
    plt.plot(hyp_g_list, 'b.', label='gamma')
    plt.plot(hyp_C_list, 'r.', label='C')
    plt.legend()
    plt.title('Evolution of hyperparameters')
    plt.xlabel('iteration * committee')
    plt.ylabel('C, gamma')
    plt.show()

# Save data files
np.savetxt('DATA_sig_iter_{}.txt'.format(nsamples_to_generate), sig)
np.savetxt('DATA_sunit_iter_{}.txt'.format(nsamples_to_generate), sunit)
np.savetxt('variance.txt', var)
