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

# import sys
# import os
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylabfea as FE
from scipy.optimize import differential_evolution
from scipy.optimize import fsolve
from matplotlib.lines import Line2D


# matplotlib.use('Agg')  # required???
# sys.path.append('src/data-gen')  # required???
# sys.path.append('src/verify')


def spherical_to_cartesian(angles):
    """
    Convert a list of 5 spherical angles to Cartesian coordinates.
    Parameters
    ----------
    angles : (N,)-array
        List of 5 angles in radians.
    Returns
    -------
    coordinates : (N,6) array
        Cartesian coordinates computed from the input spherical angles.
    """
    assert len(angles) == 5
    x1 = np.cos(angles[0])
    x2 = np.sin(angles[0]) * np.cos(angles[1])
    x3 = np.sin(angles[0]) * np.sin(angles[1]) * np.cos(angles[2])
    x4 = np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.cos(angles[3])
    x5 = np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(angles[3]) * np.cos(angles[4])
    x6 = np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(angles[3]) * np.sin(angles[4])
    return np.array([x1, x2, x3, x4, x5, x6])


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


def find_yloc(x, sig, mat):
    """
    Function to expand unit stresses by factor and calculate yield function;
    used by search algorithm to find zeros of yield function.
    Parameters
    ----------
    x : (N,)-array
        Multiplyer for stress
    sig : (N,6) array
        unit stress
    Returns
    -------
    f : 1d-array
        Yield function evaluated at sig=x.sp
    """
    f = mat.calc_yf(sig * x[:, None])
    return f


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
    x = spherical_to_cartesian(angles)
    y = np.zeros(len(committee))
    for i, member in enumerate(committee):
        y[i] = member.calc_yf(x)
    variance = np.sum(np.square(y - np.mean(y) * np.ones_like(y)))  # /len(committee) ???
    return -variance


def plot_variances(var_list):
    plt.plot(var_list, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Variance')
    plt.title('Variance vs Iteration')
    plt.grid()
    plt.savefig('variances_vs_iterations_weight=999.png', dpi=300)
    plt.show()
    # plt.close()


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


''' Unused subroutines
def read_vectors(file_name):
    # Load the vectors from the file
    vectors = np.loadtxt(file_name)
    return vectors


def save_hard_test_cases(a, b, num_tests=5):
    test_arrays = [[] for _ in range(num_tests)]
    for i in range(num_tests):
        sig_test = hard_test_cases(a, b)
        np.savetxt(f'sig_test_{i + 1}.txt', sig_test)
        test_arrays[i] = sig_test
    return test_arrays


def hard_test_cases(a, b):
    """
    Generate hard test cases by creating random unit vectors, scaling them based on the
    solutions from the function 'find_yloc', and then concatenating the results.
    Parameters
    ----------
    npoints : int
        Number of points to generate.
    Returns
    -------
    sig : (2*N, 6) array
        Array of concatenated test cases.
    """
    # Create random unit vectors and scale them by 0.99 times the solution from 'find_yloc'
    sunit1 = FE.load_cases(number_3d=a, number_6d=b)
    npoints = (a + b)
    x1 = fsolve(find_yloc, np.ones(npoints) * mat_h.sy, args=(sunit1, mat_h), xtol=1.e-5)
    sig1 = sunit1 * 0.99 * x1[:, None]
    # Create another set of random unit vectors and scale them by 1.01 times the solution from 'find_yloc'
    sunit2 = FE.load_cases(number_3d=a, number_6d=b)
    x2 = fsolve(find_yloc, np.ones(npoints) * mat_h.sy, args=(sunit2, mat_h), xtol=1.e-5)
    sig2 = sunit2 * 1.01 * x2[:, None]
    sig = np.concatenate((sig1, sig2))
    print(sig)
    return sig
    '''

# Query by committee parameters
# max disagreement for yf-predictions,
# for classifiers generally possible: vote_entropy, consensus_entropy or
# maximum_disagreement, cf. https://modal-python.readthedocs.io/en/latest/content/query_strategies/Disagreement-sampling.html#disagreement-sampling

nmembers = 5  # Number of committee members
nsamples_init = 100  # Number of initial samples
nsamples_to_generate = 200  # Number of iterations, std. 30
subset_percentage = 0.8

# setup reference material with Hill-like anisotropy
# path = os.path.dirname(__file__)
sy = 50.
E = 200000.
nu = 0.3
hill = [1.4, 1.0, 0.7, 1.3, 0.8, 1.0]
mat_h = FE.Material(name='Hill-reference')
mat_h.elasticity(E=E, nu=nu)
mat_h.plasticity(sy=sy, hill=hill)
# mat_h.calc_properties(eps=0.0013, sigeps=True)
# c = int(nsamples_init*2/3)
# d = nsamples_init - c
# sunit = FE.load_cases(number_3d=c, number_6d=d)
sunit = creator_rnd(nsamples_init, 8)
np.savetxt('Test_Cases.txt', sunit)
# create set of unit stresses and
print('Created {0} unit stresses (6d Voigt tensor).'.format(nsamples_init))
x1 = fsolve(find_yloc, np.ones(nsamples_init) * mat_h.sy, args=(sunit, mat_h), xtol=1.e-5)
sig = sunit * x1[:, None]
print('Calculated {} yield stresses.'.format(nsamples_init))
# train SVC with yield stress data generated from Hill flow rules
C = 6.0
gamma = 2.0
Fe = 0.5
Ce = 0.99
Nseq = 20
vlevel = 0
cvals = [5., 6., 8.]
gvals = [1., 2., 3.]

mat_ml = FE.Material(name='ML-Hill')  # define material
mat_ml.train_SVC(C=C, gamma=gamma, Fe=Fe, Ce=Ce, Nseq=Nseq, sdata=sig,
                 gridsearch=True, cvals=[1., 2., 4.], gvals=[0.5, 1., 1.5], verbose=vlevel)
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
    for j in range(nmembers):
        idx = np.random.choice(np.arange(sig.shape[0]), int(sig.shape[0] * subset_percentage), replace=False)
        mat_ml = FE.Material(name='ML-Hill_{}'.format(j))
        mat_ml.train_SVC(C=C, gamma=gamma, Fe=Fe, Ce=Ce, Nseq=Nseq,
                         sdata=sig[idx, :],
                         gridsearch=True, cvals=cvals, gvals=gvals, verbose=vlevel)
        hyp_C_list.append(mat_ml.C_yf)
        hyp_g_list.append(mat_ml.gam_yf)
        committee.append(mat_ml)

    # Search for next unit vector to query
    res = differential_evolution(eval_variance, bounds, args=(committee,), popsize=90, polish=True,
                                 updating='immediate')
    sunit_neww = res.x
    sunit_new = spherical_to_cartesian(sunit_neww)
    variance = res.fun

    var.append(-variance)
    # print(-variance)
    # print("number of iteration is:", i)
    # Calculate corresponding stress state and update data set
    x1 = fsolve(find_yloc, mat_h.sy, args=(sunit_new, mat_h), xtol=1.e-5)
    sig_new = sunit_new * x1[:, None]
    sig = np.vstack([sig, sig_new])
    sunit = np.vstack([sunit, sunit_new])

mat_ml = FE.Material(name='ML-Hill')  # define material
mat_ml.train_SVC(C=C, gamma=gamma, Fe=Fe, Ce=Ce, Nseq=Nseq,
                 sdata=sig,
                 gridsearch=True, cvals=cvals, gvals=gvals, verbose=vlevel)

# Create ML model with conventional training approach
Ntot = nsamples_init + nsamples_to_generate
c = int(Ntot / 3)
d = Ntot - c
sunit_r = FE.load_cases(number_3d=c, number_6d=d)
x1 = fsolve(find_yloc, np.ones(Ntot) * mat_h.sy, args=(sunit_r, mat_h), xtol=1.e-5)
sig_r = sunit_r * x1[:, None]
mat_ml_r = FE.Material(name='ML-Hill')  # define material
mat_ml_r.train_SVC(C=C, gamma=gamma, Fe=Fe, Ce=Ce, Nseq=Nseq,
                   sdata=sig_r,
                   gridsearch=True, cvals=cvals, gvals=gvals, verbose=vlevel)

plot_yield_locus(mat_ml, mat_h, nsamples_to_generate, mat3=mat_ml_r)
plot_variances(var)

fig = plt.figure()
plt.plot(hyp_g_list, 'b.', label='gamma')
plt.plot(hyp_C_list, 'r.', label='C')
plt.legend()
plt.title('Evolution of hyperparameters')
plt.xlabel('iteration * committee')
plt.ylabel('C, gamma')
plt.show()

np.savetxt('DATA_sig_iter_{}.txt'.format(i + 1), sig)
np.savetxt('DATA_sunit_iter_{}.txt'.format(i + 1), sunit)
np.savetxt('variance.txt', var)
