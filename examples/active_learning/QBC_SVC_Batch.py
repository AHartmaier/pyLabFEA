#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script introduces an Active Learning method for SVM to enhance data selection
in training Machine Learning yield functions. Using the Query-By-Committee algorithm
we prioritize data points where model predictions show high disagreement,
leading to reduced variance in predictions.

Authers: Ronak Shoghi1, Lukas Morand2, Alexandere Hartmaier1
1: [ICAMS/Ruhr University Bochum, Germany]
2: [Fraunhofer Institute for Mechanics of Materials (IWM)]
July 2023
"""

import sys
import numpy as np
from scipy.spatial import distance_matrix
sys.path.append('src/data-gen')
sys.path.append('src/verify')
import pylabfea as FE
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

matplotlib.use('Agg')
print('pyLabFEA version', FE.__version__)

def calculate_knn_distances(points, k):
    """Calculate the distance to the kth nearest neighbor for each point."""
    distances = distance_matrix(points, points)
    np.fill_diagonal(distances, np.inf)  # Ignore distance to self
    sorted_distances = np.sort(distances, axis=1)
    return sorted_distances[:, k-1]  # kth nearest neighbor distance

def apply_repulsion(points, k, iterations, learning_rate):
    """Apply repulsion to maximize the distance to the kth nearest neighbor."""
    for _ in range(iterations):
        distances = distance_matrix(points, points)
        np.fill_diagonal(distances, np.inf)
        indices = np.argsort(distances, axis=1)[:, :k]  # Indices of k nearest neighbors

        # Initialize displacement vector
        displacement = np.zeros_like(points)

        # Calculate repulsion from each of the k nearest neighbors
        for i, point in enumerate(points):
            neighbors = points[indices[i]]
            diff = point - neighbors  # Vector from neighbors to point
            distances_to_neighbors = distances[i, indices[i]].reshape(-1, 1)
            repulsion = diff / distances_to_neighbors ** 2  # Repulsion proportional to inverse square of distance
            displacement[i] = repulsion.sum(axis=0)

        # Update points with displacement
        points += learning_rate * displacement

        # Normalize to keep points on the sphere surface
        norms = np.linalg.norm(points, axis=1).reshape(-1, 1)
        points /= norms

    return points


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
    x1=np.cos(angles[0])
    x2=np.sin(angles[0]) * np.cos(angles[1])
    x3=np.sin(angles[0]) * np.sin(angles[1]) * np.cos(angles[2])
    x4=np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.cos(angles[3])
    x5=np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(angles[3]) * np.cos(angles[4])
    x6=np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(angles[3]) * np.sin(angles[4])
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
    points=[]
    for i in range(npoints):
        point=[]
        while True:
            for j in range(6):
                value=np.random.uniform(-1, 1)
                point.append(value)
            norm=np.linalg.norm(point)
            if norm != 0:
                break
            else:
                point=[]
        point_normalized=np.array(point) / norm
        point_rounded=np.around(point_normalized, decimals = precision)
        points.append(point_rounded)
    return np.vstack(points)


def hard_test_cases(a,b):
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
    sunit1= FE.load_cases(number_3d=a, number_6d=b)
    npoints= (a + b)
    x1=fsolve(find_yloc, np.ones(npoints) * mat_h.sy, args = (sunit1, mat_h), xtol = 1.e-5)
    sig1=sunit1 * 0.99 * x1[:, None]
    # Create another set of random unit vectors and scale them by 1.01 times the solution from 'find_yloc'
    sunit2=FE.load_cases(number_3d=a, number_6d=b)
    x2=fsolve(find_yloc, np.ones(npoints) * mat_h.sy, args = (sunit2, mat_h), xtol = 1.e-5)
    sig2=sunit2 * 1.01 * x2[:, None]
    sig=np.concatenate((sig1, sig2))
    print(sig)
    return sig

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
    f=mat.calc_yf(sig * x[:, None])
    return f


def eval_max_disagreement(angles, committee, sunit_new_list, average_distance, penalty_weight, type, mat_h, sig):
    """
    Evaluate the maximum disagreement among the committee based on the given angles
    and penalize new points that are too close to existing points.

    Parameters:
        angles: List of spherical angles.
        committee: List of committee members (each member has a 'calc_yf' method).
        sunit_new_list: List of existing points in the feature space.
        min_distance: Minimum allowed distance to existing points.
        penalty_weight: Weight for the distance penalty.
        type: Type of evaluation ('standard' or 'input_comparison').
        mat_h: Reference material.
        sig: Stress tensor.

    Returns:
        The negative variance of the yield function outputs, adjusted by a penalty if needed.
    """
    # Convert from spherical to Cartesian coordinates
    x=spherical_to_cartesian(np.array(angles))

    # Calculate the variance among the committee members
    y=np.zeros(len(committee))
    for i, member in enumerate(committee):
        y[i]=member.calc_yf(x)
    variance=np.sum(np.square(y - np.mean(y) * np.ones_like(y)))

    # Standard evaluation without considering the input comparison
    if type == 'standard':
        return -variance

    # Evaluation with input comparison and penalization for being too close to existing points
    elif type == 'input_comparison':
        penalty=0.0
        if len(sunit_new_list) > 0:
            current_k=min(5, len(sunit_new_list))  # Adjust the number of neighbors
            nn=NearestNeighbors(n_neighbors = current_k)
            nn.fit(sunit_new_list)
            distances, _=nn.kneighbors(x.reshape(1, -1))
            for dist in distances[0]:
                if dist < average_distance:
                    penalty+=(average_distance - dist) / average_distance
        return -variance - penalty * penalty_weight

    else:
        raise NotImplementedError()

def read_vectors(file_name):
    # Load the vectors from the file
    vectors=np.loadtxt(file_name)
    return vectors

def plot_variances(var_list):
    plt.plot(var_list, marker = 'o')
    plt.xlabel('Iteration')
    plt.ylabel('Variance')
    plt.title('Variance vs Iteration')
    plt.grid()
    plt.savefig('variances_vs_iterations.png', dpi = 300)
    plt.close()

def save_hard_test_cases(a , b, num_tests=5):
    test_arrays=[[] for _ in range(num_tests)]
    for i in range(num_tests):
        sig_test=hard_test_cases(a, b)
        np.savetxt(f'sig_test_{i + 1}.txt', sig_test)
        test_arrays[i]=sig_test
    return test_arrays

# Query by committee parameters

nmembers=5  # Number of committee members
# Number of initial samples - can be chosen by the user
nsamples_to_generate=70 #  Number of iterations
sampling_scheme='max_disagreement'  # max disagreement for yf-predictions, for classifiers generally possible: vote_entropy, consensus_entropy or maximum_disagreement, cf. https://modal-python.readthedocs.io/en/latest/content/query_strategies/Disagreement-sampling.html#disagreement-sampling
subset_percentage=0.8
subset_assignment='random'
# setup reference material with Hill-like anisotropy
path=os.path.dirname(__file__)
sy=50.
E=200000.
nu=0.3
hill=[1.4, 1, 0.7, 1.3, 0.8, 1]
mat_h=FE.Material(name = 'Hill-reference')
mat_h.elasticity(E = E, nu = nu)
mat_h.plasticity(sy = sy, hill = hill)
mat_h.calc_properties(eps = 0.0013, sigeps = True)
c = 8
d = 22
N = c+d
nsamples_init = N
sunit_random= FE.load_cases(number_3d=c, number_6d=d)
sunit = apply_repulsion(sunit_random, k=5, iterations=60, learning_rate=0.01)
np.savetxt('Test_Cases.txt', sunit)
final_knn_distances = calculate_knn_distances(final_points, k =5)
average_distance = np.mean(final_knn_distances)
# create set of unit stresses and
print('Created {0} unit stresses (6d Voigt tensor).'.format(N))
x1=fsolve(find_yloc, np.ones(N) * mat_h.sy, args = (sunit, mat_h), xtol = 1.e-5)
sig=sunit * x1[:, None]
print('Calculated {} yield stresses.'.format(N))
sc0=FE.sig_princ2cyl(sig)
np.savetxt('DATA_sig_iter_0.txt', sig)
np.savetxt('DATA_sunit_iter_0.txt', sunit)
var = []
for i in range(nsamples_to_generate):
    # train SVC committee with yield stress data generated from Hill flow rule
    C=2
    gamma=2.5
    committee=[]
    for j in range(nmembers):
        if subset_assignment == 'random':
            idx=np.random.choice(np.arange(sig.shape[0]), int(sig.shape[0] * subset_percentage), replace = False)
        else:
            raise NotImplementedError('chosen subset assignment not implemented')
        mat_ml=FE.Material(name = 'ML-Hill_{}'.format(j))
        mat_ml.train_SVC(C = C, gamma = gamma, sdata = sig[idx, :], gridsearch = True)
        committee.append(mat_ml)

    # Search for next unit vector to query
    bounds=[(0, np.pi)] + [(0, 2 * np.pi)] * 4
    if sampling_scheme == 'max_disagreement':
        res=differential_evolution(
            eval_max_disagreement,
            bounds,
            args = (committee, repulsed_points, average_distance, 999, 'input_comparison', mat_h, sig),
            popsize = 90,
            polish = True,
            updating = 'immediate'
        )
        sunit_neww=res.x
        sunit_new=spherical_to_cartesian(sunit_neww)
        variance=res.fun
        # Calculate corresponding stress state and update data set
        x1=fsolve(find_yloc, mat_h.sy, args = (sunit_new, mat_h), xtol = 1.e-5)
        sig_new=sunit_new * x1[:, None]
        sig=np.vstack([sig, sig_new])
        sunit=np.vstack([sunit, sunit_new])
        sunit_new_list.append(sunit_new)
        sig_new_list.append(sig_new)
        variance=res.fun
        average_distance=calculate_average_min_distance(sunit_new_list)

    if i == nsamples_to_generate - 1:
        np.savetxt('DATA_sig_iter_{}.txt'.format(i + 1), sig)
        np.savetxt('DATA_sunit_iter_{}.txt'.format(i + 1), sunit)
    # train SVC with yield stress data generated from Hill flow rules
    C=2
    gamma=2.5
    mat_ml=FE.Material(name = 'ML-Hill')  # define material
    mat_ml.train_SVC(C = C, gamma = gamma, sdata = sig, gridsearch = True)
    # stress strain curves
    print("Calculating properties of ML material, this might take a while ...")
    mat_ml.elasticity(E = E, nu = nu)
    mat_ml.plasticity(sy = sy)
    mat_ml.calc_properties(verb = False, eps = 0.0013, sigeps = True)

    if i == nsamples_to_generate - 1 or i == 0:
        ngrid=50
        xx, yy=np.meshgrid(np.linspace(-1, 1, ngrid), np.linspace(0, 2, ngrid))
        yy*=mat_ml.scale_seq
        xx*=np.pi
        hh=np.c_[yy.ravel(), xx.ravel()]
        Z=mat_ml.calc_yf(FE.sig_cyl2princ(hh))  # value of yield function for every grid point
        Z2=mat_h.calc_yf(FE.sig_cyl2princ(hh))
        fig, ax=plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8))
        line=mat_ml.plot_data(Z, ax, xx, yy, c = 'black')
        line2=mat_h.plot_data(Z2, ax, xx, yy, c = 'blue')
        ax.set_xlabel(r'$\theta$ (rad)', fontsize = 22)
        ax.set_ylabel(r'$\sigma_{eq}$ (MPa)', fontsize = 22)
        ax.tick_params(axis = "x", labelsize = 18)
        ax.tick_params(axis = "y", labelsize = 18)
        plt.savefig('PLOTS_equiv_yield_stresses_iter_{}.png'.format(i + 1))
        plt.close('all')

    var.append(-variance)
    print(-variance)
    print("number of iteration is:", i)

np.savetxt('variance.txt', var)
plot_variances(var)
