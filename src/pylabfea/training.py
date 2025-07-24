# Module pylabfea.training
"""Module pylabfea.training introduces methods to create training data for ML flow rule
in shape of unit stresses that are evenly distributed in the stress space to
define the load cases for which the critical stress tensor at which plastic yielding
starts needs to be determined.

uses NumPy, ScipPy, MatPlotLib, sklearn, and pyLabFEA.basic

Version: 4.0 (2021-11-27)
Authors: Ronak Shoghi, Alexander Hartmaier, ICAMS/Ruhr University Bochum, Germany
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)

Subroutines int_sin_m, primes and uniform_hypersphere have been adapted from
code published by Stack Overflow under the CC-BY-SA 4.0 license, see
https://stackoverflow.com/questions/57123194/how-to-distribute-points-evenly-on-the-surface-of-hyperspheres-in-higher-dimensi/59279721#59279721
These subroutines are distributed here under the CC-BY-SA 4.0 license, see https://creativecommons.org/licenses/by-sa/4.0/
"""

from pylabfea.basic import sig_eq_j2
import numpy as np
from itertools import count
from scipy.special import gamma
from scipy.optimize import root_scalar
from sklearn.metrics import mean_absolute_error, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
import collections
import pylabfea as FE


def int_sin_m(x, m):
    """Computes the integral of sin^m(t) dt from 0 to x recursively

    Parameters
    ----------
    x : float
        Upper limit of integration
    m : int
        Power of trigonometric function to be considered

    Returns
    -------
    f : float
        Value of integral
    """
    if m == 0:
        hh = x
    elif m == 1:
        hh = 1. - np.cos(x)
    else:
        hh = (m - 1) / m * int_sin_m(x, m - 2) - np.cos(x) * np.sin(x) ** (m - 1) / m
    return hh


def primes():
    """Infinite generator of prime numbers"""
    yield from (2, 3, 5, 7)
    composites = {}
    ps = primes()
    next(ps)
    p = next(ps)
    assert p == 3
    psq = p * p
    for i in count(9, 2):
        if i in composites:  # composite
            step = composites.pop(i)
        elif i < psq:  # prime
            yield i
            continue
        else:  # composite, = p*p
            assert i == psq
            step = 2 * p
            p = next(ps)
            psq = p * p
        i += step
        while i in composites:
            i += step
        composites[i] = step


def uniform_hypersphere(d, n, method='brentq'):
    """Generate n usnits stresse on the d dimensional hypersphere
    representing create load cases in 3D or 6D stress space

    Parameters
    ----------
    d : int
        Dimension of stress space in which to create unit stresses
    n : int
        Number of stresses to be created

    Returns
    -------
    points : (n,6)-array
        Unit stresses
    """

    def dim_func(y, x):
        return mult * int_sin_m(y, dim - 1) - x

    points = np.ones((n, d))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points[:, 0] = np.sin(t)
    points[:, 1] = np.cos(t)
    for dim, prime in zip(range(2, d), primes()):
        offset = np.sqrt(prime)
        mult = gamma(0.5 * (dim + 1)) / (gamma(0.5 * dim) * np.sqrt(np.pi))

        for i in range(n):
            res = root_scalar(dim_func, args=(i * offset % 1), method=method,
                              bracket=[0, np.pi], xtol=1.e-8)  # search root of int_sin-arg in range [0, pi]
            deg = res.root
            if not res.converged:
                print('Root finding with method "{}" not converged. Rootresults={}' \
                      .format(method, res))
            for j in range(dim):
                points[i, j] *= np.sin(deg)
            points[i, dim] *= np.cos(deg)
    return points


def load_cases(number_3d, number_6d, method='brentq'):
    """Generate unit stresses in principal stress space (3d) and full stress space (6d)

    Parameters
    ----------
    number_3d : int
        Number of principal unit stresses to be created
    number_6d : int
        Number of full unit stresses to be created

    Returns
    -------
    allsig : (number_3d+number6d, 6)-array
        Unit stresses
    """
    sig_3d = np.zeros((number_3d, 6))
    sig_3d[:, 0:3] = uniform_hypersphere(3, number_3d, method=method)
    sig_6d = uniform_hypersphere(6, number_6d)
    allsig = np.concatenate((sig_3d, sig_6d))
    seq = sig_eq_j2(allsig)
    ind = np.nonzero(seq < 1.e-3)[0]
    if len(ind) > 0:
        print('WARNING: Small stresses detected:', ind)
    allsig /= seq[:, None]
    return allsig


def training_score(yf_ref, yf_ml, plot=False):
    """Calculate the accuracy of the training result in form of different measures
    as compared to given reference values.

    Parameters
    ----------
    yf_ref : (N,)-array
        Yield function values of reference material
    yf_ml : (N,)-array
        Yield function values of ML material at identical sequence of stresses
        at which reference material is evaluated.

    Returns
    -------
    mae : float
        Mean Average Error
    precision : float
        Ratio of true positives w.r.t. all positives
    Accuracy : float
        Ratio of true positives and true negative w.r.t. all results
    Recall : float
        Ratio of true positives w.r.t. true positives and false negatives
    F1Score : float
        F1 score
    MCC : float
        Matthews Correlation Coefficient
    """
    res_yf_ref = np.sign(yf_ref)
    ind = np.nonzero(np.abs(res_yf_ref) < 0.9)[0]
    res_yf_ref[ind] = 1.  # change points with yf=0 to +1
    res_yf_ml = np.sign(yf_ml)
    ind = np.nonzero(np.abs(res_yf_ml) < 0.9)[0]
    res_yf_ml[ind] = 1.  # change points with yf=0 to +1

    if plot:
        cm = confusion_matrix(res_yf_ref, res_yf_ml)
        plt.figure(figsize=(2, 2))  # Set the figure size to 4x4 inches
        cmd = ConfusionMatrixDisplay(cm, display_labels=['Elastic', 'Plastic'])
        cmd.plot(cmap='viridis', colorbar=False)
        ax = plt.gca()
        ax.set_xlabel('Predicted label', fontsize=16)
        ax.set_ylabel('True label', fontsize=16)
        ax.set_xticklabels(['Elastic', 'Plastic'], fontsize=14)
        ax.set_yticklabels(['Elastic', 'Plastic'], fontsize=14)
        for text in ax.texts:
            text.set_size(14)
        plt.colorbar(cmd.im_, ax=ax).set_label(label="Number of samples", size=14)
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.show()
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(res_yf_ref)):
        if (res_yf_ref[i] == 1) & (res_yf_ml[i] == 1):
            TP += 1
        if (res_yf_ref[i] == 1) & (res_yf_ml[i] == -1):
            FN += 1
        if (res_yf_ref[i] == -1) & (res_yf_ml[i] == 1):
            FP += 1
        if (res_yf_ref[i] == -1) & (res_yf_ml[i] == -1):
            TN += 1
    mae = mean_absolute_error(yf_ref, yf_ml)
    MCC = matthews_corrcoef(np.sign(yf_ref), np.sign(yf_ml), sample_weight=None)
    print("Mean Absolut Error is", mae)
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    if TP + FP > 0:
        precision = (TP) / (TP + FP)
    else:
        precision = 0.0
    print('Precision:', precision)
    if TP + FP + FN + TN > 0:
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
    else:
        Accuracy = 0.0
    print('Accuracy:', Accuracy)
    if TP + FN > 0:
        Recall = (TP) / (TP + FN)
    else:
        Recall = 0.0
    print('Recall:', Recall)
    if Recall + precision > 1.0e-4:
        F1Score = 2 * (Recall * precision) / (Recall + precision)
    else:
        F1Score = 0.0
    print('F1score:', F1Score)
    print('MCC score:', MCC)
    return mae, precision, Accuracy, Recall, F1Score, MCC


def create_test_sig(file, number_sig_per_strain=4):
    """ A function to generate test data for micromechanical simulations based on a given material's stress-strain data.
    Parameters
    ----------
    Json : str
        Path to the JSON file containing the stress-strain data of the material.

    Number_sig_per_strain : int, optional
        The number of test cases to generate for each strain level in the elastic or in the plastic range.
        The total number of load cases per strain level will be 2 * Number_sig_per_strain. Default is 4.

    Returns
    -------
    ts_sig : np.ndarray
        A numpy array of stress values for the generated test cases.

    epl_tot : np.ndarray
        A numpy array of strain values for the generated test cases.

    yf_ref : np.ndarray
        A numpy array with the same length as ts_sig and epl_tot, filled with +1 for the first half of the elements
        and -1 for the second half. This represents the known sign of the yield function: +1 for upscaling and -1 for downscaling.
        This array will be used as reference in the training_score function.
    """

    db2 = FE.Data(file, 
             epl_crit=2.e-3, epl_start=1.e-3, epl_max=0.03,
             depl=0.0)
    #mat_ts = FE.Material(name="Test")  # define material
    #mat_ts.elasticity(CV=db2.mat_data['elast_const'])
    #mat_ts.plasticity(sy=db2.mat_data['sy_av'], khard=4.5e3)
    #mat_ts.calc_properties(verb=False, eps=0.03, sigeps=True)
    #mat_ts.from_data(db2.mat_data)
    pl_sig = []
    el_sig = []
    epl_ts = []

    for j in range(len(db2.mat_data['plastic_strain'])):
        pl_sig.append(db2.mat_data['flow_stress'][j] * 1.5)
        pl_sig.append(db2.mat_data['flow_stress'][j] * 1.2)
        pl_sig.append(db2.mat_data['flow_stress'][j] * 1.1)
        pl_sig.append(db2.mat_data['flow_stress'][j] * 1.01)
        el_sig.append(db2.mat_data['flow_stress'][j] * 0.99)
        el_sig.append(db2.mat_data['flow_stress'][j] * 0.9)
        el_sig.append(db2.mat_data['flow_stress'][j] * 0.8)
        el_sig.append(db2.mat_data['flow_stress'][j] * 0.5)
        for nsps in range(int(number_sig_per_strain)):
            epl_ts.append(db2.mat_data['plastic_strain'][j].tolist())

    sig_tot = pl_sig + el_sig
    epl_tot = epl_ts + epl_ts
    ts_sig = np.array(sig_tot)
    epl_tot = np.array(epl_tot)
    half_len = len(ts_sig) // 2
    pos = np.ones(half_len)
    neg = np.ones(len(ts_sig) - half_len) * -1
    yf_ref = np.concatenate((pos, neg), axis=None)

    return (ts_sig, epl_tot, yf_ref)
