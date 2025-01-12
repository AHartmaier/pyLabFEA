#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ML flow rule to reference material with Hill-type anisotropy
in plastic behavior.
Linear strain hardening is considered.
Application of trained ML flow rule in FEA

Authors: Ronak Shoghi, Alexander Hartmaier
ICAMS/Ruhr University Bochum, Germany
January 2025

Published as part of pyLabFEA package under GNU GPL v3 license
"""

import pylabfea as FE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import fsolve


def create_data(mat, Nlc=300, epl_max=0.03, depl=1.e-3):
    # create set of unit stresses and assess yield stresses
    nl3d = int(Nlc / 3)
    nl6d = Nlc - nl3d
    sunit = FE.load_cases(number_3d=nl3d, number_6d=nl6d)
    # calculate yield stresses from unit stresses
    x1 = fsolve(mat.find_yloc, np.ones(Nlc) * mat.sy, args=(sunit,), xtol=1.e-5)
    sig_ideal = sunit * x1[:, None]  # initial yield stress at zero plastic strain
    assert len(sig_ideal) == Nlc
    # add strain data
    SV = np.linalg.inv(mat.CV)
    lc_data = dict()
    for i, st in enumerate(sig_ideal):
        epl = np.zeros(6)
        peeq = 0.0
        sig_list = []
        epl_list = []
        etot_list = []
        seq = FE.sig_eq_j2(st)
        sunit = st / seq
        ind = np.zeros(6, dtype=int)
        for j, su in enumerate(sunit):
            if su > 0.0:
                ind[j] = 1
            elif su < 0.0:
                ind[j] = 2
        key = f'Us_A{ind[0]}B{ind[1]}C{ind[2]}D{ind[3]}E{ind[4]}F{ind[5]}_HI{i:03d}_NNNNN_Tx_NN'
        dsig = seq / 5
        for j in range(6):
            sig = sunit * j * dsig
            sig_list.append(sig)
            epl_list.append(np.array(epl))
            etemp = np.dot(SV, sig)
            # etemp[3:6] *= 0.5
            etot_list.append(etemp)
        while peeq < epl_max:
            peeq = FE.eps_eq(epl) + depl
            sig = sunit * (seq + peeq * khard)
            epl_inc = mat.calc_fgrad(sig=sig, epl=epl) * depl
            epl += epl_inc
            etemp = np.dot(SV, sig)
            # etemp[3:6] *= 0.5
            etot = epl + etemp
            sig_list.append(sig)
            epl_list.append(np.array(epl))
            etot_list.append(etot)

        sig_ = np.array(sig_list)
        epl_ = np.array(epl_list)
        etot_ = np.array(etot_list)
        lc_data[key] = {"Stress": sig_,
                        "Eq_Stress": FE.sig_eq_j2(sig_),
                        "Strain_Plastic": epl_,
                        "Eq_Strain_Plastic": FE.eps_eq(epl_),
                        "Shifted_Strain_Plastic": None,  # required ???
                        "Strain_Total": etot_,
                        "Eq_Strain_Total": FE.eps_eq(etot_),
                        }
    return lc_data


def plot_sig_eps(mat, sig0, epl_max=0.02, depl=1.e-3):
    if not len(sig0) == 6:
        raise ValueError('Parameter "sig0" must be given as unit Voigt stress tensor.')
    sig0 = np.array(sig0, dtype=float)
    seq = FE.sig_eq_j2(sig0)
    if not np.isclose(seq, 0.0):
        sig0 /= seq
    else:
        raise ValueError('Parameter "sig0" has zero equivalent stress.')
    if not isinstance(mat, list):
        mat_list = [mat]
    else:
        mat_list = mat
    fig = plt.figure()
    err = []
    clist = ['r', 'b', 'm', 'k']
    for i, mat in enumerate(mat_list):
        x1 = fsolve(mat.find_yloc_scalar, mat.sy, args=(sig0,), xtol=1.e-5)
        sig = sig0 * x1
        seq = FE.sig_eq_j2(sig)
        peeq = 0.0
        epl = np.zeros(6)
        sig_plt = [seq]
        eps_plt = [0.0]
        val_yf = [mat.calc_yf(sig)]
        nc = 0
        while peeq <= epl_max and nc < 300:
            peeq = FE.eps_eq(epl) + depl
            sig += sig0 * peeq * khard
            epl_inc = mat.calc_fgrad(sig=sig, epl=epl) * depl
            epl += epl_inc
            x1 = fsolve(mat.find_yloc_scalar, mat.sy, args=(sig0, epl), xtol=1.e-5)
            sig = sig0 * x1
            sig_plt.append(FE.sig_eq_j2(sig))
            eps_plt.append(FE.eps_eq(epl))
            val_yf.append(mat.calc_yf(sig, epl=epl))
            nc += 1
        if nc >= 300:
            print(f'WARNING: Too many iterations, "epl_max" not reached. Stooping at PEEQ={eps_plt[-1]}.')
        plt.plot(eps_plt, sig_plt, color=clist[i], marker='o', label=mat.name)
        err.append(val_yf)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel(xlabel="Equivalent Plastic Strain (.)", fontsize=14)
    plt.ylabel(ylabel="Equivalent Stress (MPa)", fontsize=14)
    plt.title(f'LC: {sig0}')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return err

def plot_lc(mat, ilc=1, epl_crit=0.002):
    # Reconstruct Stress-Strain Curve
    # ilc: number of load case to be plotted
    # sig0: unit stress along which t reconstruct stress-strain curve
    if not isinstance(ilc, int):
        raise ValueError('One of parameters "ilc" or "sig0" must be specified to select load case.')
    istart = mat.msparam[0]['lc_indices'][ilc]
    istop = mat.msparam[0]['lc_indices'][ilc + 1]
    sig_dat = mat.msparam[0]['flow_stress'][istart:istop, :]  # filter below 0.02% strain
    epl_dat = mat.msparam[0]['plastic_strain'][istart:istop, :]  # filter below 0.02% strain
    epl_plt = FE.eps_eq(epl_dat)
    valid_indices = np.nonzero(epl_plt >= epl_crit)[0]
    # Update epl_dat and sig_dat based on the filter
    epl_dat_filtered = epl_dat[valid_indices, :]
    sig_dat_filtered = sig_dat[valid_indices, :]
    epl_plt_filtered = FE.eps_eq(epl_dat_filtered)
    # define unit stress in loading direction
    sig0 = sig_dat_filtered[-1, :] / FE.sig_eq_j2(sig_dat_filtered[-1, :])

    Np = len(epl_dat_filtered)
    sig_ml = []
    for i in range(Np):
        x1 = fsolve(mat.find_yloc, mat.sy, args=(sig0, epl_dat_filtered[i, :]), xtol=1.e-5)
        sig_ml.append(sig0 * x1)
    sig_ml = np.array(sig_ml)
    fig = plt.figure()
    plt.scatter(epl_plt_filtered, FE.sig_eq_j2(sig_dat_filtered), label="Data", s=9, color='black')
    plt.scatter(epl_plt_filtered, FE.sig_eq_j2(sig_ml), label="ML", s=10, color='#d60404')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel(xlabel="Equivalent Plastic Strain (.)", fontsize=14)
    plt.ylabel(ylabel="Equivalent Stress (MPa)", fontsize=14)
    plt.legend()
    plt.show()
    plt.close(fig)


# define Hill model as reference material
Nlc = 300
E = 200.e3  # Young's modulus in MPa
nu = 0.3  # Poisson ratio
sy = 50.  # yield strength in MPa
khard = 1000.0  # linear hardening coefficient
rv = [1.2, 1.0, 0.8, 1.0, 1.0, 1.0]  # parameters for yield stress ratios
# rv = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # for isotropic J2 plasticity
epl_max = 0.03  # maximum plastic strain to be considered in data generation
depl = 1.e-3  # plastic strain increments for plastic strain generation
mat_h = FE.Material(name='Hill-reference', num=1)
mat_h.elasticity(E=E, nu=nu)
mat_h.plasticity(sy=sy, rv=rv, khard=khard, sdim=6)

# define material as basis for ML flow rule
C = 2.0
gamma = 1.5
Ce = 0.99
Fe = 0.7
Nseq = 10
nbase = 'ML_Hill_hardening'
name = f'{nbase}_C{C:3.1f}_G{gamma:3.1f}'
lc_dict = create_data(mat_h, Nlc=Nlc, epl_max=epl_max, depl=depl)  # generate dictionary with stress-strain data
data_dict = FE.Data(lc_dict, mat_name=nbase,
                    epl_start=0.0, epl_crit=0.0,
                    epl_max=epl_max, depl=depl,
                    wh_data=True)
mat_mlh = FE.Material(name=name, num=2)  # define material
mat_mlh.from_data(data_dict.mat_data)  # data-based definition of material
print('\nComparison of basic material parameters:')
print(f"Young's modulus: Ref={mat_h.E} MPa, ML={mat_mlh.E} MPa")
print(f"Poisson's ratio: ref={mat_h.nu}, ML={mat_mlh.nu}")
print(f'Yield strength: Ref={mat_h.sy} MPa, ML={mat_mlh.sy} MPa')

# train SVC with data generated Hill reference material
mat_mlh.train_SVC(C=C, gamma=gamma,
                  Ce=Ce, Fe=Fe, Nseq=Nseq,
                  gridsearch=False)
# plot train yield locus with support vectors
sv = mat_mlh.svm_yf.support_vectors_ * mat_mlh.scale_seq
Nsv = len(sv)
sc = FE.sig_princ2cyl(sv[:, 0:6])
yf = mat_mlh.calc_yf(sv, pred=True)
print("ML material with {} support vectors, C={}, gamma={}, stress dimensions={}"
      .format(Nsv, mat_mlh.C_yf, mat_mlh.gam_yf, mat_mlh.sdim))
print("Plot shows initial yield locus of trained ML material and reference material "
      "together with all support vectors in stress space.")
mat_mlh.polar_plot_yl(data=sc, dname='support vectors', cmat=[mat_h], arrow=True)
# export ML parameters for use in UMAT
# mat_mlh.export_MLparam(__file__, path='./')

# analyze training result
size = 150
scale = 2
offset = 5
X1 = np.random.normal(loc=sy, scale=scale, size=int(size / 4))
X2 = np.random.normal(loc=(sy - offset), scale=scale, size=int(size / 2))
X3 = np.random.normal(loc=(sy + offset), scale=scale, size=int(size / 4))
X = np.concatenate((X1, X2, X3))
sunittest = FE.load_cases(number_3d=0, number_6d=len(X))
sig_test = sunittest * X[:, None]
yf_ml = mat_mlh.calc_yf(sig_test)
yf_ref = mat_h.calc_yf(sig_test)
FE.training_score(yf_ref, yf_ml)

# Plot initial and final hardening level of trained ML yield function together with data points
peeq_dat = FE.eps_eq(mat_mlh.msparam[0]['plastic_strain'])
ind0 = np.nonzero(np.logical_and(peeq_dat >= 0.0009, peeq_dat < 0.0011))[0]
sig_d0 = FE.s_cyl(mat_mlh.msparam[0]['flow_stress'][ind0, :], mat_mlh)
ind1 = np.nonzero(np.logical_and(peeq_dat > 0.0199, peeq_dat < 0.0201))[0]
sig_d1 = FE.s_cyl(mat_mlh.msparam[0]['flow_stress'][ind1, :], mat_mlh)
ngrid = 100
scale_seq, pi_factor = mat_mlh.scale_seq, np.pi
xx, yy = np.meshgrid(np.linspace(-1, 1, ngrid), np.linspace(0, 2, ngrid))
yy, xx = yy * scale_seq, xx * pi_factor
Cart_hh = FE.sp_cart(np.c_[yy.ravel(), xx.ravel()])
Cart_hh_6D = np.hstack((Cart_hh, np.zeros((ngrid ** 2, 3))))
grad_hh = mat_mlh.calc_fgrad(Cart_hh_6D)
normalized_grad_hh = grad_hh / FE.eps_eq(grad_hh)[:, None]
fig = plt.figure(figsize=(6.7, 4))
fig.set_constrained_layout(True)
ax = fig.add_subplot(111, projection='polar')
Z0 = mat_mlh.calc_yf(sig=Cart_hh_6D, epl=normalized_grad_hh * 0.001, pred=False)
Z1 = mat_mlh.calc_yf(sig=Cart_hh_6D, epl=normalized_grad_hh * 0.02, pred=False)
mat_mlh.plot_data(Z0, ax, xx, yy, c="#600000")
mat_mlh.plot_data(Z1, ax, xx, yy, c="#ff5050")
plt.scatter(sig_d0[:, 1], sig_d0[:, 0], s=5, c="black")
plt.scatter(sig_d1[:, 1], sig_d1[:, 0], s=5, c="black")
handle1 = Line2D([0], [0], color="#600000", label='Equivalent Plastic Strain : 0.1%')
handle2 = Line2D([0], [0], color="#ff5050", label='Equivalent Plastic Strain : 2.0%')
ax.legend(handles=[handle1, handle2], loc='upper left', bbox_to_anchor=(1.05, 1))
plt.show()
plt.close(fig)

# calculate and plot stress strain curves
print('Calculating stress-strain data ...')
emax = 0.01
mat_h.calc_properties(eps=emax, sigeps=True)
mat_mlh.calc_properties(verb=False, eps=emax, sigeps=True)
mat_mlh.plot_stress_strain()
mat_h.plot_stress_strain()

# plot yield locus with stress states
stx = mat_mlh.sigeps['stx']['sig'][:, 0:2] / mat_mlh.sy
sty = mat_mlh.sigeps['sty']['sig'][:, 0:2] / mat_mlh.sy
et2 = mat_mlh.sigeps['et2']['sig'][:, 0:2] / mat_mlh.sy
ect = mat_mlh.sigeps['ect']['sig'][:, 0:2] / mat_mlh.sy
s = 10
ax = mat_mlh.plot_yield_locus(xstart=-1.8, xend=1.8, ref_mat=mat_h, Nmesh=200)
ax.plot(stx[1:, 0], stx[1:, 1], 'or', markersize=s, markeredgecolor='#cc0000')
ax.plot(sty[1:, 0], sty[1:, 1], 'ob', markersize=s, markeredgecolor='#0000cc')
ax.plot(et2[1:, 0], et2[1:, 1], 'o', c='#808080', markersize=s, markeredgecolor='k')
ax.plot(ect[1:, 0], ect[1:, 1], 'om', markersize=s, markeredgecolor='#cc00cc')
# plot flow stresses from reference material
stx = mat_h.sigeps['stx']['sig'][:, 0:2] / mat_h.sy
sty = mat_h.sigeps['sty']['sig'][:, 0:2] / mat_h.sy
et2 = mat_h.sigeps['et2']['sig'][:, 0:2] / mat_h.sy
ect = mat_h.sigeps['ect']['sig'][:, 0:2] / mat_h.sy
s = 5
ax.plot(stx[1:, 0], stx[1:, 1], 'ok', markersize=s)
ax.plot(sty[1:, 0], sty[1:, 1], 'ok', markersize=s)
ax.plot(et2[1:, 0], et2[1:, 1], 'ok', markersize=s)
ax.plot(ect[1:, 0], ect[1:, 1], 'ok', markersize=s)
handles = ax.legend_.legend_handles
marker1 = Line2D([0], [0], marker='o', color='b', linestyle='None')
marker2 = Line2D([0], [0], marker='o', color='k', linestyle='None')
handles.append(marker1)
handles.append(marker2)
labels = [mat_mlh.name, mat_h.name, 'ML stress state', 'Reference stress state']
plt.legend(handles, labels, loc='upper left')
plt.show()
plt.close('all')

err = plot_sig_eps([mat_mlh, mat_h], sig0=[-1, 0, 0, 0, 0, 0])
print(f'Maximum absolute error in yield function for x-compression: '
      f'ML={max(np.abs(err[0]))}, REF={max(np.abs(err[1]))}')
err = plot_sig_eps([mat_mlh, mat_h], sig0=[0, -1, 0, 0, 0, 0])
print(f'Maximum absolute error in yield function for y-compression: '
      f'ML={max(np.abs(err[0]))}, REF={max(np.abs(err[1]))}')
err = plot_sig_eps([mat_mlh, mat_h], sig0=[-1, 1, 0, 0, 0, 0])
print(f'Maximum absolute error in yield function for shear loading: '
      f'ML={max(np.abs(err[0]))}, REF={max(np.abs(err[1]))}')
err = plot_sig_eps([mat_mlh, mat_h], sig0=[1, 1, 0, 0, 0, 0])
print(f'Maximum absolute error in yield function for biaxial loading:'
      f' ML={max(np.abs(err[0]))}, REF={max(np.abs(err[1]))}')
