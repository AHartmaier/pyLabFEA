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

print('pyLabFEA version', FE.__version__)

# set standard font size
# font = {'size': 16}
# plt.rc('font', **font)


def create_data(mat, Nlc=300, epl_max=0.03, depl=1.e-3):
    # create set of unit stresses and assess yield stresses
    nl3d = int(Nlc / 3)
    nl6d = Nlc - nl3d
    sunit = FE.load_cases(number_3d=nl3d, number_6d=nl6d)
    # calculate yield stresses from unit stresses
    x1 = fsolve(mat.find_yloc, np.ones(Nlc) * mat.sy, args=(sunit,), xtol=1.e-5)
    sig_ideal = sunit * x1[:, None]  # initial yield stress at zero plastic strain
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
        key = f'Us_A{ind[0]}B{ind[1]}C{ind[2]}D{ind[3]}E{ind[4]}F{ind[5]}_HI{Nlc:03d}_NNNNN_Tx_NN'
        dsig = seq / 5
        for j in range(6):
            sig = sunit * j * dsig
            sig_list.append(sig)
            epl_list.append(np.array(epl))
            etot_list.append(np.dot(SV, sig))
        while peeq < epl_max:
            peeq = FE.eps_eq(epl) + depl
            sig = sunit * (seq + peeq * khard)
            epl_inc = mat.calc_fgrad(sig, epl=epl)
            epl += epl_inc * depl
            etot = epl + np.dot(SV, sig)
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


def calc_yield(mat):
    # calculate reference yield stresses for uniaxial, equi-biaxial and pure shear load cases
    sunit = np.zeros((5, 6))
    sunit[0, 0] = 1.
    sunit[1, 1] = 1.
    sunit[2, 2] = 1.
    sunit[3, 0] = 1.
    sunit[3, 1] = 1.
    sunit[4, 0] = -1. / np.sqrt(3.)
    sunit[4, 1] = 1. / np.sqrt(3.)
    x1 = fsolve(mat.find_yloc, np.ones(5) * mat.sy, args=(sunit,), xtol=1.e-5)
    sy_ref = sunit * x1[:, None]
    return sy_ref


# define Hill model as reference material
E = 200.e3  # Young's modulus in MPa
nu = 0.3  # Poisson ratio
sy = 50.  # yield strength in MPa
khard = 1000.0  # linear hardening coefficient
# rv = [1.2, 1.0, 0.8, 1.0, 1.0, 1.0]  # parameters for yield stress ratios
rv = [1., 1.0, 1., 1.0, 1.0, 1.0]  # for isotropic J2 plasticity
epl_max = 0.03  # maximum plastic strain to be considered in data generation
depl = 1.e-3  # plastic strain increments for plastic strain generation
mat_h = FE.Material(name='Hill-reference', num=1)
mat_h.elasticity(E=E, nu=nu)
mat_h.plasticity(sy=sy, rv=rv, khard=khard, sdim=6)
mat_h.calc_properties(eps=0.01, sigeps=True)
lc_dict = create_data(mat_h, Nlc=300, epl_max=epl_max, depl=depl)  # generate dictionary with stress-strain data
sy_ref = calc_yield(mat_h)  # calculate yield stresses for reference load cases
seq_ref = FE.sig_eq_j2(sy_ref)

# define material as basis for ML flow rule
C = 2.0
gamma = 0.5
Ce = 0.95
Fe = 0.7
Nseq = 4
nbase = 'ML_Hill_hardening'
name = f'{nbase}_C{C:3.1f}_G{gamma:3.1f}'
data_dict = FE.Data(lc_dict, mat_name=nbase, wh_data=True)
mat_mlh = FE.Material(name=name, num=1)  # define material
mat_mlh.from_data(data_dict.mat_data)  # data-based definition of material

print('\nComparison of basic material parameters:')
print("Young's modulus: Ref={}MPa, ML={}MPa".format(mat_h.E, mat_mlh.E))
print("Poisson's ratio: ref={}, ML={}".format(mat_h.nu, mat_mlh.nu))
print('Yield strength: Ref={}MPa, ML={}MPa'.format(mat_h.sy, mat_mlh.sy))


# train SVC with data generated from Barlat model for material with Goss texture
mat_mlh.train_SVC(C=C, gamma=gamma,
                  Ce=Ce, Fe=Fe, Nseq=Nseq,
                  gridsearch=False)
sc = FE.sig_princ2cyl(mat_mlh.msparam[0]['sig_ideal'])
mat_mlh.polar_plot_yl(data=sc, dname='training data', cmat=[mat_h], arrow=True)
# export ML parameters for use in UMAT
# mat_mlh.export_MLparam(__file__, path='./')
# analyze training result
loc = 40
scale = 10
size = 200
offset = 5
X1 = np.random.normal(loc=loc, scale=scale, size=int(size / 4))
X2 = np.random.normal(loc=(loc - offset), scale=scale, size=int(size / 2))
X3 = np.random.normal(loc=(loc + offset), scale=scale, size=int(size / 4))
X = np.concatenate((X1, X2, X3))
sunittest = FE.load_cases(number_3d=0, number_6d=len(X))
sig_test = sunittest * X[:, None]
yf_ml = mat_mlh.calc_yf(sig_test)
yf_GB = mat_h.calc_yf(sig_test)
FE.training_score(yf_GB, yf_ml)

# Reconstruct Stress-Strain Curve
ilc = 0  # number of load case to be plotted keep this section.
offs = 0 if ilc == 0 else 1
istart = mat_mlh.msparam[0]['lc_indices'][ilc - 1] + offs
istop = mat_mlh.msparam[0]['lc_indices'][ilc]
sig_dat = mat_mlh.msparam[0]['flow_stress'][istart:istop, :]  # filter below 0.02% strain
epl_dat = mat_mlh.msparam[0]['plastic_strain'][istart:istop, :]  # filter below 0.02% strain
epl_plt = FE.eps_eq(epl_dat)
valid_indices = epl_plt >= 0.002
# Update epl_dat and sig_dat based on the filter
epl_dat_filtered = epl_dat[valid_indices, :]
sig_dat_filtered = sig_dat[valid_indices, :]
epl_plt_filtered = FE.eps_eq(epl_dat_filtered)
# define unit stress in loading direction
sig0 = sig_dat_filtered[-1, :] / FE.sig_eq_j2(sig_dat_filtered[-1, :])
Nlc = len(epl_dat_filtered)
sig_ml = []
for i in range(Nlc):
    x1 = fsolve(mat_mlh.find_yloc, mat_mlh.sy, args=(sig0, epl_dat_filtered[i, :]), xtol=1.e-5)
    sig_ml.append(sig0 * x1)
sig_ml = np.array(sig_ml)
fig = plt.figure(figsize=(5.6, 4.7))
plt.scatter(epl_plt_filtered, FE.sig_eq_j2(sig_dat_filtered), label="Data", s=9, color='black')
plt.scatter(epl_plt_filtered, FE.sig_eq_j2(sig_ml), label="ML", s=10, color='#d60404')
text_size = 12
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel(xlabel="Equivalent Plastic Strain (.)", fontsize=14)
plt.ylabel(ylabel="Equivalent Stress (MPa)", fontsize=14)
legend_font_size = 12
legend = plt.legend(fontsize=legend_font_size)
plt.tight_layout()
plt.show()

"""
# analyze support vectors to plot them in stress space
sv = mat_mlh.svm_yf.support_vectors_ * mat_mlh.scale_seq
Nsv = len(sv)
sc = FE.sig_princ2cyl(sv[:, 0:6])
yf = mat_mlh.calc_yf(sv, pred=True)
print("ML material with {} support vectors, C={}, gamma={}, stress dimensions={}"
      .format(Nsv, mat_mlh.C_yf, mat_mlh.gam_yf, mat_mlh.sdim))
mat_mlh.polar_plot_yl(data=sc, dname='support vectors', cmat=[mat_h], arrow=True)"""

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
'''
# calculate and plot stress strain curves
print('Calculating stress-strain data ...')
mat_mlh.calc_properties(verb=False, eps=0.01, sigeps=True)
mat_mlh.plot_stress_strain()
mat_h.plot_stress_strain()
print('\nYield stresses from reference material:')
print('---------------------------------------------------------')
print(f'J2 yield stress under uniax-x loading: {seq_ref[0]:6.3f} MPa')
print(f'J2 yield stress under uniax-y loading: {seq_ref[1]:6.3f} MPa')
print(f'J2 yield stress under equibiax loading: {seq_ref[3]:6.3f} MPa')
print(f'J2 yield stress under shear loading: {seq_ref[4]:6.3f} MPa')
print('---------------------------------------------------------')

# plot yield locus with stress states
s = 80
ax = mat_mlh.plot_yield_locus(xstart=-1.8, xend=1.8, ref_mat=mat_h, Nmesh=200)
stx = mat_mlh.sigeps['stx']['sig'][:, 0:2] / mat_mlh.sy
sty = mat_mlh.sigeps['sty']['sig'][:, 0:2] / mat_mlh.sy
et2 = mat_mlh.sigeps['et2']['sig'][:, 0:2] / mat_mlh.sy
ect = mat_mlh.sigeps['ect']['sig'][:, 0:2] / mat_mlh.sy
ax.scatter(stx[1:, 0], stx[1:, 1], s=s, c='r', edgecolors='#cc0000')
ax.scatter(sty[1:, 0], sty[1:, 1], s=s, c='b', edgecolors='#0000cc', label='FEA result')
ax.scatter(et2[1:, 0], et2[1:, 1], s=s, c='#808080', edgecolors='k')
ax.scatter(ect[1:, 0], ect[1:, 1], s=s, c='m', edgecolors='#cc00cc')
# plot flow stresses from reference material
s = 40
stx = mat_h.sigeps['stx']['sig'][:, 0:2] / mat_h.sy
sty = mat_h.sigeps['sty']['sig'][:, 0:2] / mat_h.sy
et2 = mat_h.sigeps['et2']['sig'][:, 0:2] / mat_h.sy
ect = mat_h.sigeps['ect']['sig'][:, 0:2] / mat_h.sy
ax.scatter(stx[1:, 0], stx[1:, 1], s=s, c='c', edgecolors='#cc0000')
ax.scatter(sty[1:, 0], sty[1:, 1], s=s, c='c', edgecolors='#0000cc')
ax.scatter(et2[1:, 0], et2[1:, 1], s=s, c='c', edgecolors='k')
ax.scatter(ect[1:, 0], ect[1:, 1], s=s, c='c', edgecolors='#cc00cc')
plt.show()
plt.close('all')'''

'''
# setup material definition for soft elastic square-shaped inclusion embedded
# in elastic-plastic material with trained ML flow rule
print('Calculating FE model with elastic inclusion')
mat_el = FE.Material(num=2)  # define soft elastic material for inclusion
mat_el.elasticity(E=1.e3, nu=0.27)
# define array for geometrical arrangement
NX = NY = 12
el = np.ones((NX, NY))  # field with material 1: ML plasticity
NXi1 = int(NX / 3)
NXi2 = 2 * NXi1
NYi1 = int(NY / 3)
NYi2 = 2 * NYi1
el[NXi1:NXi2, NYi1:NYi2] = 2  # center material 2: elastic

# create FE model to perform uniaxial tensile test of model with inclusion
fem = FE.Model(dim=2, planestress=False)
fem.geom(sect=2, LX=4., LY=4.)  # define geometry with two sections
fem.assign([mat_mlh, mat_el])  # define sections for reference, ML and elastic material
fem.bcbot(0., 'disp')  # fixed bottom layer
fem.bcleft(0., 'force')  # free lateral edges
fem.bcright(0., 'force')
fem.bctop(0.005 * fem.leny, 'disp')  # apply displacement at top nodes (uniax y-stress)
fem.mesh(elmts=el, NX=NX, NY=NY)
# fix lateral displacements of corner node to prevent rigid body motion
hh = [no in fem.nobot for no in fem.noleft]
noc = np.nonzero(hh)[0]  # find corner node
fem.bcnode(noc, 0., 'disp', 'x')  # fix lateral displacement
fem.solve()

# plot results
fem.plot('mat', shownodes=False, mag=0)
fem.plot('seq', shownodes=False, showmesh=False, mag=10)
fem.plot('peeq', shownodes=False, showmesh=False, mag=10)'''
