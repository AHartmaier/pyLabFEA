#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ML flow rule to reference material with Hill-type anisotropy
in plastic behavior;
application of trained ML flow rule in FEA

Authors: Ronak Shoghi, Alexander Hartmaier
ICAMS/Ruhr University Bochum, Germany
January 2022
"""

import pylabfea as FE
import numpy as np
import matplotlib.pyplot as plt

print('pyLabFEA version', FE.__version__)

# set standard font size
font = {'size': 16}
plt.rc('font', **font)

# define Hill model as reference material
E = 200.e3  # Young's modulus in MPa
nu = 0.3  # Poisson ratio
sy = 50.  # yield strength in MPa
hill = [1.4, 1.0, 0.7, 1.3, 0.8, 1.0]  # parameters for Hill-type anisotropy
rv = [1.2, 1.0, 0.8, 1.0, 1.0, 1.0]  # parameters for yield stress ratios
mat_h = FE.Material(name='Hill-reference', num=1)
mat_h.elasticity(E=E, nu=nu)
mat_h.plasticity(sy=sy, rv=rv, sdim=6)
mat_h.calc_properties(eps=0.01, sigeps=True)

# define material as basis for ML flow rule
C = 15.
gamma = 2.5
nbase = 'ML-Hill-p1'
name = '{0}_C{1}_G{2}'.format(nbase, int(C), int(gamma * 10))
mat_mlh = FE.Material(name, num=2)  # define ML material
# train ML flow rule from reference material
mat_mlh.train_SVC(C=C, gamma=gamma, mat_ref=mat_h, Nlc=300)

# analyze support vectors to plot them in stress space
sv = mat_mlh.svm_yf.support_vectors_ * mat_mlh.scale_seq
Nsv = len(sv)
sc = FE.sig_princ2cyl(sv)
yf = mat_mlh.calc_yf(sv, pred=True)
print("ML material with {} support vectors, C={}, gamma={}, stress dimensions={}"
      .format(Nsv, mat_mlh.C_yf, mat_mlh.gam_yf, mat_mlh.sdim))
mat_mlh.polar_plot_yl(data=sc, dname='support vectors', cmat=[mat_h], arrow=True)
# export parameters of trained ML flow rule in UMAT format
# mat_mlh.export_MLparam(__file__, path='../models/')

# create plot of trained yield function in cylindrical stress space
print('Plot of trained SVM classification with test data in 2D cylindrical stress space')
ngrid = 50
xx, yy = np.meshgrid(np.linspace(-1, 1, ngrid), np.linspace(0, 2, ngrid))
yy *= mat_mlh.scale_seq
xx *= np.pi
hh = np.c_[yy.ravel(), xx.ravel()]
Z = mat_mlh.calc_yf(FE.sig_cyl2princ(hh))  # value of yield function for every grid point
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
line = mat_mlh.plot_data(Z, ax, xx, yy, c='black')
pts = ax.scatter(sc[:, 1], sc[:, 0], s=20, c=yf, cmap=plt.cm.Paired, edgecolors='k')  # plot support vectors
ax.set_xlabel(r'$\theta$ (rad)', fontsize=20)
ax.set_ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=20)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
plt.legend([line[0], pts], ['ML yield locus', 'support vectors'], loc='lower right')
plt.ylim(0., 2. * mat_mlh.sy)
plt.show()

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
yf_h = mat_h.calc_yf(sig_test)
FE.training_score(yf_h, yf_ml)

# stress strain curves
print('Calculating stress-strain data ...')
mat_mlh.calc_properties(verb=False, eps=0.01, sigeps=True)
mat_mlh.plot_stress_strain()
mat_h.plot_stress_strain()

# plot yield locus with flow stress states
s = 80
ax = mat_mlh.plot_yield_locus(xstart=-1.8, xend=1.8, ref_mat=mat_h, Nmesh=200)
stx = mat_mlh.sigeps['stx']['sig'][:, 0:2] / mat_mlh.sy
sty = mat_mlh.sigeps['sty']['sig'][:, 0:2] / mat_mlh.sy
et2 = mat_mlh.sigeps['et2']['sig'][:, 0:2] / mat_mlh.sy
ect = mat_mlh.sigeps['ect']['sig'][:, 0:2] / mat_mlh.sy
ax.scatter(stx[1:, 0], stx[1:, 1], s=s, c='r', edgecolors='#cc0000')
ax.scatter(sty[1:, 0], sty[1:, 1], s=s, c='b', edgecolors='#0000cc')
ax.scatter(et2[1:, 0], et2[1:, 1], s=s, c='#808080', edgecolors='k')
ax.scatter(ect[1:, 0], ect[1:, 1], s=s, c='m', edgecolors='#cc00cc')

# plot flow stresses from reference material
s = 40
stx = mat_h.sigeps['stx']['sig'][:, 0:2] / mat_h.sy
sty = mat_h.sigeps['sty']['sig'][:, 0:2] / mat_h.sy
et2 = mat_h.sigeps['et2']['sig'][:, 0:2] / mat_h.sy
ect = mat_h.sigeps['ect']['sig'][:, 0:2] / mat_h.sy
ax.scatter(stx[1:, 0], stx[1:, 1], s=s, c='y', edgecolors='#cc0000')
ax.scatter(sty[1:, 0], sty[1:, 1], s=s, c='y', edgecolors='#0000cc')
ax.scatter(et2[1:, 0], et2[1:, 1], s=s, c='y', edgecolors='k')
ax.scatter(ect[1:, 0], ect[1:, 1], s=s, c='y', edgecolors='#cc00cc')
plt.show()

print('===================================================')
print('=== Combined FEA with reference and ML material ===')
print('===================================================')
# setup material definition for soft elastic square-shaped inclusion in
# elastic-plastic material (lhs: analytic Hill flow rule, rhs: ML flow rule)
# define soft elastic material for inclusion
mat_el = FE.Material(num=3)
mat_el.elasticity(E=1.e3, nu=0.27)
# define array for geometrical arrangement
NX = NY = 12
el = np.ones((NX, NY))  # field with material 1: reference
NX2 = int(NX / 2)
el[NX2:NX, 0:NY] = 2  # rhs for material 2: ML flow rule
NXi1 = int(NX / 3)
NXi2 = 2 * NXi1
NYi1 = int(NY / 3)
NYi2 = 2 * NYi1
el[NXi1:NXi2, NYi1:NYi2] = 3  # center material 3: elastic

# create FE model
fem = FE.Model(dim=2, planestress=False)
fem.geom(sect=3, LX=4., LY=4.)  # define geometry with two sections
fem.assign([mat_h, mat_mlh, mat_el])  # define sections for reference, ML and elastic material
fem.bcbot(0., 'disp')  # fixed bottom layer
fem.bcleft(0., 'force')  # free lateral edges
fem.bcright(0., 'force')
fem.bctop(0.002 * fem.leny, 'disp')  # apply displacement at top nodes (uniax y-stress)
fem.mesh(elmts=el, NX=NX, NY=NY)
# fix lateral displacements of corner node to prevent rigid body motion
hh = [no in fem.nobot for no in fem.noleft]
noc = np.nonzero(hh)[0]  # find corner node
fem.bcnode(noc, 0., 'disp', 'x')  # fix lateral displacement
fem.solve()

# plot results
fem.plot('mat', shownodes=False, mag=0)
fem.plot('stress2', shownodes=False, showmesh=False, mag=10)
fem.plot('seq', shownodes=False, showmesh=False, mag=10)
print('Note: Different definitions of SEQ for Hill and J2.')
fem.plot('peeq', shownodes=False, showmesh=False, mag=10)
