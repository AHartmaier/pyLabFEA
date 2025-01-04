#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ML flow rule to reference material with Goss texture
reference material described by parameters for Barlat Yld2004-18p model
application of trained ML flow rule in FEA

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
font = {'size': 16}
plt.rc('font', **font)


def find_yloc(x, sig, mat):
    # Expand unit stresses 'sig' by factor 'x' and calculate yield function
    return mat.calc_seq(sig * x[:, None]) - mat.sy


# define Barlat material for Goss texture
# fitted to micromechanical data
bp = [0.81766901, -0.36431565, 0.31238124, 0.84321164, -0.01812166, 0.8320893, 0.35952332,
      0.08127502, 1.29314957, 1.0956107, 0.90916744, 0.27655112, 1.090482, 1.18282173,
      -0.01897814, 0.90539357, 1.88256105, 0.0127306]
mat_GB = FE.Material(name='Yld2004-18p_from_Goss')
mat_GB.elasticity(E=151220., nu=0.3)
mat_GB.plasticity(sy=46.76, barlat=bp[0:18], barlat_exp=8)
# create set of unit stresses and assess yield stresses
sunit = FE.load_cases(number_3d=100, number_6d=200)
N = len(sunit)
# calculate yield stresses from unit stresses
x1 = fsolve(find_yloc, np.ones(N) * mat_GB.sy, args=(sunit, mat_GB), xtol=1.e-5)
sig = sunit * x1[:, None]

# calculate reference yield stresses for uniaxial, equi-biaxial and pure shear load cases
sunit = np.zeros((5, 6))
sunit[0, 0] = 1.
sunit[1, 1] = 1.
sunit[2, 2] = 1.
sunit[3, 0] = 1.
sunit[3, 1] = 1.
sunit[4, 0] = 1. / np.sqrt(3.)
sunit[4, 1] = -1. / np.sqrt(3.)
x1 = fsolve(find_yloc, np.ones(5) * mat_GB.sy, args=(sunit, mat_GB), xtol=1.e-5)
sy_ref = sunit * x1[:, None]
seq_ref = FE.sig_eq_j2(sy_ref)

# define material as basis for ML flow rule
C = 3.0
gamma = 1.5
Ce = 0.99
Fe = 0.1
Nseq = 25
nbase = 'ML-Goss-Barlat'
name = name = f'{nbase}_C{C:3.1f}_G{gamma:3.1f}'
data_GS = FE.Data(sig, mat_name="Goss-Barlat", wh_data=False)
mat_mlGB = FE.Material(name, num=1)  # define material
mat_mlGB.from_data(data_GS.mat_data)  # data-based definition of material
mat_mlGB.elasticity(C11=mat_GB.C11, C12=mat_GB.C12, C44=mat_GB.C44)  # elastic properties from reference material

print('\nComparison of basic material parameters:')
print("Young's modulus: Ref={}MPa, ML={}MPa".format(mat_GB.E, mat_mlGB.E))
print("Poisson's ratio: ref={}, ML={}".format(mat_GB.nu, mat_mlGB.nu))
print('Yield strength: Ref={}MPa, ML={}MPa'.format(mat_GB.sy, mat_mlGB.sy))

# train SVC with data generated from Barlat model for material with Goss texture
mat_mlGB.train_SVC(C=C, gamma=gamma,
                   Ce=Ce, Fe=Fe, Nseq=Nseq,
                   gridsearch=False)
sc = FE.sig_princ2cyl(mat_mlGB.msparam[0]['sig_ideal'])
mat_mlGB.polar_plot_yl(data=sc, dname='training data', cmat=[mat_GB], arrow=True)
# export ML parameters for use in UMAT
mat_mlGB.export_MLparam(__file__, path='./')

# analyze support vectors to plot them in stress space
sv = mat_mlGB.svm_yf.support_vectors_ * mat_mlGB.scale_seq
Nsv = len(sv)
sc = FE.sig_princ2cyl(sv)
yf = mat_mlGB.calc_yf(sv, pred=True)
print("ML material with {} support vectors, C={}, gamma={}, stress dimensions={}"
      .format(Nsv, mat_mlGB.C_yf, mat_mlGB.gam_yf, mat_mlGB.sdim))
mat_mlGB.polar_plot_yl(data=sc, dname='support vectors', cmat=[mat_GB], arrow=True)

# create plot of trained yield function in cylindrical stress space
print('Plot of trained SVM classification with test data in 2D cylindrical stress space')
ngrid = 50
xx, yy = np.meshgrid(np.linspace(-1, 1, ngrid), np.linspace(0, 2, ngrid))
yy *= mat_mlGB.scale_seq
xx *= np.pi
hh = np.c_[yy.ravel(), xx.ravel()]
Z = mat_mlGB.calc_yf(FE.sig_cyl2princ(hh))  # value of yield function for every grid point
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
cont = mat_mlGB.plot_data(Z, ax, xx, yy, c='black')
line = Line2D([0], [0], color=cont.colors, lw=2)
pts = ax.scatter(sc[:, 1], sc[:, 0], s=20, c=yf, cmap=plt.cm.Paired, edgecolors='k')  # plot support vectors
ax.set_xlabel(r'$\theta$ (rad)', fontsize=20)
ax.set_ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=20)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
plt.legend([line, pts], ['ML yield locus', 'support vectors'], loc='lower right')
plt.ylim(0., 2. * mat_mlGB.sy)
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
yf_ml = mat_mlGB.calc_yf(sig_test)
yf_GB = mat_GB.calc_yf(sig_test)
FE.training_score(yf_GB, yf_ml)

# calculate and plot stress strain curves
print('Calculating stress-strain data ...')
mat_mlGB.calc_properties(verb=False, eps=0.01, sigeps=True)
mat_mlGB.plot_stress_strain()

# plot yield locus with stress states
s = 80
ax = mat_mlGB.plot_yield_locus(xstart=-1.8, xend=1.8, ref_mat=mat_GB, Nmesh=200)
stx = mat_mlGB.sigeps['stx']['sig'][:, 0:2] / mat_mlGB.sy
sty = mat_mlGB.sigeps['sty']['sig'][:, 0:2] / mat_mlGB.sy
et2 = mat_mlGB.sigeps['et2']['sig'][:, 0:2] / mat_mlGB.sy
ect = mat_mlGB.sigeps['ect']['sig'][:, 0:2] / mat_mlGB.sy
ax.scatter(stx[1:, 0], stx[1:, 1], s=s, c='r', edgecolors='#cc0000')
ax.scatter(sty[1:, 0], sty[1:, 1], s=s, c='b', edgecolors='#0000cc')
ax.scatter(et2[1:, 0], et2[1:, 1], s=s, c='#808080', edgecolors='k')
ax.scatter(ect[1:, 0], ect[1:, 1], s=s, c='m', edgecolors='#cc00cc')
plt.show()
plt.close('all')
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
fem.assign([mat_mlGB, mat_el])  # define sections for reference, ML and elastic material
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
