#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ML flow rules to reference material with isotropic J2 plasticity

Author: Alexander Hartmaier
ICAMS / Ruhr University Bochum, Germany
January 2025

Published as part of pyLabFEA package under GNU GPL v3 license
"""

import pylabfea as FE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

print('pyLabFEA version', FE.__version__)

# define J2 model as reference
E = 200000.
nu = 0.3
sy = 60.
mat_J2 = FE.Material(name='J2-reference')
mat_J2.elasticity(E=E, nu=nu)
mat_J2.plasticity(sy=sy, sdim=6)
mat_J2.calc_properties(eps=0.01, min_step=10, sigeps=True)

# define material as basis for ML flow rule
C = 1.
gamma = 1.
Ce = 0.95
Fe = 0.7
Nseq = 10
Nlc = 200
nbase = 'ML-J2'
name = f'{nbase}_C{C:3.1f}_G{gamma:3.1f}'
mat_ml = FE.Material(name)  # define material
mat_ml.dev_only = True
mat_ml.train_SVC(C=C, gamma=gamma, mat_ref=mat_J2, Nlc=Nlc,
                 gridsearch=False,
                 Ce=Ce, Fe=Fe, Nseq=Nseq)
mat_ml.export_MLparam(__file__, path='./')

# analyze support vectors to plot them in stress space
sv = mat_ml.svm_yf.support_vectors_ * mat_ml.scale_seq
Nsv = len(sv)
sc = FE.sig_princ2cyl(sv)
yf = mat_ml.calc_yf(sv, pred=True)
print("ML material with {} support vectors, C={}, gamma={}, stress dimensions={}"
      .format(Nsv, mat_ml.C_yf, mat_ml.gam_yf, mat_ml.sdim))
mat_ml.polar_plot_yl(data=sc, dname='support vectors', cmat=[mat_J2], arrow=True)

# create plot of trained yield function in cylindrical stress space
print('Plot of trained SVM classification with test data in 2D cylindrical stress space')
# create mesh in stress space
ngrid = 50
xx, yy = np.meshgrid(np.linspace(-1, 1, ngrid), np.linspace(0, 2, ngrid))
yy *= mat_ml.scale_seq
xx *= np.pi
hh = np.c_[yy.ravel(), xx.ravel()]
st = FE.sig_cyl2princ(hh)
Z = mat_ml.calc_yf(st)  # value of yield function for every grid point
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
cont = mat_ml.plot_data(Z, ax, xx, yy, c='black')
line = Line2D([0], [0], color=cont.colors, lw=2)
pts = ax.scatter(sc[:, 1], sc[:, 0], s=20, c=yf, cmap=plt.cm.Paired, edgecolors='k')  # plot support vectors
ax.set_xlabel(r'$\theta$ (rad)', fontsize=20)
ax.set_ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=20)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
plt.legend([line, pts], ['ML yield locus', 'support vectors'], loc='lower right')
plt.ylim(0., 2. * mat_ml.sy)
# fig.savefig('SVM-yield-fct.pdf', format='pdf', dpi=300)
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
yf_ml = mat_ml.calc_yf(sig_test)
yf_J2 = mat_J2.calc_yf(sig_test)
FE.training_score(yf_J2, yf_ml)

# stress strain curves
print('Calculating stress-strain data ...')
mat_ml.calc_properties(verb=False, eps=0.01, sigeps=True)
mat_ml.plot_stress_strain()
mat_J2.plot_stress_strain()
mat_ml.pckl(path='./')

# plot yield locus with stress states
s = 80
ax = mat_ml.plot_yield_locus(xstart=-1.8, xend=1.8, ref_mat=mat_J2, Nmesh=200)
stx = mat_ml.sigeps['stx']['sig'][:, 0:2] / mat_ml.sy
sty = mat_ml.sigeps['sty']['sig'][:, 0:2] / mat_ml.sy
et2 = mat_ml.sigeps['et2']['sig'][:, 0:2] / mat_ml.sy
ect = mat_ml.sigeps['ect']['sig'][:, 0:2] / mat_ml.sy
ax.scatter(stx[1:, 0], stx[1:, 1], s=s, c='r', edgecolors='#cc0000')
ax.scatter(sty[1:, 0], sty[1:, 1], s=s, c='b', edgecolors='#0000cc')
ax.scatter(et2[1:, 0], et2[1:, 1], s=s, c='#808080', edgecolors='k')
ax.scatter(ect[1:, 0], ect[1:, 1], s=s, c='m', edgecolors='#cc00cc')

s = 40
stx = mat_J2.sigeps['stx']['sig'][:, 0:2] / mat_J2.sy
sty = mat_J2.sigeps['sty']['sig'][:, 0:2] / mat_J2.sy
et2 = mat_J2.sigeps['et2']['sig'][:, 0:2] / mat_J2.sy
ect = mat_J2.sigeps['ect']['sig'][:, 0:2] / mat_J2.sy
ax.scatter(stx[1:, 0], stx[1:, 1], s=s, c='y', edgecolors='#cc0000')
ax.scatter(sty[1:, 0], sty[1:, 1], s=s, c='y', edgecolors='#0000cc')
ax.scatter(et2[1:, 0], et2[1:, 1], s=s, c='y', edgecolors='k')
ax.scatter(ect[1:, 0], ect[1:, 1], s=s, c='y', edgecolors='#cc00cc')
plt.show()
