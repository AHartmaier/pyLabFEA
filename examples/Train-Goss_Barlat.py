#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ML flow rules to data sets for random and Goss textures

Created on Tue Jan  5 16:07:44 2021

@author: Alexander Hartmaier
"""

import pylabfea as FE
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
print('pyLabFEA version', FE.__version__)

def find_yloc(x, sig, mat):
    # Expand unit stresses 'sig' by factor 'x' and calculate yield function
    return mat.calc_seq(sig*x[:,None]) - mat.sy

# define Barlat material for Goss texture (RVE data, combined data set)
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

#calculate reference yield stresses for uniaxial, equi-biaxial and pure shear load cases
sunit =np.zeros((5,6))
sunit[0,0] = 1.
sunit[1,1] = 1.
sunit[2,2] = 1.
sunit[3,0] = 1.
sunit[3,1] = 1.
sunit[4,0] =  1./np.sqrt(3.)
sunit[4,1] = -1./np.sqrt(3.)
x1 = fsolve(find_yloc, np.ones(5) * mat_GB.sy, args=(sunit, mat_GB), xtol=1.e-5)
sy_ref = sunit * x1[:, None]
seq_ref = FE.seq_J2(sy_ref)

# define material as basis for ML flow rule
C=15.
gamma=2.5
nbase='ML-Goss-Barlat'
name = '{0}_C{1}_G{2}'.format(nbase,int(C),int(gamma*10))
data_GS = FE.Data(sig, None, name="Goss-Barlat", sdim=6, mirror=False)
mat_mlGB = FE.Material(name)  # define material
mat_mlGB.from_data(data_GS.mat_param)  # data-based defininition of material

print('Comparison of basic materials parameters:')
print('Youngs modulus: Ref={}MPa, ML={}MPa'.format(mat_GB.E, mat_mlGB.E))
print('Poisson ratio: ref={}, ML={}'.format(mat_GB.nu, mat_mlGB.nu))
print('Yield strength: Ref={}MPa, ML={}MPa'.format(mat_GB.sy, mat_mlGB.sy))

# train SVC with data generated from Barlat model for material with Goss texture
mat_mlGB.train_SVC(C=C, gamma=gamma)
sc = FE.s_cyl(mat_mlGB.msparam[0]['sig_yld'][0])
mat_mlGB.polar_plot_yl(data=sc, dname='training data', cmat=[mat_GB], arrow=True)
mat_mlGB.export_MLparam(__file__, path='../models/')

#analyze support vectors to plot them in stress space
sv = mat_mlGB.svm_yf.support_vectors_ * mat_mlGB.scale_seq
Nsv = len(sv)
sc = FE.s_cyl(sv)
yf = mat_mlGB.calc_yf(sv, pred=True)
print("ML material with {} support vectors, C={}, gamma={}, stress dimensions={}"\
    .format(Nsv,mat_mlGB.C_yf,mat_mlGB.gam_yf,mat_mlGB.sdim))
mat_mlGB.polar_plot_yl(data=sc, dname='support vectors', cmat=[mat_GB], arrow=True)

#create plot of trained yield function in cylindrical stress space
print('Plot of trained SVM classification with test data in 2D cylindrical stress space')
#create mesh in stress space
ngrid = 50
xx, yy = np.meshgrid(np.linspace(-1, 1, ngrid),np.linspace(0, 2, ngrid))
yy *= mat_mlGB.scale_seq
xx *= np.pi
hh = np.c_[yy.ravel(),xx.ravel()]
Z = mat_mlGB.calc_yf(FE.sp_cart(hh))  # value of yield function for every grid point
fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
line = mat_mlGB.plot_data(Z, ax, xx, yy, c='black')
pts  = ax.scatter(sc[:,1], sc[:,0], s=20, c=yf, cmap=plt.cm.Paired, edgecolors='k') # plot support vectors
ax.set_xlabel(r'$\theta$ (rad)', fontsize=20)
ax.set_ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=20)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
plt.legend([line[0], pts], ['ML yield locus', 'support vectors'],loc='lower right')
plt.ylim(0.,2.*mat_mlGB.sy)
#fig.savefig('SVM-yield-fct.pdf', format='pdf', dpi=300)
plt.show()

#analyze training result
loc = 40
scale = 10
size = 200
offset = 5
X1 = np.random.normal(loc=loc, scale=scale, size=int(size/4))
X2 = np.random.normal(loc=(loc-offset), scale=scale, size=int(size/2))
X3 = np.random.normal(loc=(loc+offset), scale=scale, size=int(size/4))
X = np.concatenate((X1, X2, X3))
sunittest = FE.load_cases(number_3d=0, number_6d=len(X))
sig_test = sunittest * X[:, None]
yf_ml = mat_mlGB.calc_yf(sig_test)
yf_GB = mat_GB.calc_yf(sig_test)
FE.training_score(yf_GB,yf_ml)

# stress strain curves
print('Calculating stress-strain data ...')
mat_mlGB.calc_properties(verb=False, eps=0.01, sigeps=True)
mat_mlGB.plot_stress_strain()
mat_mlGB.pckl(path='../materials/')

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

