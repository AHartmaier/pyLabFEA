#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyis and visualization of results from Abaqus block model
with Hill anisotropic plasticity for several load cases.


@author: Alexander Hartmaier, ICAMS / Ruhr-Universität Bochum, Germany

Version 1.0
April 2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import pylabfea as FE

#plt.locator_params(axis='x', nbins=5)

sig_names = ['S11','S22','S33','S12','S23','S13']
epl_names = ['Ep11','Ep22','Ep33','Ep12','Ep23','Ep13']
eps_names = ['E11','E22','E33','E12','E23','E13']
ubc_names = ['ux', 'uy', 'uz']
fbc_names = ['fx', 'fy', 'fz']

#read results of Abaqus simulation with ML-UMAT
adat = pd.read_csv('results/abq_ML-Hill-p1_C15_G25-res.csv', header=0, sep=';')
sig = adat[sig_names].to_numpy()
epl = np.nan_to_num(adat[epl_names].to_numpy())
eps = adat[eps_names].to_numpy()
peeq = np.nan_to_num(adat['PEEQ'].to_numpy())
mises = adat['MISES'].to_numpy()
lc  = adat[ubc_names].to_numpy()

toler = 0.01 * np.linalg.norm(lc[0])
stm = 4.e-4
stm_sqrt = 2.82843e-04
epl_max = 0.008
seq_ref = np.ones(5)*300.
seq_ref[0] *= 1.2
seq_ref[2] *= 0.8

# plot stress-strain curves
# normal and biaxial loads
s = 6
isx = np.nonzero(np.linalg.norm(lc-np.array([stm,0,0]),axis=1)<toler)[0]
plt.plot(peeq[isx], mises[isx], '-r', label=('uniax: x'))
#plt.plot(0., seq_ref[0], 'sr', markersize=s)

isy = np.nonzero(np.linalg.norm(lc-np.array([0, stm, 0]),axis=1)<toler)[0]
plt.plot(peeq[isy], mises[isy], '-b', label=('uniax: y'))
#plt.plot(0., seq_ref[1], 'sb', markersize=s)

isz = np.nonzero(np.linalg.norm(lc-np.array([0,0, stm]),axis=1)<toler)[0]
plt.plot(peeq[isz], mises[isz], '-g', label=('uniax: z'))
#plt.plot(0., seq_ref[2], 'sg', markersize=s)

ind = np.nonzero(np.linalg.norm(lc-np.array([stm_sqrt, stm_sqrt, 0.]),axis=1)<toler)[0]
plt.plot(peeq[ind], mises[ind], ':g', label=('biax (x,y)'))

ind = np.nonzero(np.linalg.norm(lc-np.array([stm_sqrt, 0, stm_sqrt]),axis=1)<toler)[0]
plt.plot(peeq[ind], mises[ind], ':b', label=('biax (x,z)'))

ind = np.nonzero(np.linalg.norm(lc-np.array([0., stm_sqrt, stm_sqrt]),axis=1)<toler)[0]
plt.plot(peeq[ind], mises[ind], ':r', label=('biax (y,z)'))

plt.xlabel('equiv. plastic strain (.)')
plt.ylabel('equiv. stress (MPa)')
plt.legend(loc='lower right')
#plt.xlim((-epl_max*0.2,epl_max))
plt.show()

# shear loads
ind = np.nonzero(np.linalg.norm(lc-np.array([-stm_sqrt, stm_sqrt, 0.]),axis=1)<toler)[0]
plt.plot(peeq[ind], mises[ind], '-r', label=('shear (-x,y)'))

ind = np.nonzero(np.linalg.norm(lc-np.array([stm_sqrt, 0, -stm_sqrt]),axis=1)<toler)[0]
plt.plot(peeq[ind], mises[ind], '-b', label=('shear (x,-z)'))

ind = np.nonzero(np.linalg.norm(lc-np.array([0., -stm_sqrt, stm_sqrt]),axis=1)<toler)[0]
plt.plot(peeq[ind], mises[ind], '-g', label=('shear (-y,z)'))

plt.legend(loc='lower right')
plt.xlabel('equiv. plastic strain (.)')
plt.ylabel('equiv. stress (MPa)')
#plt.xlim((-epl_max*0.2,epl_max))
plt.show()

# plot r-values (ratios of transverse plastic strain)
peqx = peeq[isx]
pe1 = epl[isx, 1]
pe2 = epl[isx, 2]
ind = np.nonzero(np.abs(pe2) > 1.e-8)
rvx = pe1[ind]/pe2[ind]

pe1 = epl[isy, 0]
pe2 = epl[isy, 2]
ind = np.nonzero(np.abs(pe2) > 1.e-8)
rvy = pe1[ind]/pe2[ind]

pe1 = epl[isz, 0]
pe2 = epl[isz, 1]
ind = np.nonzero(np.abs(pe2) > 1.e-8)
rvz = pe1[ind]/pe2[ind]

rv = [np.mean(rvx), np.mean(rvy), np.mean(rvz)]
labels = ['x-loading', 'y-loading', 'z-loading']
plt.bar([1, 2, 3], rv, tick_label=labels)
plt.ylabel('r-value (.)')
plt.show()

