#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 2022

@author: Ronak Shoghi
"""
import pylabfea as FE
import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import json
import os
import pickle
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import NuSVC
from pandas.plotting import parallel_coordinates
import plotly.express as px

db = FE.Data("Data_Base_Updated_Final_Rotated.json", Work_Hardening=True)
mat_ref = FE.Material(name="reference") # define reference material, J2 plasticity, linear w.h.
mat_ref.elasticity(E=db.mat_data['E_av'], nu=db.mat_data['nu_av'])             # identic elastic properties as mat1
mat_ref.plasticity(sy=db.mat_data['sy_av'], khard= 4.5e3)        # same yield strength as mat1 and mat2, high w.h. coefficient 4.5e3)
mat_ref.calc_properties(verb=False, eps=0.03, sigeps=True)

# db.plot_yield_locus(db =db, mat_data= db.mat_data, active ='flow_stress')
print(f'Successfully imported data for {db.mat_data["Nlc"]} load cases')
mat_ml=FE.Material(db.mat_data['Name'], num = 1)  # define material
mat_ml.from_data(db.mat_data)  # data-based definition of material

#train SVC with data from all microstructures
mat_ml.train_SVC(C = 30, gamma = 0.1, Fe=0.75, Ce=0.9, Nseq= 1, gridsearch= False, plot = False)

#Testing


# create plot of trained yield function in cylindrical stress space
print('Plot of trained SVM classification with test data in 2D cylindrical stress space')

#create mesh in stress space
ngrid = 100
xx, yy = np.meshgrid(np.linspace(-1, 1, ngrid), np.linspace(0, 2, ngrid))
yy *= mat_ml.scale_seq
xx *= np.pi
hh = np.c_[yy.ravel(), xx.ravel()]
Cart_hh = FE.sp_cart(hh)
print(Cart_hh)
zeros_array = np.zeros((10000, 3))
Cart_hh_6D = np.hstack((Cart_hh, zeros_array))
grad_hh = mat_ml.calc_fgrad(Cart_hh_6D)
norm_6d = np.linalg.norm(grad_hh)
normalized_grad_hh = grad_hh / norm_6d
normalized_grad_hh_mult = normalized_grad_hh * 0.002
Z = mat_ml.calc_yf(sig=Cart_hh_6D, epl=normalized_grad_hh_mult, pred=False) # value of yield function for every grid point
# Z2 = mat_ref.calc_yf(sig=Cart_hh_6D,epl=normalized_grad_hh_mult,  pred=False)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
line = mat_ml.plot_data(Z, ax, xx, yy, c='black')
plt.show()

