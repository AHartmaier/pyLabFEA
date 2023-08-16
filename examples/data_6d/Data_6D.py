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
import matplotlib.pyplot as plt
from sklearn.svm import NuSVC
from pandas.plotting import parallel_coordinates
import plotly.express as px
import seaborn as sns
import matplotlib.lines as mlines
import src.pylabfea.training as CTD

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

#Training
db = FE.Data("Data_Base_Updated_Final_Rotated_Train.json", Work_Hardening=True)
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
sig_tot, epl_tot, yf_ref = CTD.Create_Test_Sig(Json = "Data_Base_Updated_Final_Rotated_Test.json")
yf_ml = mat_ml.calc_yf(sig_tot, epl_tot, pred = False)
Results = CTD.training_score(yf_ref, yf_ml)

# create plot of trained yield function in cylindrical stress space
print('Plot of trained SVM classification with test data in 2D cylindrical stress space')

#create mesh in stress space
ngrid = 100
xx, yy = np.meshgrid(np.linspace(-1, 1, ngrid), np.linspace(0, 2, ngrid))
yy *= mat_ml.scale_seq
xx *= np.pi
hh = np.c_[yy.ravel(), xx.ravel()]
Cart_hh = FE.sp_cart(hh)
zeros_array = np.zeros((10000, 3))
Cart_hh_6D = np.hstack((Cart_hh, zeros_array))
grad_hh = mat_ml.calc_fgrad(Cart_hh_6D)
norm_6d = np.linalg.norm(grad_hh)
normalized_grad_hh = grad_hh / norm_6d
Z = mat_ml.calc_yf(sig=Cart_hh_6D, epl=normalized_grad_hh * 0, pred=False) # value of yield function for every grid point
Z2 = mat_ml.calc_yf(sig=Cart_hh_6D, epl=normalized_grad_hh * 0.5, pred=False) # value of yield function for every grid point
Z3 = mat_ml.calc_yf(sig=Cart_hh_6D, epl=normalized_grad_hh * 1, pred=False) # value of yield function for every grid point
Z4 = mat_ml.calc_yf(sig=Cart_hh_6D, epl=normalized_grad_hh * 1.5, pred=False) # value of yield function for every grid point
Z5 = mat_ml.calc_yf(sig=Cart_hh_6D, epl=normalized_grad_hh * 2, pred=False) # value of yield function for every grid point

# Z2 = mat_ref.calc_yf(sig=Cart_hh_6D,epl=normalized_grad_hh_mult,  pred=False)
colors = sns.color_palette("husl", 6)  # Create a color palette with 6 colors
colors_hex = [rgb_to_hex(color) for color in colors]

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.grid(True)  # Turn the grid on

line = mat_ml.plot_data(Z, ax, xx, yy, c=colors_hex[0])
line2 = mat_ml.plot_data(Z2, ax, xx, yy, c=colors_hex[1])
line3 = mat_ml.plot_data(Z3, ax, xx, yy, c=colors_hex[2])
line4 = mat_ml.plot_data(Z4, ax, xx, yy, c=colors_hex[3])
line5 = mat_ml.plot_data(Z5, ax, xx, yy, c=colors_hex[4])

# Create custom legend handles
handle1 = mlines.Line2D([], [], color=colors_hex[0], label='0.00 Eq. Plastic Strain')
handle2 = mlines.Line2D([], [], color=colors_hex[1], label='0.005 Eq. Plastic Strain')
handle3 = mlines.Line2D([], [], color=colors_hex[2], label='0.01 Eq. Plastic Strain')
handle4 = mlines.Line2D([], [], color=colors_hex[3], label='0.015 Eq. Plastic Strain')
handle5 = mlines.Line2D([], [], color=colors_hex[4], label='0.02 Eq. Plastic Strain')
ax.legend(handles=[handle1, handle2, handle3, handle4, handle5])

plt.show()

