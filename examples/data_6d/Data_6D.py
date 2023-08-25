#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 2022
Updated on Fri Aug 25 2023
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
import random
import math
from matplotlib.lines import Line2D
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

# Import Data
db = FE.Data("Data_Base_Updated_Final_Rotated_Train.json", Work_Hardening=True)
mat_ref = FE.Material(name="reference") # define reference material, J2 plasticity, linear w.h.
mat_ref.elasticity(E=db.mat_data['E_av'], nu=db.mat_data['nu_av'])
mat_ref.plasticity(sy=db.mat_data['sy_av'], khard= 4.5e3)
mat_ref.calc_properties(verb=False, eps=0.03, sigeps=True)

# db.plot_yield_locus(db =db, mat_data= db.mat_data, active ='flow_stress')
print(f'Successfully imported data for {db.mat_data["Nlc"]} load cases')
mat_ml = FE.Material(db.mat_data['Name'], num = 1)  # define material
mat_ml.from_data(db.mat_data)  # data-based definition of material

#Train SVC with data from all microstructures
mat_ml.train_SVC(C = 1, gamma = 0.5, Fe=0.7, Ce=0.9, Nseq= 2, gridsearch= False, plot = False)
print(len(mat_ml.svm_yf.support_vectors_))

# Testing
sig_tot, epl_tot, yf_ref = CTD.Create_Test_Sig(Json = "Data_Base_Updated_Final_Rotated_Test.json")
yf_ml = mat_ml.calc_yf(sig_tot, epl_tot, pred = False)
Results = CTD.training_score(yf_ref, yf_ml)

# Plot Hardening levels over a meshed space
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
Z6 = mat_ml.calc_yf(sig=Cart_hh_6D, epl=normalized_grad_hh * 3, pred=False)
colors = sns.color_palette("plasma", 6)  # Create a color palette with 6 colors
colors_hex = [rgb_to_hex(color) for color in colors]
fig = plt.figure(figsize=(4.2, 4.2))
ax = fig.add_subplot(111, projection='polar')
ax.grid(True)
line = mat_ml.plot_data(Z, ax, xx, yy, c=colors_hex[0])
line2 = mat_ml.plot_data(Z2, ax, xx, yy, c=colors_hex[1])
line3 = mat_ml.plot_data(Z3, ax, xx, yy, c=colors_hex[2])
line4 = mat_ml.plot_data(Z4, ax, xx, yy, c=colors_hex[3])
line5 = mat_ml.plot_data(Z5, ax, xx, yy, c=colors_hex[4])
line6 = mat_ml.plot_data(Z6, ax, xx, yy, c=colors_hex[5])
fig.savefig('Hardening_Levels.png', dpi=300)
handle1 = mlines.Line2D([], [], color=colors_hex[0], label='Equivalent Plastic Strain : 0 ')
handle2 = mlines.Line2D([], [], color=colors_hex[1], label='Equivalent Plastic Strain : 0.5% ')
handle3 = mlines.Line2D([], [], color=colors_hex[2], label='Equivalent Plastic Strain : 1% ')
handle4 = mlines.Line2D([], [], color=colors_hex[3], label='Equivalent Plastic Strain : 1.5% ')
handle5 = mlines.Line2D([], [], color=colors_hex[4], label='Equivalent Plastic Strain : 2% ')
handle6 = mlines.Line2D([], [], color=colors_hex[5], label='Equivalent Plastic Strain : 3% ')
fig_leg = plt.figure(figsize=(4, 4))
ax_leg = fig_leg.add_subplot(111)
ax_leg.axis('off')
ax_leg.legend(handles=[handle1, handle2, handle3, handle4, handle5, handle6], loc="center")
fig_leg.savefig('Legend.png', dpi=300)
plt.show()

# Plot initial yield locus in pi-plane with the average yield strength from data, now the average was taken manually but it can be achieved using Z = mat_ref.calc_yf(sig=Cart_hh_6D, epl=normalized_grad_hh * 0, pred=False)
Z = mat_ml.calc_yf(sig=Cart_hh_6D, epl=normalized_grad_hh * 0, pred=False)  # value of yield function for every grid point
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.6, 4.5))
linet, = mat_ml.plot_data(Z, ax, xx, yy, c='black')
linet.set_linewidth(2.2)
lineu = ax.axhline(48.81, color='#d60404', lw=2)
legend_elements = [
    Line2D([0], [0], color='black', lw=2, label='ML'),
    Line2D([0], [0], color='#d60404', lw=2, label='Data')
]
ax.set_xlabel(r'$\theta$ (rad)', fontsize=14)
ax.set_ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
plt.tight_layout()
fig.savefig('Initial_Yield_Locus.png', dpi=300)
plt.show()

# Reconstruct Stress-Strain Curve
Keys = list(db.Data_Visualization.keys())
Key = "Us_A1B2C2D1E1F2_4092b_5e411_Tx_Rnd"  #random.choice(Keys) #Select Data from database randomly
print ("Selected Key is: {}".format(Key))
Stresses = db.Data_Visualization[Key]["Stress"]
Eq_Stresses = db.Data_Visualization[Key]["Eq_Stress"]
Strains = db.Data_Visualization[Key]["Strain"]
Eq_Strains = list(db.Data_Visualization[Key]["Eq_Strain"])
Eq_Shifted_Strains = list(db.Data_Visualization[Key]["Eq_Shifted_Strain"])
Response = []
Strains_for_Response = []
Eq_Stress_Drawing = []
Eq_Strains_Drawing = []

for counter, Eq_Strain in enumerate(Eq_Strains):
    if math.isnan(Eq_Shifted_Strains[counter]):
        continue
    else:
        if Eq_Shifted_Strains[counter] < 0 or Eq_Shifted_Strains[counter] > 0.03:
            continue
        else:
            Z=mat_ml.calc_yf(sig = Stresses, epl = Strains[counter],
                             pred = False)  # value of yield function for every grid point
            for counter2, val in enumerate(Z):
                if val > 0:
                    Response.append(Eq_Stresses[counter2])
                    Strains_for_Response.append(Eq_Strains[counter])
                    break

for counter, Eq_Strain in enumerate(Eq_Strains):
    if Eq_Strains[counter] > 0.03:
        continue
    else:
        Eq_Stress_Drawing.append(Eq_Stresses[counter])
        Eq_Strains_Drawing.append(Eq_Strains[counter])

fig = plt.figure(figsize=(5.6, 4.7))

plt.scatter(Eq_Strains_Drawing, Eq_Stress_Drawing, label="Data", s=9, color='#d60404')
plt.scatter(Strains_for_Response, Response, label="ML", s=10,  color='black')
plt.axvline(x=0.002, color='black', linestyle='--', ymax=50.4/plt.ylim()[1])
text_x_position = 0.002 + 0.0005
text_y_position = 46
text_size = 12
plt.text(text_x_position, text_y_position, '0.2% Equivalent Plastic Strain', color='black', ha='left', fontsize=text_size)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel(xlabel="Equivalent Plastic Strain", fontsize=14)
plt.ylabel(ylabel="Equivalent Stress (MPa)", fontsize=14)
legend_font_size = 12
legend = plt.legend(fontsize=legend_font_size)
legend.legendHandles[0]._sizes = [50]
legend.legendHandles[1]._sizes = [50]
plt.tight_layout()
fig.savefig('Reconstructed_Stress_Strain_Curve.png', dpi=300)
plt.show()


