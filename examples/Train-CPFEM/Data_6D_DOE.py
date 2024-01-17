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

import src.pylabfea.training as CTD

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

#Training
db = FE.Data("Database_Random.json", wh_data=True)
mat_ref = FE.Material(name="reference") # define reference material, J2 plasticity, linear w.h.
mat_ref.elasticity(E=db.mat_data['E_av'], nu=db.mat_data['nu_av'])             # identic elastic properties as mat1
mat_ref.plasticity(sy=db.mat_data['sy_av'], khard= 4.5e3)        # same yield strength as mat1 and mat2, high w.h. coefficient 4.5e3)
mat_ref.calc_properties(verb=False, eps=0.005, sigeps=True)

# db.plot_yield_locus(db =db, mat_data= db.mat_data, active ='flow_stress')
print(f'Successfully imported data for {db.mat_data["Nlc"]} load cases')
mat_ml=FE.Material(db.mat_data['Name'], num = 1)  # define material
mat_ml.from_data(db.mat_data)  # data-based definition of material

Result_4 = {}
FeS = [0.65, 0.7, 0.75, 0.8]
CeS = [0.85, 0.9, 0.95, 0.99]
CS = [4]
GammaS = [0.5]
Test_Counter = 0

for Fe in FeS:
    for Ce in CeS:
        for C in CS:
            for gamma in GammaS:
                Result_File = open('Results_Grid_Search_N=1_newCriti.json')
                Result = json.load(Result_File)
                Result_File.close()
                path = os.path.dirname(__file__)
                Ns = 1
                ntrunk = '{}_C{}_G{}_Na{}'.format(mat_ml.name, int(C), int(gamma * 10), Ns)
                m_name = 'material_{}.pkl'.format(ntrunk)
                Test_Counter += 1
                print (Test_Counter)
                mat_ml.train_SVC(C = C, gamma = gamma, Fe=Fe, Ce=Ce, Nseq= Ns, gridsearch= True, plot = False)
                sig_tot, epl_tot, yf_ref=CTD.Create_Test_Sig(Json = "Data_Base_Updated_Final_Rotated_Test.json")
                yf_ml=mat_ml.calc_yf(sig_tot, epl_tot, pred = False)
                Results=CTD.training_score(yf_ref, yf_ml)
                Name = 'Test Case {}'.format(Test_Counter)
                if Name not in Result:
                    Result[Name] = {}
                Result[Name]['Fe'] = Fe
                Result[Name]['Ce'] = Ce
                Result[Name]['C'] = mat_ml.grid.best_params_["C"]
                Result[Name]['gamma'] = mat_ml.grid.best_params_["gamma"]
                Result[Name]['Precision'] = Results[1]
                Result[Name]['Accuracy'] = Results[2]
                Result[Name]['Recall'] = Results[3]
                Result[Name]['F1score'] = Results[4]
                Result[Name]['mcc'] = Results[5]
                with open('Results_gridSearchoff.json', 'w') as result_file:
                    json.dump(Result, result_file)
                    result_file.close()