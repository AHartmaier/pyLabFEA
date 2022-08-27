#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 2022

@author: Ronak Shoghi
"""
import pylabfea as FE


db = FE.Data("Data_Base_Final_New.json")
print(f'Successfully imported data for {db.mat_data["Nlc"]} load cases')
mat_ml = FE.Material(db.mat_data['Name'], num=1)  # define material
mat_ml.from_data(db.mat_data)  # data-based definition of material
