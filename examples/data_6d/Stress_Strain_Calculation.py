"""
Use trained ML yield function tos calculate the elastic-plastic behavior of 
the material in a strain-controlled loading.
Authors:  Ronak Shoghi, Alexander Hartmaier
ICAMS/Ruhr University Bochum, Germany
October 2023
"""

import pylabfea as FE

# Import Data
db = FE.Data("Data_Base_Updated_Final_Rotated_Train.JSON", wh_data=True)

# db.plot_yield_locus(db=db, mat_data=db.mat_data, active='flow_stress')
print(f'Successfully imported data for {db.mat_data["Nlc"]} load cases')
mat_ml = FE.Material(db.mat_data['Name'], num=1)  # define material
mat_ml.from_data(db.mat_data)  # data-based definition of material

# Train SVC with data and plot resulting stress-strain curves for 
# simple load cases
mat_ml.train_SVC(C=2, gamma=0.1, Fe=0.7, Ce=0.9, Nseq=1,
                 gridsearch=False, plot=False)

print("\nCalculating stress-strain data. I'll be back!")
mat_ml.calc_properties(eps=0.02, sigeps=True)
mat_ml.plot_stress_strain()
