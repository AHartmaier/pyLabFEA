"""
Use trained ML yield function tos calculate the elastic-plastic behavior of 
the material in a strain-controlled loading.
Authors:  Ronak Shoghi, Alexander Hartmaier
ICAMS/Ruhr University Bochum, Germany
October 2023
"""

import pylabfea as FE

# Import Data
db = FE.Data("Data_Base_Updated_Final_Rotated_Train.json", wh_data=True)

# db.plot_yield_locus(db=db, mat_data=db.mat_data, active='flow_stress')
print(f'Successfully imported data for {db.mat_data["Nlc"]} load cases')
mat_ml = FE.Material(db.mat_data['Name'], num=1)  # define material
mat_ml.from_data(db.mat_data)  # data-based definition of material

# Train SVC with data and plot yield locus
mat_ml.train_SVC(C=2, gamma=0.1, Fe=0.7, Ce=0.9, Nseq=2,
                 gridsearch=False, plot=False)
print(f'Training completed. Generated {len(mat_ml.svm_yf.support_vectors_)}' +
      ' support vectors.')
sc = FE.sig_princ2cyl(mat_ml.msparam[0]['sig_ideal'])
mat_ml.polar_plot_yl(data=sc, dname='training data', arrow=True)
# export ML parameters for use in UMAT
mat_ml.export_MLparam(__file__, file='abq_data-C2-g01-Fe7-Ce9-N2', path='./')

# Calculate and plot resulting stress-strain curves for 
# simple load cases
print("\nCalculating stress-strain data. I'll be back!")
mat_ml.calc_properties(eps=0.03, sigeps=True)
mat_ml.plot_stress_strain()
