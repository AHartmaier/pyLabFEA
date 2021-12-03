import pytest
import os
import pylabfea as FE
import numpy as np

def test_ml_plasticity():
    #check if nonlinear stress-strain data is correct
    assert np.abs(mat_ml.propJ2['stx']['ys'] - 146.60448659713813) < 1E-5
    assert np.abs(mat_ml.propJ2['sty']['seq'][-1] - 162.32017645) < 1E-5
    assert np.abs(mat_ml.propJ2['sty']['peeq'][-1] - 0.04919651) < 1E-7
    assert np.abs(mat_ml.propJ2['et2']['ys'] - 137.0452401057186) < 1E-5
    assert np.abs(mat_ml.propJ2['ect']['peeq'][-1] - 0.04549045) < 1E-7
    assert np.abs(mat_ml.propJ2['ect']['seq'][-1] - 160.99255742) < 1E-5
    
def ml_shear():
    assert np.abs(fem.glob['sig'][5] - 32.83846715429141) < 1E-5
    assert np.abs(fem.element[3].epl[5] - 0.008614134401382851) < 1E-7
    assert np.abs(fem.element[3].sig[1] - 20.194800131408773) < 1E-5
    
 

#define two elastic-plastic materials with identical yield strength and elastic properties
E=200.e3
nu=0.3
sy = 150.
#anistropic Hill-material as reference
mat_h = FE.Material(name='anisotropic Hill')
mat_h.elasticity(E=E, nu=nu)
mat_h.plasticity(sy=sy, hill=[0.7,1.,1.4], drucker=0., khard=0., sdim=3)
#isotropic material for ML flow rule
mat_ml = FE.Material(name='ML flow rule')
mat_ml.elasticity(E=E, nu=nu)
mat_ml.plasticity(sy=sy, sdim=3)
#Training and testing data for ML yield function, based on reference Material mat_h
ndata = 36
ntest = np.maximum(20, int(ndata/10))
x_train, y_train = mat_ml.create_sig_data(ndata, mat_ref=mat_h, extend=True)

#initialize and train SVC as ML yield function
#implement ML flow rule into mat_ml
train_sc, test_sc = mat_ml.setup_yf_SVM_3D(x_train, y_train, C=10, gamma=4., fs=0.3)
mat_ml.calc_properties(eps=0.05, sigeps=True, min_step=12)

#test simple shear
hill = [1.4, 1., 0.7, 1.2, .8, 1.]
mat_h = FE.Material(name='Hill-shear')
mat_h.elasticity(E=E, nu=nu)
mat_h.plasticity(sy=sy, hill=hill, sdim=6)

mat_mlh = FE.Material('Hill-ML')  # define ML material
mat_mlh.train_SVC(C=15, gamma=2.5, mat_ref=mat_h, Nlc=300) # train ML flowrule from reference material

# do FEA
fem=FE.Model(dim=2,planestress=True)
fem.geom([2], LY=2.)  # define sections in absolute lengths
fem.assign([mat_mlh])  # create a model with trained ML-flow rule and reference material 
fem.bcbot(0., bctype='disp', bcdir='y')
fem.bcbot(0., bctype='disp', bcdir='x')
fem.bcleft(0., bctype='force')
fem.bcright(0., bctype='force')
fem.bctop(0.01*fem.leny, bctype='disp', bcdir='x')  # apply shear displacement at top nodes 
fem.bctop(0.,bctype='disp', bcdir='y')
fem.mesh(NX=6, NY=3)
fem.solve()
fem.calc_global()