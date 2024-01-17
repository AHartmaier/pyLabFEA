import pytest
import pylabfea as FE
import numpy as np


def test_ml_plasticity():
    # define two elastic-plastic materials with identical yield strength and elastic properties
    E = 200.e3
    nu = 0.3
    sy = 150.
    # anisotropic Hill-material as reference
    mat_h = FE.Material(name='anisotropic Hill')
    mat_h.elasticity(E=E, nu=nu)
    mat_h.plasticity(sy=sy, hill=[0.7, 1., 1.4], drucker=0., khard=0., sdim=3)
    # isotropic material for ML flow rule
    mat_ml = FE.Material(name='ML flow rule')
    mat_ml.elasticity(E=E, nu=nu)
    mat_ml.plasticity(sy=sy, sdim=3)
    # Training and testing data for ML yield function, based on reference Material mat_h
    ndata = 36
    x_train, y_train = mat_ml.create_sig_data(ndata, mat_ref=mat_h, extend=True)
    # initialize and train SVC as ML yield function
    # implement ML flow rule into mat_ml
    train_sc, test_sc = mat_ml.setup_yf_SVM_3D(x_train, y_train, C=10, gamma=4., fs=0.3)
    mat_ml.calc_properties(eps=0.01, sigeps=True, min_step=12)

    # check if nonlinear stress-strain data is correct
    assert np.abs(mat_ml.propJ2['stx']['ys'] - 149.62302821433968) < 1E-5
    assert np.abs(mat_ml.propJ2['sty']['seq'][-1] - 157.25971534002542) < 1E-5
    assert np.abs(mat_ml.propJ2['ect']['peeq'][-1] - 0.00855380746615942) < 1E-7


def test_ml_shear():
    E = 200.e3
    nu = 0.3
    sy = 150.
    # test simple shear
    hill = [1.4, 1., 0.7, 1.2, .8, 1.]
    mat_h = FE.Material(name='Hill-shear')
    mat_h.elasticity(E=E, nu=nu)
    mat_h.plasticity(sy=sy, hill=hill, sdim=6)

    mat_mlh = FE.Material('Hill-ML')  # define ML material
    # train ML flowrule from reference material
    mat_mlh.train_SVC(C=2, gamma=0.5, mat_ref=mat_h, Nseq=4, Nlc=300, Fe=0.7, Ce=0.95)
    mat_mlh.dev_only = False

    # do FEA
    fem = FE.Model(dim=2, planestress=True)
    fem.geom([2], LY=2.)  # define sections in absolute lengths
    fem.assign([mat_mlh])  # create a model with trained ML-flow rule and reference material 
    fem.bcbot(0., bctype='disp', bcdir='y')
    fem.bcbot(0., bctype='disp', bcdir='x')
    fem.bcleft(0., bctype='force')
    fem.bcright(0., bctype='force')
    fem.bctop(0.006 * fem.leny, bctype='disp', bcdir='x')  # apply shear displacement at top nodes
    fem.bctop(0., bctype='disp', bcdir='y')
    fem.mesh(NX=6, NY=3)
    fem.solve()
    fem.calc_global()

    assert np.abs(fem.glob['sig'][5] - 79.6435732946163) < 1E-5
    assert np.abs(fem.element[3].epl[5] - 0.003800884140613953) < 1E-7
    assert np.abs(fem.element[3].sig[1] - 43.846140704269885) < 1E-5


def test_ml_training():
    # test generation of stress data in 6D stress space
    # define J2 model as reference
    E = 200000.
    nu = 0.3
    sy = 60.
    mat_J2 = FE.Material(name='J2-reference')
    mat_J2.elasticity(E=E, nu=nu)
    mat_J2.plasticity(sy=sy, sdim=6)

    # define material as basis for ML flow rule
    C = 15.
    gamma = 2.5
    nbase = 'ML-J2'
    name = '{0}_C{1}_G{2}'.format(nbase, int(C), int(gamma * 10))
    mat_ml2 = FE.Material(name)  # define material
    mat_ml2.dev_only = False
    mat_ml2.train_SVC(C=C, gamma=gamma, mat_ref=mat_J2, Nlc=150)
    mat_ml2.calc_properties(verb=False, eps=0.01, sigeps=True)

    # analyze training result
    loc = 40
    scale = 10
    size = 200
    offset = 5
    X1 = np.random.normal(loc=loc, scale=scale, size=int(size / 4))
    X2 = np.random.normal(loc=(loc - offset), scale=scale, size=int(size / 2))
    X3 = np.random.normal(loc=(loc + offset), scale=scale, size=int(size / 4))
    X = np.concatenate((X1, X2, X3))
    sunittest = FE.load_cases(number_3d=0, number_6d=len(X))
    sig_test = sunittest * X[:, None]
    yf_ml = mat_ml2.calc_yf(sig_test)
    yf_J2 = mat_J2.calc_yf(sig_test)
    mae, precision, Accuracy, Recall, F1Score, mcc = \
        FE.training_score(yf_J2, yf_ml, plot=False)

    assert np.abs(mae < 16.)
    assert np.abs(mat_ml2.propJ2['et2']['ys'] - 60.57834343495059) < 1E-4
    assert np.abs(mat_ml2.propJ2['ect']['peeq'][-1] - 0.008987491147924685) < 1E-7
