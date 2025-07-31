import pytest
import pylabfea as FE
import numpy as np
import urllib.request
import os
import time
import glob


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

    assert np.abs(fem.glob['sig'][5] - 77.53778881971623) < 6E-4
    assert np.abs(fem.element[3].epl[5] - 0.003942707316047761) < 1E-7
    assert np.abs(fem.element[3].sig[1] - 43.9060552472426) < 5E-3


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
    mat_ml2.train_SVC(C=C, gamma=gamma, mat_ref=mat_J2, Nlc=150,
                      Nseq=25, Fe=0.1, Ce=0.99)
    mat_ml2.calc_properties(verb=False, eps=0.01, sigeps=True)

    # analyze training result
    loc = sy
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

    assert np.abs(mae < 7.)
    assert np.abs(mat_ml2.propJ2['et2']['ys'] - 60.5) < 1.0
    assert np.abs(mat_ml2.propJ2['ect']['peeq'][-1] - 0.00898749114723422) < 2E-6


def test_ml_data():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/AHartmaier/pyLabFEA/master/examples/Train_CPFEM/Data_Random_Texture.json",
        "data.json")
    time.sleep(20)
    db = FE.Data("data.json",
                 epl_crit=2.e-3, epl_start=1.e-3, epl_max=0.03,
                 depl=1.e-3,
                 wh_data=True)
    os.remove("data.json")
    mat_ml = FE.Material(db.mat_data['Name'], num=1)  # define material
    mat_ml.from_data(db.mat_data)  # data-based definition of material
    mat_ml.train_SVC(C=4, gamma=0.5, Fe=0.7, Ce=0.9, Nseq=2, plot=False)  # Train SVC with data

    assert 'Us_A2B2C2D2E2F2_36e6f_5e411_Tx_Rnd' in db.lc_data.keys()
    assert np.isclose(db.mat_data['sy_av'], 49.008502278682954)
    assert np.isclose(mat_ml.CV[0, 0], 204130.19078123142)
    # assert np.abs(len(mat_ml.svm_yf.support_vectors_) - 2853) < 10  # JS: with new std_scaler 3764 SVs. Old: 2093
    sig = db.lc_data['Us_A2B2C2D2E2F2_36e6f_5e411_Tx_Rnd']['Stress'][180]
    epl = db.lc_data['Us_A2B2C2D2E2F2_36e6f_5e411_Tx_Rnd']['Strain_Plastic'][180]
    vyf = mat_ml.ML_full_yf(sig=sig, epl=epl)
    assert vyf + (-2.9777289669532507) < 1.e-3  # JS: with new std_scaler -2.8355836068514293. Old: 3.6322538456276874


def test_texture():
    # 0) Set variables
    path_db = "examples/Texture/Data_CPFFT"
    wh_data = False
    
    # 1) Import Data from Micromechanical Simulations
    res_dirs_list = glob.glob(os.path.join(path_db, "*"))
    
    # 1.2) Create FE Data objects in a loop
    db_dict = {}
    for res_dir in res_dirs_list:
        if 'success.csv' in res_dir:
            continue
        try:
            db = FE.Data("Data_Base.json", path_data=res_dir, wh_data=wh_data, mode='JS', tx_data=True)
            db_dict[db.mat_data['tx_key']] = db
        except FileNotFoundError:
            print(f"{res_dir} contains no Data_Base.json")

    # 2) Create Material from the list of DB
    db_list = list(db_dict.values())
    mat_ml = FE.Material(db_list[0].mat_data['tx_name'], num=1)
    mat_ml.from_data([data_obj.mat_data for data_obj in db_list])

    # 3) Train SVC
    train_sc, test_sc = mat_ml.train_SVC(C=10, gamma=1, Fe=0.8, Ce=0.95, Nseq=2, gridsearch=False, plot=False)

    # 4) Check train score
    assert np.abs(train_sc - 99.93506493506493) < 0.3
