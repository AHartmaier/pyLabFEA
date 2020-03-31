# Module Material
'''Introduces class ``Material`` that contains that attributes and methods
needed for elastic-plastic material definitions in FEA. The module Model is defined in
module pyLabFEM.

uses NumPy, MatPlotLib, sklearn and pyLabFEM

Version: 1.1 (2020-03-31)
Author: Alexander Hartmaier, ICAMS/Ruhr-University Bochum, March 2020
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)'''
import numpy as np
import pylabfea.pyLabFEM as FE
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn import svm
import sys

'=========================='
'define class for materials'
'=========================='
class Material(object):
    '''Define class for Materials including material parameters (attributes), constitutive relations (methods)
    and derived properties und various loading conditions (dictionary)

    Parameters
    ----------
    name : str
        Name of material (optional, default: 'Material')

    Attributes
    ----------
    name    : str
        Name of material
    sy      : float
        Yield strength
    ML_yf   : Boolean
        Existence of trained machine learning (ML) yield function (default: False)
    ML_grad : Boolean
        Existence of trained ML gradient (default: False)
    Tresca  : Boolean
        Indicate if Tresca equivalent stress should be used (default: False)
    msg     : dictionary
        Messages returned
    prop    : dictionary
        Derived properties under defined load paths
    propJ2  : dictionary
        Derived properties in form of J2 equivalent stress
    sigeps  : dictionary
        Data of stress-strain curves under defined load paths
    C11, C12, C44 : float
        Anisotropic elastic constants
    E, nu : float
        Isotropic elastic constants, Young's modulus and Poisson's number

    Keyword Arguments
    -----------------
    prop-propJ2 :
        Store properties of material (Hill-formulation or J2 formulation) in sub-dictonaries:
        'stx' (tensile horiz. stress), 'sty' (tensile vert. stress),
        'et2' (equibiaxial tensile strain), 'ect' (pure shear)
    stx-sty-et2-ect  : sub-dictionaries
        Store data for 'ys' (float - yield strength), seq (array - eqiv. stress),
        'eeq' (array - equiv. total strain), 'peeq' (array - equiv. plastic strain),
        'sytel' (str - line style for plot), 'name' (str - name in legend)
    sigeps :
        Store tensorial stress strain data in sub-directories;
        Contains data for 'sig' (2d-array - stress), 'eps' (2d-array - strain),
        'epl' (2d-array - plastic strain)
    '''
    'Methods'
    #elasticity: define elastic material parameters C11, C12, C44
    #plasticity: define plastic material parameter sy, khard
    #epl_dot: calculate plastic strain rate

    def __init__(self, name='Material'):
        self.sy = None  # Elasticity will be considered unless sy is set
        self.ML_yf = False # use conventional plasticity unless trained ML functions exists
        self.ML_grad = False # use conventional gradient unless ML function exists
        self.Trseca = False  # use J2 or Hill equivalent stress unless defined otherwise
        self.name = name
        self.msg = {
            'yield_fct' : None,
            'gradient'  : None
        }
        self.prop = {    # stores strength and stress-strain data along given load paths
            'stx'   : {'ys':None,'seq':None,'eeq':None,'peeq':None,'style':None,'name':None},
            'sty'   : {'ys':None,'seq':None,'eeq':None,'peeq':None,'style':None,'name':None},
            'et2'   : {'ys':None,'seq':None,'eeq':None,'peeq':None,'style':None,'name':None},
            'ect'   : {'ys':None,'seq':None,'eeq':None,'peeq':None,'style':None,'name':None}
        }
        self.propJ2 = {    # stores J2 equiv strain data along given load paths
            'stx'   : {'ys':None,'seq':None,'eeq':None,'peeq':None},
            'sty'   : {'ys':None,'seq':None,'eeq':None,'peeq':None},
            'et2'   : {'ys':None,'seq':None,'eeq':None,'peeq':None},
            'ect'   : {'ys':None,'seq':None,'eeq':None,'peeq':None}
        }
        self.sigeps = {    # calculates strength and stress strain data along given load paths
            'stx'   : {'sig':None,'eps':None,'epl':None},
            'sty'   : {'sig':None,'eps':None,'epl':None},
            'et2'   : {'sig':None,'eps':None,'epl':None},
            'ect'   : {'sig':None,'eps':None,'epl':None}
        }

    def calc_yf(self, sig, peeq=0., ana=False, pred=False):
        '''Calculate yield function

        Parameters
        ----------
        sig  : (3,), (6,) or (3,N) array
            Principal stresses (Voigt stress, princ. stress, or array of princ. stresses
        peeq : float or array
            Equivalent plastic strain (scalar or same length as sig) (optional, default: 0)
        ana  : Boolean
            Indicator if analytical solution should be used, rather than ML yield fct (optional, default: False)
        pred : Boolean
            Indicator if prediction value should be returned, rather then decision function (optional, default: False)

        Returns
        -------
        f    : flot or 1d-array
            Yield function for given stress (same length as sig)
        '''
        sh = np.shape(sig)
        if self.ML_yf and not ana:
            if sh==(3,):
                sig = np.array([sig])
                N = 1
            elif sh==(6,):
                sig = np.array([FE.Stress(sig).p])
                N = 1
            else:
                N = len(sig)
            x = np.zeros((N,2))
            x[:,0] = FE.seq_J2(sig)/self.sy - 1.
            x[:,1] = FE.polar_ang(sig)/np.pi
            if pred:
                # use prediction, returns either -1 or +1
                f = self.svm_yf.predict(x)
                self.msg['yield_fct'] = 'ML_yf-predict'
            else:
                # use smooth decision function in range [-1,+1]
                f = self.svm_yf.decision_function(x)
                self.msg['yield_fct'] = 'ML_yf-decision-fct'
            if N==1:
                f = f[0]
        else:
            if sh ==():  # eqiv. stress
                seq = sig
            elif sh==(6,):   # Voigt stress
                seq = FE.Stress(sig).seq(self)  # calculate equiv. stress
            else:  # array of princ. stresses
                seq = self.calc_seq(sig)
            f = seq - (self.sy + peeq*self.khard)
            self.msg['yield_fct'] = 'analytical'
        return f

    def ML_full_yf(self, sig, ld=None, verb=False):
        '''Calculate full ML yield function as distance of given stress to yield locus in loading direction.

        Parameters
        ----------
        sig : Voigt stress tensor
            Stress
        ld  : (3,) array
            Vector of loading direction in princ. stress space (optional)
        verb : Boolean
            Indicate whether to be verbose in text output (optional, default: False)

        Returns
        -------
        yf : float
            Full ML yield function, i.e. distance of sig to yield locus
        '''
        sp = FE.Stress(sig).p
        seq = self.calc_seq(sp)
        if seq<0.01 and ld is None:
            yf = seq - 0.85*self.sy
        else:
            if (seq>=0.01):
                ld = sp*self.sy/seq
            x0 = 1.
            if ld[0]*ld[1] < 0.:
                x0 = 0.5
            x1, infodict, ier, msg = fsolve(self.find_yloc, x0, args=ld, xtol=1.e-5, full_output=True)
            y1 = infodict['fvec']
            if np.abs(y1)<1.e-3 and x1[0]<3.:
                # zero of ML yield fct. detected at x1*sy
                yf = seq - x1[0]*self.calc_seq(ld)
            else:
                # zero of ML yield fct. not found: get conservative estimate
                yf = seq - 0.85*self.sy
                if verb:
                    print('Warning in calc_scf')
                    print('*** detection not successful. yf=', yf,', seq=', seq)
                    print('*** optimization result (x1,y1,ier,msg):', x1, y1, ier, msg)
        return yf

    def setup_yf_SVM(self, x, y_train, x_test=None, y_test=None, C=10., gamma=1., fs=0.1,
                     plot=False, cyl=False, inherit=None):
        '''Initialize and train Support Vector Classifier (SVC) as machine learning (ML) yield function. Training and test
        data (features) are accepted as either 3D principle stresses or cylindrical stresses. ML yield functions can also be inherited
        from other materials with a ML yield function. Graphical output on the trained SVC yield function is possible.


        Parameters
        ----------
        x   :  (N,2) or (N,3) array
            Training data either as Cartesian princ. stresses (N,3) or cylindrical stresses (N,2)
        cyl : Boolean
            Indicator for cylindrical stresses if x is has shape (N,3)
        y_train : 1d-array
            Result vector for training data (same size as x)
        x_test  : (N,2) or (N,3) array
            Test data either as Cartesian princ. stresses (N,3) or cylindrical stresses (N,2) (optional)
        y_test
            Result vector for test data (optional)
        C  : float
            Parameter for training of SVC (optional, default: 10)
        gamma  : float
            Parameter for kernel function of SVC (optional, default: 1)
        fs  : float
            Parameters for size of periodic continuation of training data (optional, default:0.1)
        inherit : object of class ``Material``
            Material from which trained ML function is inherited
        plot : Boolean
            Indicates if plot of decision function should be generated (optional, default: False)

        Returns
        -------
        train_sc : float
            Training score
        test_sc  : float
            test score
        '''
        'transformation of princ. stress into cyl. coordinates'
        self.gam_yf = gamma
        N = len(x)
        sh = np.shape(x)
        X_train = np.zeros((N,2))
        if sh==(N,3) and not cyl:
            'data format: princ. stresses'
            X_train[:,0] = FE.seq_J2(x)/self.sy - 1.
            X_train[:,1] = FE.polar_ang(x)/np.pi
            print('Using priciples stresses for training')
        elif sh==(N,2) or cyl:
            'data format: seq, theta values'
            X_train[:,0] = x[:,0]/self.sy - 1.
            X_train[:,1] = x[:,1]/np.pi
            print('Using cyclindrical stresses for training')
        else:
            print('*** N, sh', N, sh)
            sys.exit('Data format of training data not recognized')
        'copy left and right borders to enforce periodicity in theta'
        indr = np.nonzero(X_train[:,1]>1.-fs)
        indl = np.nonzero(X_train[:,1]<fs-1.)
        Xr = X_train[indr]
        Xl = X_train[indl]
        Xr[:,1] -= 2.  # shift angle theta
        Xl[:,1] += 2.
        Xh = np.append(Xr, Xl, axis=0)
        yh = np.append(y_train[indr], y_train[indl], axis=0)
        X_train = np.append(X_train, Xh, axis=0)
        y_train = np.append(y_train, yh, axis=0)
        'coordinate transformation for test data'
        if x_test is not None:
            Ntest = len(x_test)
            X_test = np.zeros((Ntest,2))
            if sh==(N,3):
                X_test[:,0] = FE.seq_J2(x_test)/self.sy - 1.
                X_test[:,1] = FE.polar_ang(x_test)/np.pi
            else:
                X_test[:,0] = x_test[:,0]/self.sy - 1.
                X_test[:,1] = x_test[:,1]/np.pi
        'define and fit SVC'
        self.svm_yf = svm.SVC(kernel='rbf',C=C,gamma=gamma)
        self.svm_yf.fit(X_train, y_train)
        self.ML_yf = True
        if (FE.ptol>=1.):
            FE.ptol=0.9
            print('Warning: ptol must be <1 for ML yield function, set to 0.9')
        'calculate scores'
        train_sc = 100*self.svm_yf.score(X_train, y_train)
        if x_test is None:
            test_sc = None
        else:
            test_sc  = 100*self.svm_yf.score(X_test, y_test)
        'create plot if requested'
        if plot:
            print('Plot of extended training data for SVM classification in 2D cylindrical stress space')
            xx, yy = np.meshgrid(np.linspace(-1.-fs, 1.+fs, 50),np.linspace(-1., 1., 50))
            fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
            feat = np.c_[yy.ravel(),xx.ravel()]
            Z = self.svm_yf.decision_function(feat)
            hl = self.plot_data(Z, ax, xx, yy, c='black')
            h1 = ax.scatter(X_train[:,1], X_train[:,0], s=10, c=y_train, cmap=plt.cm.Paired)
            ax.set_title('extended SVM yield function in training')
            ax.set_xlabel('$theta/\pi$')
            ax.set_ylabel('$\sigma_{eq}/\sigma_y$')
            plt.show()
        return train_sc, test_sc

    def calc_fgrad(self, sig, seq=None, ana=False):
        '''Calculate gradient to yield surface. Three different methods can be used: (i) analytical gradient to Hill-like yield
        function (default if no ML yield function exists - ML_yf=False), (ii) gradient to ML yield function (default if ML yield
        function exists - ML_yf=True; can be overwritten if ana=True), (iii) ML gradient fitted seperately from ML yield function
        (activated if ML_grad=True and ana=False)

        Parameters
        ----------
        sig : (3,) or (N,3) array
            Principal stresses
        seq : float or (N,) array
            Equivalent stresses (optional)
        ana : Boolean
            Indicator if analytical solution should be used, rather than ML yield fct (optional, default: False)

        Returns
        -------
        fgrad : (3,) or (N,3) array
            Gradient to yield surface at given position in princ. stress space
        '''
        N = len(sig)
        fgrad = np.zeros((N,3))
        sh = np.shape(sig)
        if sh==(3,):
            N = 1 # sig is vector of principle stresses
            sig = np.array([sig])
        elif sh!=(N,3):
            sys.exit('Error: Unknown format of stress in calc_fgrad')
        if self.ML_grad and not ana:
            'use SVR fitted to gradient'
            sig = sig/self.sy
            fgrad[:,0] = self.svm_grad0.predict(sig)*self.gscale[0]
            fgrad[:,1] = self.svm_grad1.predict(sig)*self.gscale[1]
            fgrad[:,2] = self.svm_grad2.predict(sig)*self.gscale[2]
            self.msg['gradient'] = 'SVR gradient'
        elif self.ML_yf and not ana:
            'use numerical gradient of SVC yield fct. in sigma space'
            'gradient of rbf-Kernel function w.r.t. theta'
            def grad_rbf(x,xp):
                hv = x-xp
                hh = np.sum(hv*hv,axis=1) # ||x-x'||^2=sum_i(x_i-x'_i)^2
                k = np.exp(-self.gam_yf*hh)
                arg = -2.*self.gam_yf*hv[:,1]
                grad = k*arg
                return grad
            'define Jacobian of coordinate transformation'
            def Jac(sig):
                global flag
                J = np.ones((3,3))
                hyd = np.sum(sig)/3.  # hydrostatic stress
                dev = sig - hyd  # deviatoric princ. stress
                vn = np.linalg.norm(dev)*np.sqrt(1.5)  #norm of stress vector
                if vn>0.1:
                    'only calculate Jacobian if sig>0'
                    dseqds = 3.*dev/vn
                    J[:,2] /= 3.
                    J[:,0] = dseqds
                    dsa = np.dot(sig,FE.a_vec)
                    dsb = np.dot(sig,FE.b_vec)
                    sc = dsa + 1j*dsb
                    z = -1j*((FE.a_vec+1j*FE.b_vec)/sc - dseqds/vn)
                    J[:,1] = np.real(z)
                return J
            N = len(sig)
            x = np.zeros((N,2))
            x[:,0] = FE.seq_J2(sig)/self.sy - 1.
            x[:,1] = FE.polar_ang(sig)/np.pi
            dc = self.svm_yf.dual_coef_[0,:]
            sv = self.svm_yf.support_vectors_
            for i in range(N):
                dKdt = np.sum(dc*grad_rbf(x[i,:],sv))
                fgrad[i,:] = Jac(sig[i,:])@np.array([1,dKdt,0])
            self.msg['gradient'] = 'gradient to ML_yf'
        else:
            h0 = self.hill[0]
            h1 = self.hill[1]
            h2 = self.hill[2]
            d3 = self.drucker/3.
            if (seq is None):
                seq = self.calc_seq(sig)
            fgrad[:,0] = ((h0+h2)*sig[:,0] - h0*sig[:,1] - h2*sig[:,2])/seq + d3
            fgrad[:,1] = ((h1+h0)*sig[:,1] - h0*sig[:,0] - h1*sig[:,2])/seq + d3
            fgrad[:,2] = ((h2+h1)*sig[:,2] - h2*sig[:,0] - h1*sig[:,1])/seq + d3
            self.msg['gradient'] = 'analytical'
        if sh==(3,):
            fgrad = fgrad[0,:]
        return fgrad

    def setup_fgrad_SVM(self, X_grad_train, y_grad_train, C=10., gamma=0.1):
        '''Inititalize and train SVM for gradient evaluation

        Parameters
        ----------
        X_grad_train : (N,3) array
        y_grad_train : (N,) array
        C     : float
            Paramater for training of Support Vector Regression (SVR) (optional, default:10)
        gamma : float
            Parameter for kernel of SVR (optional, default: 0.1)
        '''
        'define support vector regressor parameters'
        self.svm_grad0 = svm.SVR(C=C, cache_size=3000, coef0=0.0, degree=3, epsilon=0.01, gamma=gamma,
              kernel='rbf', max_iter=-1, shrinking=True, tol=0.0001, verbose=False)
        self.svm_grad1 = svm.SVR(C=C, cache_size=3000, coef0=0.0, degree=3, epsilon=0.01, gamma=gamma,
              kernel='rbf', max_iter=-1, shrinking=True, tol=0.0001, verbose=False)
        self.svm_grad2 = svm.SVR(C=C, cache_size=3000, coef0=0.0, degree=3, epsilon=0.01, gamma=gamma,
              kernel='rbf', max_iter=-1, shrinking=True, tol=0.0001, verbose=False)

        'fit SVM to training data'
        X_grad_train = X_grad_train / self.sy
        gmax = np.amax(y_grad_train, axis=0)
        gmin = np.amin(y_grad_train, axis=0)
        self.gscale = gmax - gmin
        y_grad_train0 = y_grad_train[:,0]/self.gscale[0]
        y_grad_train1 = y_grad_train[:,1]/self.gscale[1]
        y_grad_train2 = y_grad_train[:,2]/self.gscale[2]
        self.svm_grad0.fit(X_grad_train, y_grad_train0)
        self.svm_grad1.fit(X_grad_train, y_grad_train1)
        self.svm_grad2.fit(X_grad_train, y_grad_train2)
        self.ML_grad = True

    def calc_seq(self, sprinc):
        '''Calculate generalized equivalent stress from pricipal stresses;
        equivalent to J2 for isotropic flow behavior and tension compression invariance;
        Hill-type approach for anisotropic plastic yielding;
        Drucker-like approach for tension-compression assymetry

        Parameters
        ----------
        sprinc : (3,), (N,3) or (N,6) array
            Principal stress values

        Returns
        -------
        seq : float or (N,) array
            Hill-Drucker-type equivalent stress
        '''
        N = len(sprinc)
        sh = np.shape(sprinc)
        if sh==(3,):
            N = 1 # sprinc is single principle stress vector
            sprinc=np.array([sprinc])
        elif sh==(N,6):
            sig = sprinc
            sprinc=np.zeros((N,3))
            for i in range(N):
                sprinc[i,:] = FE.Stress(sig[i,:]).p
        elif sh!=(N,3):
            print('*** calc_seq: N, sh', N, sh, sys._getframe().f_back.f_code.co_name)
            sys.exit('Error: Unknown format of stress in calc_seq')
        if self.Tresca:
            seq = np.amax(sprinc,axis=1) - np.amin(sprinc,axis=1)
        else:
            d12 = sprinc[:,0] - sprinc[:,1]
            d23 = sprinc[:,1] - sprinc[:,2]
            d31 = sprinc[:,2] - sprinc[:,0]
            if (self.sy==None):
                h0 = h1 = h2 = 1.
                d0 = 0.
            else:
                h0 = self.hill[0]
                h1 = self.hill[1]
                h2 = self.hill[2]
                d0 = self.drucker
            # consider anisotropy in flow behavior in second invariant in Hill-type approach
            I2  = 0.5*(h0*np.square(d12) + h1*np.square(d23) + h2*np.square(d31))
            # consider hydrostatic stresses for tension-compression assymetry
            I1  = np.sum(sprinc[:,0:3], axis=1)/3.
            seq  = np.sqrt(I2) + d0*I1   # generalized eqiv. stress
        if sh==(3,):
            seq = seq[0]
        return seq

    def elasticity(self, C11=None, C12=None, C44=None, # standard parameters for crystals with cubic symmetry
                   CV=None,                            # user specified Voigt matrix
                   E=None, nu=None):                   # parameters for isotropic material
        '''Define elastic material properties

        Parameters
        ----------
        C11 : float
        C12 : float
        C44 : float
            Anisoptropic elastic constants of material (optional,
            if (C11,C12,C44) not given, either (E,nu) or CV must be specified)
        E   : float
        nu  : float
            Isotropic parameters Young's modulus and Poisson's number (optional,
            if (E,nu) not given, either (C11,C12,C44) or CV must be specified)
        CV  : (6,6) array
            Voigt matrix of elastic constants (optional, if CV not given, either
            (C11,C12,C44) or (E,nu) must be specified)
        '''
        if (E is not None):
            if (nu is None):
                sys.exit('Error: Inconsistent definition of material parameters: Only E provided')
            if ((C11 is not None)or(C12 is not None)or(C44 is not None)):
                sys.exit('Error: Inconsistent definition of material parameters: E provided together with C_ij')
            hh = E/((1.+nu)*(1.-2.*nu))
            self.C11 = (1.-nu)*hh
            self.C12 = nu*hh
            self.C44 = (0.5-nu)*hh
            self.E = E
            self.nu = nu
            self.CV = None
        elif (C11 is not None):
            if (nu is not None):
                sys.exit('Error: Inconsistent definition of material parameters: nu provided together with C_ij')
            if ((C12 is None)or(C44 is None)):
                sys.exit('Error: Inconsistent definition of material parameters: C_12 or C_44 values missing')
            self.C11 = C11
            self.C12 = C12
            self.C44 = C44
            self.nu = C12/(C11+C12)
            self.E = 2*C44*(1+self.nu) # only for isotropy
            self.CV = None
            print('Warning: E and nu calculated from anisotropic elastic parameters')
        elif (CV is not None):
            self.CV = np.array(CV)
            self.nu = self.CV[0,1]/(self.CV[0,0]+self.CV[0,1])
            self.E = 2*self.CV[3,3]*(1+self.nu) # only for isotropy
            #print('Warning: E and nu calculated from anisotropic elastic parameters')
        else:
            sys.exit('Error: Inconsistent definition of material parameters')

    def plasticity(self, sy=None, hill=[1., 1., 1.], drucker=0., khard=0., Tresca=False):
        '''Define plastic material parameters; anisotropic Hill-like and Drucker-like
        behavior is supported

        Parameters
        ----------
        sy   : float
            Yield strength
        hill : (3,) array
            Parameters for Hill-like orthotropic anisotropy (optional, default: isotropy)
        drucker : float
            Parameter for Drucker-like tension-compression asymmetry (optional, default: 0)
        khard: float
            Paramater for linear strain hardening (optional, default: 0)
        Tresca : Boolean
            Indicate if Tresca equivalent stress should be used (optional, default: False)
        '''
        self.pm = 1  # set plasticity model to Hill
        self.sy = sy   # yield strength
        self.khard = khard  # work hardening rate (linear w.h.)
        self.Hpr = khard/(1.-khard/self.E)  # constant for w.h.
        self.hill = np.array(hill)  # Hill anisotropic parameters
        self.drucker = drucker   # Drucker-Prager parameter: weight of hydrostatic stress
        self.Tresca = Tresca

    def microstructure(self, texture_param=None, grain_size=None, grain_shape=None, porosity=None):
        '''Define microstructural parameters of the material: crstallographic texture, grain size, grain shape
        and porosity.

        Paramaters
        ----------
        text_param : array-like
            Parameters describing texture (optional)
        grain_size : float
            Average grain size of material (optional)
        grain_shape : (3,) array
            Lengths of the three main axes of ellipsoid describing grain shape (optional)
        porosity : float
            Porosity of the material
        '''
        self.texture_param = texture_param
        self.grain_size = grain_size
        self.grain_shape = grain_shape
        self.porosity = porosity

    def epl_dot(self, sig, epl, Cel, deps):
        '''Calculate plastic strain increment relaxing stress back to yield locus;
        Reference: M.A. Crisfield, Non-linear finite element analysis of solids and structures,
        Chapter 6, Eqs. (6.4), (6.8) and (6.17)

        Parameters
        ----------
        sig : Voigt tensor
            Stress
        epl : Voigt tensor
            Plastic strain
        Cel : (6,6) array
            Elastic stiffnes tensor
        deps: Voigt tensor
            Strain increment from predictor step

        Returns
        -------
        pdot : Voigt tensor
            Plastic strain increment
        '''
        peeq = FE.eps_eq(epl)     # equiv. plastic strain
        yfun = self.calc_yf(sig+Cel@deps, peeq=peeq)
        #for DEBUGGING
        '''yf0  = self.calc_yf(sig, peeq=peeq)
        if yf0<-FE.ptol and yfun>FE.ptol and peeq<1.e-5:
            if self.ML_yf:
                ds = Cel@deps
                yfun = self.ML_full_yf(sig+ds)
            print('*** Warning in epl_dot: Step crosses yield surface')
            print('sig, epl, deps, yfun, yf0, peeq,caller', sig, epl, deps, yfun, yf0, peeq, sys._getframe().f_back.f_code.co_name)
            yfun=0. # catch error that can be produced for anisotropic materials'''
        if (yfun<=FE.ptol):
            pdot = np.zeros(6)
        else:
            a = np.zeros(6)
            a[0:3] = self.calc_fgrad(FE.Stress(sig).p)
            hh = a.T @ Cel @ a + self.Hpr
            lam_dot = a.T @ Cel @ deps / hh  # deps must not contain elastic strain components
            pdot = lam_dot * a
        return pdot

    def C_tan(self, sig, Cel):
        '''Calculate tangent stiffness relaxing stress back to yield locus;
        Reference: M.A. Crisfield, Non-linear finite element analysis of solids and structures,
        Chapter 6, Eqs. (6.9) and (6.18)

        Parameters
        ----------
        sig : Voigt tensor
            Stress
        Cel : (6,6) array
            Elastic stiffness tensor used for predictor step

        Returns
        -------
        Ct : (6,6) array
            Tangent stiffness tensor
        '''
        a = np.zeros(6)
        a[0:3] = self.calc_fgrad(FE.Stress(sig).princ)
        hh = a.T @ Cel @ a + self.Hpr
        ca = Cel @ a
        Ct = Cel - np.kron(ca, ca).reshape(6,6)/hh
        return Ct

    def create_sig_data(self, N=None, syc=None, Nseq=12, offs = 0.01, extend=False, rand=False):
        '''Function to create consistent data sets on the deviatoric stress plane
        for training or testing of ML yield function. Data is created in form of cylindrical stresses

        Parameters
        ----------
        N    : int
            Number of load cases (polar angles) to be created (optional,
            either N or theta must be provided)
        theta: (N,) array
            List of polar angles from which training data in deviatoric stress plane is created
            (optional, either theta or N must be provided)
        Nseq : int
            Number of equiv. stresses generated up to yield strength (optional, default: 12)
        offs : float
            Start of range for equiv. stress (optional, default: 0.01)
        extend : Boolean
            Create additional data in plastic regime (optional, default: False)
        rand   : Boolean
            Chose random load cases (polar angles) (optional, default: False)

        Returns
        -------
        st : (M,3) array
            Cylindrical training stresses, M = N (2 Nseq + Nextend)
        '''
        if syc is None:
            if N is None:
                print('Warning in create_sig_data: Neither N not theta provided.')
                print('Continuing with N=36')
                N = 36
            if not rand:
                theta = np.linspace(-np.pi, np.pi, N)
            else:
                theta = 2.*(np.random.rand(N)-0.5)*np.pi
        else:
            N = len(syc)
            sc    = syc[:,0]
            theta = syc[:,1]
        seq = np.linspace(offs, 2*self.sy, 2*Nseq)
        if extend:
            # add training points in plastic regime to avoid fallback of SVC decision fct. to zero
            seq = np.append(seq, np.array([2.4, 3., 4., 5.])*self.sy)
        Nd = len(seq)
        st = np.zeros((N*Nd,3))
        if syc is None:
            yt = None
        else:
            yt = np.zeros(N*Nd)
        for i in range(Nd):
            j0 = i*N
            j1 = (i+1)*N
            st[j0:j1, 0] = seq[i]
            st[j0:j1, 1] = theta
            if syc is not None:
                yt[j0:j1] = np.sign(seq[i]*np.ones(N) - sc)
        return st, yt

    def find_yloc(self, x, sp, ana=False):
        '''Function to expand unit stresses by factor and calculate yield function;
        used by search algorithm to find zeros of yield function.

        Parameters
        ----------
        x : 1d-array
            Multiplyer for stress
        sp : (3,) array
            Principal stress
        ana : Boolean
            Decides if analytical yield function is evaluated (optional, default: False)
        Returns
        -------
        f : 1d-array
            Yield function evaluated at sig=x.sp
        '''
        y = np.array([x,x,x]).transpose()
        f = self.calc_yf(y*sp, ana=ana, pred=False)
        return f

    def ellipsis(self, a=1., b=1./np.sqrt(3.), n=72):
        '''Create ellipsis with main axis along 45° axis, used for graphical representation of isotropic yield locus.

        Parameters
        ----------
        a : float
            Long half-axis of ellipsis (optional, default: 1)
        a : float
            Short half-axis of ellipsis (optional, default: 1/sqrt(3))
        n : int
            Number of points on ellipsis to be calculated

        Returns
        -------
        x, y : (n,) arrayy
            x and y coordinates of points on ellipsis
        '''
        t = np.arange(0., 2.1*np.pi, np.pi/n)
        x = a*np.cos(t) - b*np.sin(t)
        y = a*np.cos(t) + b*np.sin(t)
        return x, y

    def plot_data(self, Z, axs, xx, yy, field=True, c='red'):
        '''Plotting data in stress space to visualize yield loci.

        Parameters
        ----------
        Z   : array
            Data for field plot
        axs : handle
            Axes where plot is added
        xx  : meshgrid
            x-coordinates
        yy  : meshgrid
            y-coordinates
        field : Boolean
            Decide if field is plotted (optional, default: True)
        c   : str
            Color for contour line (optional, default: 'red')

        Returns
        -------
        line : handle
            Reference to plotted line
        '''
        'symmetrize Z values'
        zmin = np.amin(Z)
        zmax = np.amax(Z)
        if (-zmin < zmax):
            Z[np.nonzero(Z>-zmin)] = -zmin
        else:
            Z[np.nonzero(Z<-zmax)] = -zmax
        Z = Z.reshape(xx.shape)

        'display data'
        if field:
            im = axs.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
               origin='lower', cmap=plt.cm.PuOr_r)
            #fig.colorbar(im, ax=axs)
        contour = axs.contour(xx, yy, Z, levels=[0], linewidths=2,
                    linestyles='solid', colors=c)
        line = contour.collections
        return line

    def plot_yield_locus(self, fun=None, label=None, data=None, trange=1.e-2,
                         xstart=-2., xend=2., axis1=[0], axis2=[1], iso=False, ref_mat=None,
                         field=False, Nmesh=100, file=None, fontsize=20):
        '''Plot different cuts through yield locus in 3D principle stress space.

        Parameters
        ----------
        fun   : function handle
            Yield function to be plotted (optional, default: own yield function)
        label : str
            Label for yield function (optional, default: own name)
        data  : (N,3) array
            Stress data to be used for scatter plot (optional)
        trange : float
            Cut-off for data to be plotted on slice (optional, default: 1.e-2)
        xstart : float
            Smallest value on x-axis (optional, default: -2)
        xend   : float
            Largest value on x-axis (optional, default: 2)
        axis1 : list
            Cartesian stress coordinates to be plotted on x-axis of slices (optional, default: [0])
        axis2 : list
            Cartesian stress coordinates to be plotted on y-axis of slices (optional, default: [1])
        iso   : Boolean
            Decide if reference ellipsis for isotropic material is plotted (optional, default: False)
        ref_mat=None
            Reference material to plot yield locus (optional)
        field : Boolean
            Decide if field of yield function is plotted (optional, default: False)
        Nmesh : int
            Number of mesh points per axis on which yield function is evaluated (optional, default:100)
        file  : str
            File name for output of olot (optional)
        fontsize : int
            Fontsize for axis annotations (optional, default: 20)

        Returns
        -------
        axs : pyplot axis handle
            Axis of the plot
        '''
        xx, yy = np.meshgrid(np.linspace(xstart, xend, Nmesh),
                         np.linspace(xstart, xend, Nmesh))
        Nm2 = Nmesh*Nmesh
        Nc = len(axis1)
        if len(axis2)!=Nc:
            sys.exit('Error in plot_yield_locus: mismatch in dimensions of ax1 and ax2')

        if Nc==1:
            fs = (10, 8)
            fontsize *= 4/5
        else:
            fs = (20, 5)
        fig, axs  = plt.subplots(nrows=1, ncols=Nc, figsize=fs)
        fig.subplots_adjust(hspace=0.3)

        'loop over subplots in axis1 and axis2'
        for j in range(Nc):
            if Nc==1:
                ax=axs
            else:
                ax=axs[j]
            lines=[]
            labels=[]
            'select slice in 3D stress space'
            s1 = None
            s2 = None
            s3 = None
            'first axis'
            if axis1[j]==0:
                s1 = xx.ravel()
                title = '$\sigma_1$'
                xlab = '$\sigma_1 / \sigma_y$'
            elif axis1[j]==1:
                s2 = xx.ravel()
                title = '$\sigma_2$'
                xlab = '$\sigma_2 / \sigma_y$'
            elif axis1[j]==2:
                s3 = xx.ravel()
                title = '$\sigma_3$'
                xlab = '$\sigma_3 / \sigma_y$'
            elif axis1[j]==3:
                s1 = xx.ravel()
                s2 = xx.ravel()
                title = '$p=\sigma_1=\sigma_2$'
                xlab = '$p / \sigma_y$'
                ref = False
                axis1[j] = 0
            else:
                print('Warning in plot_yield_locus: axis1 not defined properly, set to sig_1', axis1, j)
                s1 = xx.ravel()
                title = '$\sigma_1$'
                xlab = '$\sigma_1 / \sigma_y$'
            'second axis'
            if axis2[j]==0:
                s1 = yy.ravel()
                title += '-$\sigma_1$ slice'
                ylab = '$\sigma_1 / \sigma_y$'
            elif axis2[j]==1:
                s2 = yy.ravel()
                title += '-$\sigma_2$ slice'
                ylab = '$\sigma_2 / \sigma_y$'
            elif axis2[j]==2:
                s3 = yy.ravel()
                title += '-$\sigma_3$ slice'
                ylab = '$\sigma_3 / \sigma_y$'
            elif axis2[j]==3:
                s3 = yy.ravel()
                title += '-$\sigma_3$ slice'
                ylab = '$\sigma_3 / \sigma_y$'
                axis2[j]=2
            else:
                print('Warning in plot_yield_locus: axis2 not defined properly, set to sig_2', axis2, j)
                s2 = yy.ravel()
                title += '-$\sigma_2$ slice'
                ylab = '$\sigma_2 / \sigma_y$'
            si3 = 1  # slice for data
            if s1 is None:
                s1 = np.zeros(Nm2)
                si3 = 0
            if s2 is None:
                s2 = np.zeros(Nm2)
                si3 = 1
            if s3 is None:
                s3 = np.zeros(Nm2)
                si3 = 2
            sig = np.c_[s1, s2, s3]  # set stresses for yield locus calculation

            'evaluate yield function to be plotted'
            if fun is None:
                Z = self.calc_yf(sig*self.sy, pred=True)/self.sy
            else:
                Z = fun(sig*self.sy, pred=True)/self.sy
            if label is None:
                label = self.name
            hl = self.plot_data(Z, ax, xx, yy, field=field)
            lines.extend(hl)
            labels.extend([label])
            'plot reference function if provided'
            if ref_mat is not None:
                Z = ref_mat.calc_yf(sig*ref_mat.sy, pred=True)/ref_mat.sy
                labels.extend([ref_mat.name])
                hl = self.plot_data(Z, ax, xx, yy, field=False, c='black')
                lines.extend(hl)
            'plot ellipsis as reference if requested'
            if iso:
                x0, y0 = self.ellipsis()  # reference for isotropic material
                hl = ax.plot(x0,y0,'-b')
                lines.extend(hl)
                labels.extend(['isotropic J2'])
            'plot data if provided'
            if (data is not None):
                # select data from 3D stress space within range [xstart, xend] and to fit to slice
                dat = np.array(data)/self.sy
                dsel = np.nonzero(np.logical_and(np.abs(dat[:,si3])<trange,
                    np.logical_and(dat[:,axis1[j]]>xstart, dat[:,axis1[j]]<xend)))
                ir = dsel[0]
                yf = np.sign(self.calc_yf(data[ir,:]))
                h1 = ax.scatter(dat[ir,axis1[j]], dat[ir,axis2[j]], s=60, c=yf,
                                    cmap=plt.cm.Paired, edgecolors='k')
            ax.legend(lines,labels,loc='upper left',fontsize=fontsize-4)
            #ax.set_title(title,fontsize=fontsize)
            ax.set_xlabel(xlab,fontsize=fontsize)
            ax.set_ylabel(ylab,fontsize=fontsize)
            ax.tick_params(axis="x", labelsize=fontsize-6)
            ax.tick_params(axis="y", labelsize=fontsize-6)
        'save plot to file if filename is provided'
        if file is not None:
            fig.savefig(file+'.pdf', format='pdf', dpi=300)
        return axs


    def calc_properties(self, size=2, Nel=2, verb=False, eps=0.005, min_step=None, sigeps=False):
        '''Use FE model to calculate material strength and stress-strain data along a given load path.

        Parameters
        ----------
        size : int
            Size of FE model (optional, defaul: 2)
        Nel : int
            Number of elements per axis (optional, defaul: 2)
        verb : Boolean
            Be verbose with output (optional, default: False)
        eps : float
            Maximum total strain (optional, default:0.005)
        min_step : int
            Minumum number of load steps (optional)
        sigeps : Boolean
            Decide if data for stress-strain curves in stored in dictionary Material.sigeps (optional, default: False)
        '''
        def calc_strength(vbc1, nbc1, vbc2, nbc2, sel):
            fe=FE.Model(dim=2,planestress=True)
            fe.geom([size], LY=size) # define section in absolute length
            fe.assign([self])        # assign material to section
            fe.bcleft(0.)            # fix lhs nodes in x-direction
            fe.bcbot(0.)             # fix bottom nodes in y-direction
            fe.bcright(vbc1, nbc1)   # define BC in x-direction
            fe.bctop(vbc2, nbc2)     # define BC in y-direction
            fe.mesh(NX=Nel, NY=Nel)  # create mesh
            fe.solve(verb=verb,min_step=min_step)      # solve mechanical equilibrium condition under BC
            seq  = self.calc_seq(fe.sgl)   # store time dependent mechanical data of model
            eeq  = FE.eps_eq(fe.egl)
            peeq = FE.eps_eq(fe.epgl)
            iys = np.nonzero(peeq<1.e-6)
            ys = seq[iys[0][-1]]
            self.prop[sel]['ys']   = ys
            self.prop[sel]['seq']  = seq
            self.prop[sel]['eeq']  = eeq
            self.prop[sel]['peeq'] = peeq
            seq  = FE.seq_J2(fe.sgl)   # store time dependent mechanical data of model
            eeq  = FE.eps_eq(fe.egl)
            peeq = FE.eps_eq(fe.epgl)
            iys = np.nonzero(peeq<1.e-6)  # take stress at last index of elastic regime
            ys = seq[iys[0][-1]]
            self.propJ2[sel]['ys']   = ys
            self.propJ2[sel]['seq']  = seq
            self.propJ2[sel]['eeq']  = eeq
            self.propJ2[sel]['peeq'] = peeq
            if sigeps:
                self.sigeps[sel]['sig'] = fe.sgl
                self.sigeps[sel]['eps'] = fe.egl
                self.sigeps[sel]['epl'] = fe.epgl
            return
        def calc_stx():
            u1 = eps*size
            calc_strength(u1, 'disp', 0., 'force', 'stx')
            self.prop['stx']['style'] = '-r'
            self.prop['stx']['name']  = 'uniax-x'
            return
        def calc_sty():
            u2 = eps*size
            calc_strength(0., 'force', u2, 'disp', 'sty')
            self.prop['sty']['style'] = '-b'
            self.prop['sty']['name']  = 'uniax-y'
            return
        def calc_et2():
            u1 = 0.67*eps*size
            u2 = 0.67*eps*size
            calc_strength(u1, 'disp', u2, 'disp', 'et2')
            self.prop['et2']['style'] = '-k'
            self.prop['et2']['name']  = 'equibiax'
            return
        def calc_ect():
            u1 = -0.78*eps*size
            u2 = 0.78*eps*size
            calc_strength(u1, 'disp', u2, 'disp', 'ect')
            self.prop['ect']['style'] = '-m'
            self.prop['ect']['name']  = 'shear'
            return
        self.calc = {    # calculates strength and stress strain data along given load paths
            'stx'   : calc_stx(),   # uniaxial tensile yield strength in horizontal (x-)direction
            'sty'   : calc_sty(),   # uniaxial tensile yield strength in vertical (y-)direction
            'et2'   : calc_et2(),   # strength under plane stress, equibiaxial strain in x and y direction
            'ect'   : calc_ect()    # strength under pure shear strain (x-compression, y-tension), plane stress
        }

    def plot_stress_strain(self, Hill=False, file=None, fontsize=14):
        '''Plot stress-strain data and print values for strength.

        Parameters
        ----------
        Hill : Boolean
            Decide if data for Hill-type equivalent stress is presented (optional, default: False)
        file : str
            Filename to save plot (optional)
        fontsize : int
            Fontsize for axis annotations (optional, default: 14)
        '''
        legend = []
        print('---------------------------------------------------------')
        for sel in self.prop:
            print('J2 yield stress under',self.prop[sel]['name'],'loading:', self.propJ2[sel]['ys'].round(decimals=3),'MPa')
            print('---------------------------------------------------------')
            plt.plot(self.propJ2[sel]['eeq']*100., self.propJ2[sel]['seq'], self.prop[sel]['style'])
            legend.append(self.prop[sel]['name'])
        plt.title('Material: '+self.name,fontsize=fontsize)
        plt.xlabel(r'$\epsilon_\mathrm{eq}$ (%)',fontsize=fontsize)
        plt.ylabel(r'$\sigma^\mathrm{J2}_\mathrm{eq}$ (MPa)',fontsize=fontsize)
        plt.tick_params(axis="x", labelsize=fontsize-4)
        plt.tick_params(axis="y", labelsize=fontsize-4)
        plt.legend(legend, loc='lower right',fontsize=fontsize)
        if file is not None:
            plt.savefig(file+'J2.pdf', format='pdf', dpi=300)
        plt.show()
        if Hill:
            for sel in self.prop:
                print('Hill yield stress under',self.prop[sel]['name'],'loading:', self.prop[sel]['ys'].round(decimals=3),'MPa')
                print('---------------------------------------------------------')
                plt.plot(self.prop[sel]['eeq']*100., self.prop[sel]['seq'], self.prop[sel]['style'])
                legend.append(self.prop[sel]['name'])
            plt.title('Material: '+self.name,fontsize=fontsize)
            plt.xlabel(r'$\epsilon_\mathrm{eq}$ (%)',fontsize=fontsize)
            plt.ylabel(r'$\sigma_\mathrm{eq}$ (MPa)',fontsize=fontsize)
            plt.tick_params(axis="x", labelsize=fontsize-4)
            plt.tick_params(axis="y", labelsize=fontsize-4)
            plt.legend(legend, loc='lower right',fontsize=fontsize)
            if file is not None:
                plt.savefig(file+'Hill.pdf', format='pdf', dpi=300)
            plt.show()
        return
