'''Python module for Material definitions and constitutive modeling;
uses NumPy, MatPlotLib, sklearn and pyLabFEM;
Version: 1.0 (2020-03-06)
Author: Alexander Hartmaier, ICAMS/Ruhr-University Bochum, March 2020
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)'''
import numpy as np
import pyLabFEM as FE
import matplotlib.pyplot as plt
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
        Existence of trained machine learning (ML) yield function
    ML_grad : Boolean
        Existence of trained ML gradient
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
    '''
    'Methods'
    #elasticity: define elastic material parameters C11, C12, C44
    #plasticity: define plastic material parameter sy, khard
    #epl_dot: calculate plastic strain rate
    
    def __init__(self, name='Material'):
        self.sy = None  # Elasticity will be considered unless sy is set
        self.ML_yf = False # use conventional plasticity unless trained ML functions exists
        self.ML_grad = False # use conventional gradient unless ML function exists
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
    
    
    def setup_yf_SVM(self, x, y_train, x_test=None, y_test=None, C=10., gamma=1., fs=0.1, plot=False):
        '''Initialize and train Support Vector Classifier (SVC) as machine learning (ML) yield function
        
        Parameters
        ----------
        x   :  (N,2) or (N,3) array
            Training data either as Cartesian princ. stresses (N,3) or cylindrical stresses (N,2)
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
        if sh==(N,3):
            'data format: princ. stresses'
            X_train[:,0] = FE.seq_J2(x)/self.sy - 1.
            X_train[:,1] = FE.polar_ang(x)/np.pi
        elif sh==(N,2):
            'data format: seq, theta values'
            X_train[:,0] = x[:,0]/self.sy - 1.
            X_train[:,1] = x[:,1]/np.pi
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
        '''Calculate gradient to yield surface
        
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
                hyd = np.sum(sig)/3.  # hydrostatic stress
                dev = sig - hyd  # deviatoric princ. stress
                vn = np.linalg.norm(dev)  #norms of stress vectors
                dseqds = 3.*dev/(vn*np.sqrt(1.5))
                J = np.ones((3,3))
                J[:,2] /= 3.
                J[:,0] = dseqds
                dsa = np.dot(sig,FE.a_vec)
                dsb = np.dot(sig,FE.b_vec)
                sc = dsa + 1j*dsb
                z = -1j*((FE.a_vec+1j*FE.b_vec)/sc - dseqds/(vn*np.sqrt(1.5)))
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
    
    def elasticity(self, C11=None, C12=None, C44=None, E=None, nu=None):
        '''Define elastic material properties
        
        Parameters
        ----------
        C11 : float
        C12 : float
        C44 : float 
            Anisoptropic elastic constants of material (optional, if C11, C12, C44 not given, E and nu must be specified)
        E   : float
        nu  : float 
            Isotropic parameters Young's modulus and Poisson's number (optional, 
            if E and nu not given, C11, C12, C44 must be specified)
        '''
        if (E!=None):
            if (nu==None):
                sys.exit('Error: Inconsistent definition of material parameters: Only E provided')
            if ((C11!=None)or(C12!=None)or(C44!=None)):
                sys.exit('Error: Inconsistent definition of material parameters: E provided together with C_ij')
            hh = E/((1.+nu)*(1.-2.*nu))
            self.C11 = (1.-nu)*hh
            self.C12 = nu*hh
            self.C44 = (0.5-nu)*hh
            self.E = E
            self.nu = nu
        elif (C11!=None):
            if (nu!=None):
                sys.exit('Error: Inconsistent definition of material parameters: nu provided together with C_ij')
            if ((C12==None)or(C44==None)):
                sys.exit('Error: Inconsistent definition of material parameters: C_12 or C_44 values missing')
            self.C11 = C11
            self.C12 = C12
            self.C44 = C44
            self.nu = C12/(C11+C12)
            self.E = 2*C44*(1+self.nu) # only for isotropy
            print('Warning: E and nu calculated from anisotropic elastic parameters')
        else:
            sys.exit('Error: Inconsistent definition of material parameters')   

    def plasticity(self, sy=None, hill=[1., 1., 1.], drucker=0., khard=None):
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
        '''
        self.pm = 1  # set plasticity model to Hill
        self.sy = sy   # yield strength
        self.khard = khard  # work hardening rate (linear w.h.)
        self.Hpr = khard/(1.-khard/self.E)  # constant for w.h.
        self.hill = np.array(hill)  # Hill anisotropic parameters
        self.drucker = drucker   # Drucker-Prager parameter: weight of hydrostatic stress

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
        yfun = self.calc_yf(sig,peeq=peeq)
        if (yfun<1.e-5):
            pdot = 0.
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
    
    def find_yloc(self, x, snorm, ana=False):
        '''Function to expand unit stresses by factor and calculate yield function;
        used by search algorithm to find zeros of yield function.
        
        Parameters
        ----------
        x : 1d-array
            Multiplyer for stress
        snorm : (3,) array
            Principal stress
        ana : Boolean
            Decides if analytical yield function is evaluated (optional, default: False)
        Returns
        -------
        f : 1d-array
            Yield function evaluated at sig=x.snorm
        '''
        y = np.array([x,x,x]).transpose()
        f = self.calc_yf(y*snorm, ana=ana)
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
        xx  : array
            x-coordinates 
        yy  : array
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

    def plot_yield_locus(self, fun=None, label='undefined', data=None, trange=1.e-2, 
                         xstart=-2., xend=2., axis1=[0], axis2=[1], ref=True, ref_mat=None,
                         field=True, Nmesh=100, file=None, fontsize = 20):
        '''Plot different cuts through yield locus in 3D principle stress space.
        
        Parameters
        ----------
        fun   : function handle
            Reference function to be plotted (optional)
        label : str
            Label for reference function (optional, default:'undefined')
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
        ref   : Boolean
            Decide if reference ellipsis for isotropic material is plotted (optional, default: True)
        ref_mat=None
            Reference material to plot yield locus (optional)
        field : Boolean
            Decide if field of yield function is plotted (optional, default: True)
        Nmesh : int
            Number of mesh points per axis on which yield function is evaluated (optional, default:100)
        file  : str
            File name for output of olot (optional)
        fontsize : int
            Fontsize for axis annotations (optional, default: 20)
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
            if ref_mat is None:
                Z = self.calc_yf(sig*self.sy, ana=True, pred=True)/self.sy
                labels.extend([self.name])
            else:
                Z = ref_mat.calc_yf(sig*ref_mat.sy, pred=True)/ref_mat.sy
                labels.extend([ref_mat.name])
            hl = self.plot_data(Z, ax, xx, yy, field=False)
            lines.extend(hl)
            
            if (fun is not None):
                Z = fun(sig*self.sy, pred=True)
                hl = self.plot_data(Z, ax, xx, yy, field=field, c='black')
                lines.extend(hl)
                labels.extend(label)
            if ref:
                x0, y0 = self.ellipsis()  # reference for isotropic material
                hl = ax.plot(x0,y0,'-b')
                lines.extend(hl)
                labels.extend(['isotropic J2'])
            if (data is not None):
                # select data from 3D stress space within range [xstart, xend] and to fit to slice
                dat = data/self.sy
                dsel = np.nonzero(np.logical_and(np.abs(dat[:,si3])<trange, 
                    np.logical_and(dat[:,axis1[j]]>xstart, dat[:,axis1[j]]<xend)))
                ir = dsel[0]
                yf = np.sign(self.calc_yf(data[ir,:]))
                h1 = ax.scatter(dat[ir,axis1[j]], dat[ir,axis2[j]], s=60, c=yf, 
                                    cmap=plt.cm.Paired, edgecolors='k')
            ax.legend(lines,labels,loc='upper left',fontsize=fontsize-4)
            ax.set_title(title,fontsize=fontsize)
            ax.set_xlabel(xlab,fontsize=fontsize)
            ax.set_ylabel(ylab,fontsize=fontsize)
            ax.tick_params(axis="x", labelsize=fontsize-6)
            ax.tick_params(axis="y", labelsize=fontsize-6)
        if file is not None:
            fig.savefig(file+'.pdf', format='pdf', dpi=300)
        plt.show()

    
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
                self.sigeps[sel]['eps'] = 1. #fe.egl
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
        print('==============================================================')
        for sel in self.prop:
            print('J2 yield stress under',self.prop[sel]['name'],'loading:', self.propJ2[sel]['ys'],'MPa')
            print('==============================================================')
            plt.plot(self.propJ2[sel]['eeq']*100., self.propJ2[sel]['seq'], self.prop[sel]['style'])
            legend.append(self.prop[sel]['name'])
        plt.title('Material: '+self.name,fontsize=fontsize)
        plt.xlabel(r'$\epsilon_\mathrm{eq}$ (%)',fontsize=fontsize)
        plt.ylabel(r'$\sigma^\mathrm{ J2}_\mathrm{eq}$ (MPa)',fontsize=fontsize)
        plt.tick_params(axis="x", labelsize=fontsize-4)
        plt.tick_params(axis="y", labelsize=fontsize-4)
        plt.legend(legend, loc='lower right',fontsize=fontsize)
        if file is not None:
            plt.savefig(file+'J2.pdf', format='pdf', dpi=300)
        plt.show()
        if Hill:
            for sel in self.prop:
                print('Hill yield stress under',self.prop[sel]['name'],'loading:', self.prop[sel]['ys'],'MPa')
                print('==============================================================')
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