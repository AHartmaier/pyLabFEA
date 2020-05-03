# Module pylabfea.material
'''Module pylabfea.material introduces class ``Material`` that contains attributes and methods
needed for elastic-plastic material definitions in FEA. It also enables the training of 
machine learning algorithms as yield functions for plasticity.
The module pylabfea.model is used to calculate mechanical properties of a defined material 
under various loading conditions.

uses NumPy, MatPlotLib, sklearn and pyLabFEA.model

Version: 2.1 (2020-04-01)
Author: Alexander Hartmaier, ICAMS/Ruhr-University Bochum, April 2020
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)'''
from pylabfea.basic import *
from pylabfea.model import Model
from scipy.optimize import fsolve
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
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
    E, nu  : float
        Isotropic elastic constants, Young modulus and Poisson number
    msparam : ditionary
        Dicitionary with microstructural parameters assigned to this material
    Ndof   : int
        degress of freedom for yield function, mandatory: 1:seq, 2:theta; optional: 3:work_hard, 4:texture)
        
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
    msparam :
        Store data on microstructure parameters: 'Npl', 'Nlc, 'Ntext', 'texture', 'peeq_max', 'work_hard', 'flow_stress' 
        are obtained from data analysis module. Other parameters can be added.
    '''
    'Methods'
    #elasticity: define elastic material parameters C11, C12, C44
    #plasticity: define plastic material parameter sy, khard
    #epl_dot: calculate plastic strain rate

    def __init__(self, name='Material'):
        self.sy = None  # Elasticity will be considered unless sy is set
        self.ML_yf = False # use conventional plasticity unless trained ML functions exists
        self.ML_grad = False # use conventional gradient unless ML function exists
        self.Tresca = False  # use J2 or Hill equivalent stress unless defined otherwise
        self.name = name
        self.msparam = None
        self.Ndof = 2
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
                sig = np.array([Stress(sig).p])
                N = 1
            else:
                N = len(sig)
            x = np.zeros((N,self.Ndof))
            x[:,0] = seq_J2(sig)/self.scale_seq - 1.
            x[:,1] = polar_ang(sig)/np.pi
            if self.Ndof>=3:
                x[:,2] = self.wh_cur/self.scale_wh - 1.
            if self.Ndof==4:
                x[:,3] = self.ms_cur/self.scale_text - 1.
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
                seq = Stress(sig).seq(self)  # calculate equiv. stress
            else:  # array of princ. stresses
                seq = self.calc_seq(sig)
            self.sflow = self.sy + peeq*self.Hpr
            f = seq - self.sflow
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
        sp = Stress(sig).p
        seq = self.calc_seq(sp)
        print('Full YF:seq, ld')
        if seq<0.01 and ld is None:
            yf = seq - 0.85*self.sflow
        else:
            if (seq>=0.01):
                ld = sp*self.sflow*np.sqrt(1.5)/seq
            x0 = 1.
            if ld[0]*ld[1] < -1.e-5:
                x0 = 0.5
                if self.Tresca:
                    x0 = 0.4
            x1, infodict, ier, msg = fsolve(self.find_yloc, x0, args=ld, xtol=1.e-5, full_output=True)
            y1 = infodict['fvec']
            if np.abs(y1)<1.e-3 and x1[0]<3.:
                # zero of ML yield fct. detected at x1*sy
                yf = seq - x1[0]*self.calc_seq(ld)
            else:
                # zero of ML yield fct. not found: get conservative estimate
                yf = seq - 0.85*self.sflow
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
        if self.msparam is None:
            self.scale_seq = self.sy
        else:
            'calculate scaling factors need for SVC training from microstructure parameters'
            self.scale_seq  = np.average(self.msparam['flow_seq_av'])
            self.scale_wh   = np.average(self.msparam['work_hard'])
            self.scale_text = np.average(self.msparam['texture'])
        N = len(x)
        sh = np.shape(x)
        X_train = np.zeros((N,self.Ndof))
        if not cyl:
            'princ. stresses'
            X_train[:,0] = seq_J2(x)/self.scale_seq - 1.
            X_train[:,1] = polar_ang(x)/np.pi
            print('Using principle stresses for training')
        else:
            'cylindrical stresses'
            X_train[:,0] = x[:,0]/self.scale_seq - 1.
            X_train[:,1] = x[:,1]/np.pi
            print('Using cyclindrical stresses for training')
        if self.Ndof>=3:
            X_train[:,2] = x[:,2]/self.scale_wh - 1.
            print('Using work hardening data for training: %i data sets up to PEEQ=%6.3f' % (self.msparam['Npl'], self.msparam['peeq_max']))
        if self.Ndof==4:
            X_train[:,3] = x[:,3]/self.scale_text - 1.
            print('Using texture data for training: %i data sets with texture_parameters in range [%4.2f,%4.2f]' 
                  % (self.msparam['Ntext'], self.msparam['texture'][0], self.msparam['texture'][-1]))
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
            X_test = np.zeros((Ntest,self.Ndof))
            if not cyl:
                X_test[:,0] = seq_J2(x_test)/self.scale_seq - 1.
                X_test[:,1] = polar_ang(x_test)/np.pi
            else:
                X_test[:,0] = x_test[:,0]/self.scale_seq - 1.
                X_test[:,1] = x_test[:,1]/np.pi
            if self.Ndof>=3 :
                X_test[:,2] = x_test[:,2]/self.scale_wh - 1.
            if self.Ndof==4:
                X_test[:,3] = x_test[:,3]/self.scale_text - 1.
        'define and fit SVC'
        self.svm_yf = svm.SVC(kernel='rbf',C=C,gamma=gamma)
        self.svm_yf.fit(X_train, y_train)
        self.ML_yf = True
        '''if (ptol>=1.):
            ptol=0.9
            print('Warning: ptol must be <1 for ML yield function')'''
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
            if self.Ndof==2:
                feat = np.c_[yy.ravel(),xx.ravel()]
            elif self.Ndof==3:
                feat = np.c_[yy.ravel(),xx.ravel(),np.ones(2500)*self.scale_wh]
            else:
                feat = np.c_[yy.ravel(),xx.ravel(),np.ones(2500)*self.scale_wh,np.ones(2500)*self.scale_text]
            Z = self.svm_yf.decision_function(feat)
            hl = self.plot_data(Z, ax, xx, yy, c='black')
            h1 = ax.scatter(X_train[:,1], X_train[:,0], s=10, c=y_train, cmap=plt.cm.Paired)
            ax.set_title('extended SVM yield function in training')
            ax.set_xlabel('$theta/\pi$')
            ax.set_ylabel('$\sigma_{eq}/\sigma_y$')
            plt.show()
        return train_sc, test_sc
    
    def train_SVC(self, C=10, gamma=4, Nlc=36, Nseq=25, extend=True, plot=False, fontsize=16):
        '''Train SVC for all yield functions of the microstructures provided in msparam and for flow stresses to capture 
        work hardening. In first step, the 
        training data for each set is generated by creating stresses on the deviatoric plane and calculating their catgegorial
        yield function ("-1": elastic, "+1": plastic). Furthermore, axes in different dimensions for microstructural
        features are introduced that describe the relation between the different sets.
        
        Parameters
        ----------
        C     : float
            Parameter needed for training process, larger values lead to more flexibility (optional, default: 10)
        gamma : float
            Parameter of Radial Basis Function of SVC kernel, larger values, lead to faster decay of influence of individual 
            support vectors, i.e., to more short ranged kernels (optional, default: 4)
        Nlc   : int 
            Number of load cases to be condidered, will be overwritten if material has microstructure information (optional, default: 36)
        Nseq  : int
            Number of training as test data values to be generated in elastic regime, same number will be produced in plastic regime
            (optional, default: 25)
        extend : Boolean
            Indicate whether training data should be extended further into plastic regime (optional, default: True)
        plot  : Boolean
            Indicate if graphical output should be performed (optional, default: False)
        fontsize : int
            Fontsize for graph annotations (optional, default: 16)
        '''
        print('\n---------------------------\n')
        print('SVM classification training')
        print('---------------------------\n')
        'augment raw data and create result vector (yield function) for all data on work hardening and textures'
        if self.msparam is None:
            Npl = 1
            Ntext = 1
        else:
            Nlc   = self.msparam['Nlc']
            Npl   = self.msparam['Npl']
            Ntext = self.msparam['Ntext']
        if extend:
            Ne = 4
        else:
            Ne = 0
        N0 = Nlc*(2*Nseq+Ne) # total number of training data points per level of PEEQ for each microstructure
        Nt = Ntext*Npl*N0    # total numberof training data points
        xt = np.zeros((Nt,self.Ndof))
        yt = np.zeros(Nt)
        for k in range(Ntext):    # loop over all textures
            for j in range(Npl):  # loop over work hardening levels for each texture
                'create training data in entire deviatoric stress plane from raw data'
                if self.msparam is None:
                    sc_train, yf_train = self.create_sig_data(N=Nlc, Nseq=Nseq, extend=extend)
                else:
                    sc_train, yf_train = self.create_sig_data(syc=self.msparam['flow_stress'][k,j,:,:], 
                                                            sflow=self.msparam['flow_seq_av'][k,j], Nseq=Nseq, extend=True)
                #sc_test, yf_test   = self.create_sig_data(syc=self.msparam['flow_stress'][k,j,::12,:], Nseq=15, rand=False)
                i0 = (j + k*Npl)*N0
                i1 = i0 + N0
                xt[i0:i1,0:2] = sc_train
                if self.Ndof>=3:
                    xt[i0:i1,2] = self.msparam['work_hard'][j]
                if self.Ndof==4:
                    xt[i0:i1,3] = self.msparam['texture'][k]
                yt[i0:i1] = yf_train
        '''nth = int(N0/18)
        x_test = xt[::nth,0:2]
        y_test = yt[::nth]'''

        print('%i training data sets created from %i microstructures, with %i flow stresses in %i load cases' % (Nt, Ntext, Npl, Nlc))
        if np.any(np.abs(yt)<=0.99):
            print('Warning: result vector for yield function contains more categories than "-1" and "+1". Will result in higher dimensional SVC.')
            
        'Train SVC with data from all microstructures in data'
        train_sc, test_sc = self.setup_yf_SVM(xt, yt, cyl=True, C=C, gamma=gamma, fs=0.3, plot=False)   

        print(self.svm_yf)
        print("Training set score: {} %".format(train_sc))
        
        if plot:
            'plot ML yield loci with reference and test data'
            print('Plot ML yield loci with reference curve and test data')
            if self.Ndof>=3:
                print('Initial yield locus plotted together with flow stresses for PEEQ in range [%6.3f,%6.3f]' 
                      % (self.msparam['work_hard'][0], self.msparam['work_hard'][-1]))
            if self.Ndof==4:
                print('Initial yield locus plotted for texture parameter in range [%6.3f,%6.3f]' 
                      % (self.msparam['texture'][0], self.msparam['texture'][-1]))
                Npl = 1  # only plot initial yield surface

            ncol = 2
            nrow = int(Npl*Ntext/ncol + 0.95)
            fig = plt.figure(figsize=(20, 8*nrow))
            plt.subplots_adjust(hspace=0.3)
            theta = np.linspace(-np.pi,np.pi,36)
            for k in range(Ntext):
                self.set_microstructure('texture', self.msparam['texture'][k], verb=False)
                for j in range(0,Npl,np.maximum(1,int(Npl/5))):
                    'Warning: x_test and y_test should be setup properly above!!!'
                    ind = list(range((j+k*Npl)*N0,(j+k*Npl+1)*N0,int(0.5*N0/Nlc)))
                    y_test = yt[ind]
                    x_test = xt[ind,:]
                    peeq = self.msparam['work_hard'][j] - self.msparam['epc']
                    self.set_workhard(peeq)
                    iel = np.nonzero(y_test<0.)[0]
                    ipl = np.nonzero(np.logical_and(y_test>=0., x_test[:,0]<self.sflow*1.5))[0]
                    ax = plt.subplot(nrow, ncol, j+k*Npl+1, projection='polar')
                    plt.polar(x_test[ipl,1], x_test[ipl,0], 'r.', label='test data above yield point')
                    plt.polar(x_test[iel,1], x_test[iel,0], 'b.', label='test data below yield point')
                    if self.msparam is not None:
                        syc = self.msparam['flow_stress'][k,j,:,:]
                        plt.polar(syc[:,1], syc[:,0], '-c', label='reference yield locus')
                    'ML yield fct: find norm of princ. stess vector lying on yield surface'
                    snorm = sp_cart(np.array([self.sflow*np.ones(36)*np.sqrt(1.5), theta]).T)
                    x1 = fsolve(self.find_yloc, np.ones(36), args=snorm, xtol=1.e-5)
                    sig = snorm*np.array([x1,x1,x1]).T
                    s_yld = seq_J2(sig)
                    plt.polar(theta, s_yld, '-k', label='ML yield locus', linewidth=2)
                    if self.msparam is None:
                        plt.title=self.name
                    else:
                        plt.title('Flow stress, PEEQ='+str(self.msparam['work_hard'][j].round(decimals=4))+', TP='
                                  +str(self.msparam['texture'][k].round(decimals=2)), fontsize=fontsize)
                    #plt.xlabel(r'$\theta$ (rad)', fontsize=fontsize-2)
                    #plt.ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=fontsize-2)
                    plt.legend(loc=(.95,0.85),fontsize=fontsize-2)
                    plt.tick_params(axis="x",labelsize=fontsize-4)
                    plt.tick_params(axis="y",labelsize=fontsize-4)
            plt.show()


    def calc_fgrad(self, sig, peeq=0., seq=None, ana=False):
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
                    dsa = np.dot(sig,a_vec)
                    dsb = np.dot(sig,b_vec)
                    sc = dsa + 1j*dsb
                    z = -1j*((a_vec+1j*b_vec)/sc - dseqds/vn)
                    J[:,1] = np.real(z)
                return J
            N = len(sig)
            x = np.zeros((N,self.Ndof))
            x[:,0] = seq_J2(sig)/self.scale_seq - 1.
            x[:,1] = polar_ang(sig)/np.pi
            if self.Ndof>=3:
                x[:,2] = (peeq+self.msparam['epc'])/self.scale_wh - 1.
            if self.Ndof==4:
                x[:,3] = self.ms_cur/self.scale_text - 1.
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
        '''Inititalize and train SVM regression for gradient evaluation

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
                sprinc[i,:] = Stress(sig[i,:]).p
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
        self.sy = sy   # yield strength
        self.sflow = sy # initialize flow stress (will be increase with work hardening)
        self.khard = khard  # work hardening rate (linear w.h.)
        self.Hpr = khard/(1.-khard/self.E)  # constant for w.h.
        self.hill = np.array(hill)  # Hill anisotropic parameters
        self.drucker = drucker   # Drucker-Prager parameter: weight of hydrostatic stress
        self.Tresca = Tresca

    def microstructure(self, param, grain_size=None, grain_shape=None, porosity=None):
        '''Define microstructural parameters of the material: crstallographic texture, grain size, grain shape
        and porosity.

        Parameters
        ----------
        data : Object of class ``Data``
            Assign data to microstructure of material
        grain_size : float
            Average grain size of material (optional)
        grain_shape : (3,) array
            Lengths of the three main axes of ellipsoid describing grain shape (optional)
        porosity : float
            Porosity of the material
        '''

        'import dictionary will all microstructure parameters resulting from data module'
        self.msparam = param 
        if param['Npl']>1 and self.Ndof<3:
            self.Ndof = 3   # add dof for work hardening
        if param['Ntext']>1 and self.Ndof<4:
            self.Ndof = 4   # add dof for texture
        'add additional microstructure information'
        self.msparam['grain_size'] = grain_size  # add further microstructure parameters
        self.msparam['grain_shape'] = grain_shape
        self.msparam['porosity'] =  porosity        
        
    def set_microstructure(self, active, current, verb=True):
        '''Set current microstrcuture of material if material has variable microstructure
        
        Parameters
        ----------
        active  : str
            Active microstructure component from dictionary ``msparam``
        current : float
            Value of currently active microstructure parameter
        verb : Boolean
            Be verbose 
            
        Yields
        ------
        Material.ms_act : str
            Keyword of active microstructure component, e.g. 'texture'
        Material.ms_cur : float
            Current value of microstructural parameter for active microstructure
        Material.sy  : float
            Yield stength is redefined accoring to texture parameter
        Material.khard, Material.Hpr : float
            Work hardening parameters are redefined according to texture parameter
        '''
        self.ms_act = active
        self.ms_cur = current
        hh = self.msparam[active] - current
        self.ms_index = np.argmin(np.abs(hh))
        'redefine plasticity parameters according to texture parameter'
        #print(active, current, self.msparam[active],self.msparam['flow_seq_av'][self.ms_index,0])
        self.sy = np.interp(current, self.msparam[active],self.msparam['flow_seq_av'][:,0])
        if self.msparam['Npl'] > 1:
            'set strain hardening parameters to initial value for selected texture'
            ds = self.msparam['flow_seq_av'][self.ms_index,1] - self.msparam['flow_seq_av'][self.ms_index,0] # assuming isotropic hardening
            de = self.msparam['work_hard'][1] - self.msparam['work_hard'][0]
            self.Hpr =  ds/de # linear work hardening rate b/w values for w.h. in data
            self.khard = self.Hpr/(1.+self.Hpr/self.E)  # constant for w.h.
        if verb:
            print('New microstructure selected: ',self.msparam['ms_name'][self.ms_index],
                  '(Norm parameter:', self.msparam[active][self.ms_index],')')
            print('Actual microstructure parameter:', current)
            print('Yield strength:',self.sy,'MPa')
            print('Work hardening modulus:',self.khard,'MPa')
        
    def set_workhard(self, peeq, verb=False):
        '''Set current status of work hardening.
        
        Parameters
        ----------
        peeq : float
            Current value of equivalent plastic strain 
            
        Yields
        ------
        Material.wh_cur : float
            Work hardening parameter or microstructural value of equiv. plastic
            strain, which is the mechanical peeq plus the critical value for yield onset.
        Material.sflow : float
            Flow stress associated with work hardening parameter
        '''
        self.wh_cur = peeq + self.msparam['epc']
        self.sflow  = np.interp(self.wh_cur, self.msparam['work_hard'], 
                                self.msparam['flow_seq_av'][self.ms_index,:])
        hh = self.msparam['work_hard'] - self.wh_cur
        if self.msparam['Npl'] > 1:
            ind = np.nonzero(hh<=0.)[0]
            i0 = ind[-1]  # index of w.h. parameter with value just below peeq
            i1 = i0 + 1
            if i1<self.msparam['Npl']:
                ds = self.msparam['flow_seq_av'][self.ms_index,i1] - self.msparam['flow_seq_av'][self.ms_index,i0] # assuming isotropic hardening
                de = self.msparam['work_hard'][i1] - self.msparam['work_hard'][i0]
                self.Hpr =  ds/de # linear work hardening rate b/w values for w.h. in data
                self.khard = self.Hpr/(1.+self.Hpr/self.E)  # constant for w.h.
            else:
                'assume ideal plasticity after last data point'
                self.khard = 0.
                self.Hpr = 0.
        if verb:
            print('Currect work hardening parameter:', self.wh_cur)
            print('Current flow stress (MPa): ', self.sflow)
            print('Yield strength:',self.sy,'MPa')
            print('Work hardening modulus:',self.khard,'MPa')

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
        peeq = eps_eq(epl)     # equiv. plastic strain
        yfun = self.calc_yf(sig+Cel@deps, peeq=peeq)
        #for DEBUGGING
        '''yf0  = self.calc_yf(sig, peeq=peeq)
        if yf0<-ptol and yfun>ptol and peeq<1.e-5:
            if self.ML_yf:
                ds = Cel@deps
                yfun = self.ML_full_yf(sig+ds)
            print('*** Warning in epl_dot: Step crosses yield surface')
            print('sig, epl, deps, yfun, yf0, peeq,caller', sig, epl, deps, yfun, yf0, peeq, sys._getframe().f_back.f_code.co_name)
            yfun=0. # catch error that can be produced for anisotropic materials'''
        if (yfun<=ptol):
            pdot = np.zeros(6)
        else:
            a = np.zeros(6)
            a[0:3] = self.calc_fgrad(Stress(sig).p)
            hh = a.T @ Cel @ a + 4.*self.Hpr
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
        a[0:3] = self.calc_fgrad(Stress(sig).princ)
        hh = a.T @ Cel @ a + 4.*self.Hpr
        ca = Cel @ a
        Ct = Cel - np.kron(ca, ca).reshape(6,6)/hh
        return Ct

    def create_sig_data(self, N=None, mat_ref=None, syc=None, Nseq=12, sflow=None, offs=0.01, extend=False, rand=False):
        '''Function to create consistent data sets on the deviatoric stress plane
        for training or testing of ML yield function. Either the number "N" of raw data points, i.e. load angles, to be 
        generated and a reference material "mat_ref" has to be provided, or a list of raw data points "syc" with 
        cylindrical stress tensors lying on the yield surface serves as input. Based on the raw data, stress tensors 
        from the yield locus are distributed into the entire deviatoric space, by linear downscaling into the elastic 
        region and upscaling into the plastic region. Data is created in form of cylindrical stresses that lie densly 
        around the expected yield locus and more sparsely in the outer plastic region.

        Parameters
        ----------
        N    : int
            Number of load cases (polar angles) to be created (optional,
            either N and mat_ref or syc must be provided)
        mat_ref : object of class ``Material``
            reference material needed to calculate yield function if only N is provided (optional, ignored if syc is given)
        syc: (N,3) array
            List of cyl. stress tensors lying on yield locus from which training data in entire deviatoric stress plane is created
            (optional, either syc or N and mat_ref must be provided)
        Nseq : int
            Number of equiv. stresses generated up to yield strength (optional, default: 12)
        sflow : float
            Expected flow stress of data set (optional, default: self.sy)
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
        yt : (M,) array
            Result vector of categorial yield function for supervised training
        '''
        if sflow is None:
            sflow = self.sy
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
        seq = np.linspace(offs, 2*sflow, 2*Nseq)
        if extend:
            # add training points in plastic regime to avoid fallback of SVC decision fct. to zero
            seq = np.append(seq, np.array([2.4, 3., 4., 5.])*sflow)
        Nd = len(seq)
        st = np.zeros((N*Nd,2))
        yt = np.zeros(N*Nd)
        for i in range(Nd):
            j0 = i*N
            j1 = (i+1)*N
            st[j0:j1, 0] = seq[i]
            st[j0:j1, 1] = theta
            if syc is None:
                yt[j0:j1] = np.sign(mat_ref.calc_yf(sp_cart(st[j0:j1,:])))
            else:
                yt[j0:j1] = np.sign(seq[i] - sc)
        return st, yt

    def find_yloc(self, x, sp):
        '''Function to expand unit stresses by factor and calculate yield function;
        used by search algorithm to find zeros of yield function.

        Parameters
        ----------
        x : 1d-array
            Multiplyer for stress
        sp : (N,3) array
            Principal stress
        
        Returns
        -------
        f : 1d-array
            Yield function evaluated at sig=x.sp
        '''
        y = np.array([x,x,x]).transpose()
        f = self.calc_yf(y*sp)
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
            plt.subplots_adjust(wspace=0.2)
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
        '''Use pylabfea.model to calculate material strength and stress-strain data along a given load path.

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
            fe=Model(dim=2,planestress=True)
            fe.geom([size], LY=size) # define section in absolute length
            fe.assign([self])        # assign material to section
            fe.bcleft(0.)            # fix lhs nodes in x-direction
            fe.bcbot(0.)             # fix bottom nodes in y-direction
            fe.bcright(vbc1, nbc1)   # define BC in x-direction
            fe.bctop(vbc2, nbc2)     # define BC in y-direction
            fe.mesh(NX=Nel, NY=Nel)  # create mesh
            fe.solve(verb=verb,min_step=min_step)      # solve mechanical equilibrium condition under BC
            seq  = self.calc_seq(fe.sgl)   # store time dependent mechanical data of model
            eeq  = eps_eq(fe.egl)
            peeq = eps_eq(fe.epgl)
            iys = np.nonzero(peeq<1.e-6)
            ys = seq[iys[0][-1]]
            self.prop[sel]['ys']   = ys
            self.prop[sel]['seq']  = seq
            self.prop[sel]['eeq']  = eeq
            self.prop[sel]['peeq'] = peeq
            seq  = seq_J2(fe.sgl)   # store time dependent mechanical data of model
            eeq  = eps_eq(fe.egl)
            peeq = eps_eq(fe.epgl)
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
            u1 = 0.4*eps*size
            u2 = 0.4*eps*size
            calc_strength(u1, 'disp', u2, 'disp', 'et2')
            self.prop['et2']['style'] = '-k'
            self.prop['et2']['name']  = 'equibiax'
            return
        def calc_ect():
            u1 = -0.8*eps*size
            u2 = 0.8*eps*size
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
