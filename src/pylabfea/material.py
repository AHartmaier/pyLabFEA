# Module pylabfea.material
"""Module pylabfea.material introduces class ``Material`` that contains attributes and methods
needed for elastic-plastic material definitions in FEA. It also enables the training of 
machine learning algorithms as yield functions for plasticity.
The module pylabfea.model is used to calculate mechanical properties of a defined material 
under various loading conditions.

uses NumPy, ScipPy, MatPlotLib, sklearn, pickle, and pyLabFEA.model

Version: 4.1 (2022-01-23)
Authors: Alexander Hartmaier, Ronak Shoghi, ICAMS/Ruhr University Bochum, Germany
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)"""
from pylabfea.basic import a_vec, b_vec, \
    eps_eq, sig_polar_ang, yf_tolerance, \
    sig_eq_j2, sig_cyl2princ, sig_princ, sig_dev, sig_princ2cyl
from pylabfea.model import Model
from pylabfea.training import load_cases
from scipy.optimize import root_scalar
from scipy.optimize import fsolve
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
import pickle
from sklearn.model_selection import GridSearchCV
import platform
import getpass


# ==========================
# define class for materials
# ==========================


class Material(object):
    """Define class for Materials including material parameters (attributes), constitutive relations (methods)
    and derived properties und various loading conditions (dictionary)

    Parameters
    ----------
    name : str
        Name of material (optional, default: 'Material')
    num : int
        Material number (optional, default: 1)

    Attributes
    ----------
    name    : str
        Name of material
    num     : int
        Material number
    sy      : float
        Yield strength
    ML_yf   : Boolean
        Existence of trained machine learning (ML) yield function (default: False)
    ML_grad : Boolean
        Existence of trained ML gradient (default: False)
    tresca  : Boolean
        Indicate if Tresca equivalent stress should be used (default: False)
    hill_3p  : Boolean
        Indicates whether 3-paramater Hill model should be used (default: False)
    hill_6p  : Boolean
        Indicates whether 6-paramater Hill model should be used (default: False)
    barlat  : Boolean
        Indicate if Barlat equivalent stress should be used (default: False)
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
    whdat   : Boolean
        Indicates existence of work hardening data
    txdat   : Boolean
        Indicates existance of data for different textures
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
    msg :
        Messages that can be retrieved: 'yield_fct', 'gradient', 'nsteps', 'equiv'
    """

    # Methods
    # elasticity: define elastic material parameters C11, C12, C44
    # plasticity: define plastic material parameter sy, khard
    # epl_dot: calculate plastic strain rate

    def __init__(self, name='Material', num=1):
        self.khard = None
        self.ind_tx = None
        self.ind_wh = None
        self.epc = None
        self.Nset = None
        self.grid = None
        self.C_yf = None
        self.svm_yf = None
        self.gam_yf = None
        self.scale_text = None
        self.scale_wh = None
        self.scale_seq = None
        self.CV = None
        self.C11 = None
        self.C12 = None
        self.C44 = None
        self.name = name
        self.num = num
        self.sy = None  # Elasticity will be considered unless sy is set
        self.ML_yf = False  # use conventional plasticity unless trained ML functions exists
        self.ML_grad = False  # use conventional gradient unless ML function exists
        self.tresca = False  # use J2 or Hill equivalent stress unless defined otherwise
        self.barlat = False  # Use Barlat equiv. stress if parameters are given
        self.msparam = None  # parameters for primary microstructure
        self.whdat = False
        self.txdat = False
        self.Ndof = 2
        self.hill_6p = False
        self.sdim = None  # dimensionality of stress space to be considered in ML flow rules
        self.root_method = 'brentq'
        self.msg = {
            'yield_fct': None,
            'gradient': None,
            'nsteps': 0,
            'equiv': None
        }
        self.prop = {  # stores strength and stress-strain data along given load paths
            'stx': {'ys': None, 'seq': None, 'eeq': None, 'peeq': None, 'style': None, 'name': None},
            'sty': {'ys': None, 'seq': None, 'eeq': None, 'peeq': None, 'style': None, 'name': None},
            'et2': {'ys': None, 'seq': None, 'eeq': None, 'peeq': None, 'style': None, 'name': None},
            'ect': {'ys': None, 'seq': None, 'eeq': None, 'peeq': None, 'style': None, 'name': None}
        }
        self.propJ2 = {  # stores J2 equiv strain data along given load paths
            'stx': {'ys': None, 'seq': None, 'eeq': None, 'peeq': None},
            'sty': {'ys': None, 'seq': None, 'eeq': None, 'peeq': None},
            'et2': {'ys': None, 'seq': None, 'eeq': None, 'peeq': None},
            'ect': {'ys': None, 'seq': None, 'eeq': None, 'peeq': None}
        }
        self.sigeps = {  # calculates strength and stress strain data along given load paths
            'stx': {'sig': None, 'eps': None, 'epl': None},
            'sty': {'sig': None, 'eps': None, 'epl': None},
            'et2': {'sig': None, 'eps': None, 'epl': None},
            'ect': {'sig': None, 'eps': None, 'epl': None}
        }

    # =================================================================
    # subroutines for elastic and plastic material behavior
    # =================================================================
    def response(self, sig, epl, deps, CV, maxit=50):
        """Calculate non-linear material response to deformation defined by load step, 
        corresponds to user material function.
        
        Parameters
        ----------
        sig : (6,) array
            Voigt stress tensor at start of load step (=end of previous load step)
        epl : (6,) array
            Voigt plastic strain tensor at start of load step
        deps : (6,) array
            Voigt strain tensor defining deformation (=load step)
        CV : (6,6) array
            Voigt elastic tensor
        maxit : int
            Maximum number of iteration steps (optional, default= 5)
            
        Returns
        -------
        fy1 : real
            Yield function at end of load step (indicates whether convergence is reached)
        sig : (6,) array
            Voigt stress tensor at end of load step
        depl : (6,) array
            Voigt tensor of plastic strain increment at end of load step
        grad_stiff : (6,6) array
            Tangent material stiffness matrix (d_sig/d_eps) at end of load step
        
        """
        sh = sig.shape
        if sh != (6,) and sh != (3,):
            raise ValueError(
                'Only individual stress tensors supported in material.response. Shape of argument is {}'.format(sh))
        # initialize quantities needed
        sig = np.array(sig)  # produce copy of sig to avoid changes to original
        depl = np.zeros(6)  # initialize plastic strain increment
        toler = yf_tolerance * self.get_sflow(epl)
        dsig = CV @ deps  # predictor of stress increment
        st_scal = 1.
        niter = 0

        # evaluate yield function for elastic predictor step
        if self.ML_yf:
            fy1 = self.ML_full_yf(sig + dsig, epl=epl)
        else:
            fy1 = self.calc_yf(sig + dsig, epl=epl)
        if fy1 < toler:
            # purely elastic load step
            sig += dsig  # update stress
            grad_stiff = np.array(CV)  # gradient stiffness is elastic stiffness
        else:
            # elastic predictor step reaches to plastic regime
            fy0 = self.calc_yf(sig, epl=epl)  # yield fct. at start of load step
            if fy0 < -0.15:
                # load step starts in elastic regime and ends in plastic regime
                # must be split into elastic and plastic parts
                if self.ML_yf:
                    # for categorial ML yield function, calculate fy0 as distance to yield surface
                    fy0 = self.ML_full_yf(sig)  # distance of initial stress state to yield locus
                st_scal += fy0 / self.calc_seq(dsig)
                deps_el = deps * (1. - st_scal)  # calculate elastic part of load step
                sig += CV @ deps_el  # update stress which lies now on yield locus
                grad_stiff = CV * (1. - st_scal)  # contribution to gradient stiffness
                deps_r = deps - deps_el  # remaining load step
            else:
                # load step starts on yield locus
                deps_r = np.array(deps)  # create new variable to prevent deps from being changed
                grad_stiff = np.zeros((6, 6))  # initialize stiffness matrix

            # do a first trial step with full deps_r
            ddepl = self.epl_dot(sig, epl, CV, deps_r)  # plastic strain increment
            t_stiff = self.C_tan(sig, CV, epl=epl)  # tangent stiffness
            eplt = epl + depl + ddepl
            dsig = t_stiff @ deps_r  # update stress with current tangent stiffness
            # evaluate yield function at the end of this step
            if self.ML_yf:  # and self.msparam is not None:
                fy1 = self.ML_full_yf(sig + dsig, epl=eplt)
            else:
                fy1 = self.calc_yf(sig + dsig, epl=eplt)

            # if remaining step deps_r is too large, better to subdivide it
            if fy1 > toler:
                # subdivide load step
                deps_r /= maxit
                nsteps = maxit
            else:
                nsteps = 1

            for niter in range(nsteps):
                # at this stage, the initial stress sig should lie on the yield locus
                # and the yield function fy1 points outside
                # in the following, the remaining load step is performed 
                ddepl = self.epl_dot(sig, epl, CV, deps_r)  # plastic strain increment
                t_stiff = self.C_tan(sig, CV, epl=epl)  # tangent stiffness
                eplt = epl + depl + ddepl
                dsig = t_stiff @ deps_r  # update stress with current tangent stiffness
                sig += dsig
                # evaluate yield function at the end of this step
                if self.ML_yf:  # and self.msparam is not None:
                    fy1 = self.ML_full_yf(sig, epl=eplt)
                else:
                    fy1 = self.calc_yf(sig, epl=eplt)

                if fy1 > toler:
                    # the step size was too large because it ends outside of the yield locus
                    # a correction is needed
                    # total strain must remain constant during this correction
                    # calculate compliance tensor
                    SV = np.zeros((6, 6))
                    i = (3 if CV[2, 2] > 1. else 2)
                    hh = np.linalg.inv(CV[0:i, 0:i])  # calculate inverse of sub-tensor
                    SV[0:i, 0:i] = hh
                    for i in range(3, 6):
                        if CV[i, i] > 1.: SV[i, i] = 1. / CV[i, i]

                    dsig = sig * fy1 / self.calc_seq(sig)  # excess stress tensor
                    sig -= dsig  # reduce stress about excess stress
                    ddepl += SV @ dsig  # add plastic strain to balance the elastic strain, violation of volume 
                    # conservation! 
                    eplt = epl + depl + ddepl
                    # calculate tangent stiffness matrix for correction step
                    a = np.array([[deps_r[0], 0., 0., 0., deps_r[2], deps_r[1]],
                                  [0., deps_r[1], 0., deps_r[2], 0., deps_r[0]],
                                  [0., 0., deps_r[2], deps_r[1], deps_r[0], 0.]])
                    y = np.linalg.lstsq(a, dsig[0:3], rcond=None)
                    x = y[0]
                    Ct = np.zeros((6, 6))
                    Ct[0:3, 0:3] = np.array([[x[0], x[5], x[4]],
                                             [x[5], x[1], x[3]],
                                             [x[4], x[3], x[2]]])
                    t_stiff -= Ct
                    # update yield function
                    if self.ML_yf:  # and self.msparam is not None:
                        fy1 = self.ML_full_yf(sig, epl=eplt)
                    else:
                        fy1 = self.calc_yf(sig, epl=eplt)
                grad_stiff += t_stiff * st_scal / nsteps
                depl += ddepl
        self.msg['nsteps'] = niter
        return fy1, sig, depl, grad_stiff

    def calc_yf(self, sig, epl=None, ana=False, pred=False):
        """Calculate yield function

        Parameters
        ----------
        sig  : (sdim,) or (N, sdim) array
            Stresses (arrays of Voigt or principal stresses)
        epl : (sdim, ) array
            Equivalent plastic strain tensor (optional, default: None)
        ana  : Boolean
            Indicator if analytical solution should be used, rather than ML yield fct (optional, default: False)
        pred : Boolean
            Indicator if prediction value should be returned, rather than decision function (optional, default: False)

        Returns
        -------
        f    : flot or 1d-array
            Yield function for given stress (same length as sig)
        """
        sh = np.shape(sig)
        if epl is None:
            epl = np.zeros(self.sdim)
        if type(epl) in (float, np.float64):
            # if only PEEQ is provided convert it into an arbitrary plastic strain tensor
            epl = epl * np.array([1., -0.5, -0.5, 0., 0., 0.])
        if self.ML_yf and not ana:
            if sh == (3,) or sh == (6,):
                sig = np.array([sig])
                N = 1
            else:
                N = len(sig)
            x = np.zeros((N, self.Ndof))
            if self.sdim == 3:
                x[:, 0] = sig_eq_j2(sig) / self.scale_seq - 1.
                x[:, 1] = sig_polar_ang(sig) / np.pi
            else:
                sig = sig_dev(sig)
                if sh == (N, 6) or sh == (6,):
                    x[:, 0:6] = sig[:, 0:6] / self.scale_seq
                else:
                    x[:, 0:3] = sig[:, 0:3] / self.scale_seq
            if self.whdat:
                x[:, self.ind_wh:self.ind_wh + self.sdim] = epl / self.scale_wh
            if self.txdat:
                ih = self.ind_tx
                for i in range(self.Nset):
                    x[:, ih + i] = self.tx_cur[i] / self.scale_text[i] - 1.
            if pred:
                # use prediction, returns either -1 or +1
                f = self.svm_yf.predict(x)
                self.msg['yield_fct'] = 'ML_yf-predict'
            else:
                # use continuous decision function in range [-1,+1]
                f = self.svm_yf.decision_function(x)
                self.msg['yield_fct'] = 'ML_yf-decision-fct'
            if N == 1:
                f = f[0]
        else:
            f = self.calc_seq(sig) - self.get_sflow(epl)
            self.msg['yield_fct'] = 'analytical'
        return f

    def ML_full_yf(self, sig, epl=None, ld=None, verb=True):
        """Calculate full ML yield function as distance of a single given stress
        tensor to the yield locus in loading direction.
        
        Parameters
        ----------
        sig : (sdim,) array
            Voigt stress tensor
        epl : (sdim,) array
            Equivalent plastic strain (optional, default: 0)
        ld  : (6,) array
            Vector of loading direction in princ. stress space (optional)
        verb : Boolean
            Indicate whether to be verbose in text output (optional, default: False)

        Returns
        -------
        yf : float
            Full ML yield function, i.e. distance of sig to yield locus in ld-direction
        """
        if epl is None:
            epl = np.zeros(self.sdim)
        sh = sig.shape
        if sh != (3,) and sh != (6,):
            raise ValueError(
                'Only individual stress tensors supported in material.ML_full_yf. Shape of argument is {}'.format(sh))
        seq = self.calc_seq(sig)
        sflow = self.get_sflow(epl)
        if seq < 0.01 and ld is None:
            # return conservative estimate of yield function for small stresses
            # and unknown loading direction
            yf = seq - 0.85 * sflow
        else:
            if ld is None:
                # construct unit stress in loading direction
                su = sig / seq
            else:
                # convert ld to unit stress
                hh = np.linalg.norm(ld)
                if hh < 1.e-3:
                    warnings.warn('ML_full_yf called with inconsistent ld={}'.format(ld))
                    print('calling routine: ', sys._getframe().f_back.f_code.co_name)
                    hh = 1.
                    ld = np.zeros(self.sdim)
                    ld[0] = 1.
                su = ld[0:self.sdim] * np.sqrt(1.5) / hh
            if np.any(np.isnan(sig)) or np.any(np.isnan(su)):
                print('NaN detected (MP_full_yf): sig={}, su={}'.format(sig, su))
                print('SEQ={}, ld={}, peeq={}'.format(seq, ld, eps_eq(epl)))
            x0 = sflow  # starting value of yield point search
            if su[0] * su[1] < -1.e-5:
                # correction of starting value for pure shear cases
                if self.tresca:
                    x0 *= 0.4
                else:
                    x0 *= 0.5
            x1 = x0
            while self.find_yloc_scalar(x0, su, epl) >= 0. and x0 > 0.01:
                # find x0 with negative yield fct
                x0 *= 0.95
            while self.find_yloc_scalar(x1, su, epl) < 0. and x1 < 4. * sflow:
                # find x1 with positive yield fct
                x1 *= 1.05
            f0 = self.find_yloc_scalar(x0, su, epl)
            f1 = self.find_yloc_scalar(x1, su, epl)
            if f0 * f1 > 0.:
                warnings.warn('ML_full_yf: Could not bracket yield function: ' \
                              + 'sig={}, x0={}, f0={}, x1={}, f1={}'.format(sig, x0, f0, x1, f1))
                return seq - 0.85 * sflow

            res = root_scalar(self.find_yloc_scalar, method=self.root_method, bracket=[x0, x1],
                              args=(su, epl), xtol=1.e-5)
            xs = res.root
            if res.converged and xs < 4. * sflow:
                # zero of ML yield fct. detected at x1*su
                yf = seq - xs * self.calc_seq(su)
            else:
                # zero of ML yield fct. not found: get conservative estimate
                yf = seq - 0.85 * sflow
                if verb:
                    ys = self.find_yloc_scalar(xs, su, epl)
                    warnings.warn('ML_full_yf')
                    print('*** detection not successful. yf={}, seq={}, ld={}, su:{}'.format(yf, seq, ld, su))
                    print('*** optimization result (x1={},y1={},msg={}):'.format(xs, ys, res))
        return yf

    def find_yloc(self, x, su, epl=None):
        """Function to expand unit stresses by factor and calculate yield function;
        used by search algorithm to find zeros of yield function.

        Parameters
        ----------
        x : (N,)-array
            Multiplyer for stress
        su : (N,sdim) array
            Unit stress
        epl : float
            Equivalent plastic strain (optional, default: 0)
        
        Returns
        -------
        f : (N,)-array
            Yield function evaluated at sig=x.sp
        """

        f = self.calc_yf(x[:, None] * su, epl=epl)
        return f

    def find_yloc_scalar(self, x, su, epl=None):
        """Function to expand unit stresses by factor and calculate yield function;
        used by search algorithm to find zeros of yield function.

        Parameters
        ----------
        x  : float
            Multiplier for stress
        su : (sdim,) array
            Unit stress
        epl : float
            Equivalent plastic strain (optional, default: 0)
        
        Returns
        -------
        f : float
            Yield function evaluated at sig=x.sp
        """

        f = self.calc_yf(x * su, epl=epl)
        return f

    def calc_seq(self, sig):
        """Calculate generalized equivalent stress from stress tensor;
        equivalent J2 stress for isotropic flow behavior and tension compression invariance;
        Hill-type approach for anisotropic plastic yielding;
        Drucker-like approach for tension-compression asymmetry;
        Barlat 2004-18p model for plastic anisotropy;
        Tresca equivalent stress
        
        Step 1: transform input into 
          (i)   sig: (N,6)-array of Voigt stresses (zeros added if input is princ. stresses)  
          (ii)  sp: (N,3)-array of principal stresses  
          
          N=1 if input is single stress in which case return value is of type float
        
        Step 2: Call appropriate subroutines for evaluation of equiv. stress. Currently supported are: 
          (i)   Tresca
          (ii)  Barlat Yld2004-18p
          (iii) Hill 3-parameter (hill_3p) or 6-parameter (hill_6p)
          (iv) von Mises/J2 (special case of Hill with all coefficients equal 1, 
                             material independent, works also for elastic and ML materials)  
        
        Parameters
        ----------
        sig : (sdim,) or (N,sdim) array
            Stress values (for dim=3 principal stresses are assumed, otherwise Voigt stress)

        Returns
        -------
        seq : float or (N,) array
            Hill-Drucker-type equivalent stress
        """

        N = len(sig)
        sh = np.shape(sig)
        # Step 1: Transform input
        if sh == (3,):
            N = 1  # sp is single principal stress vector
            sp = np.array([sig])
            sig = np.array([[sig[0], sig[1], sig[2], 0, 0, 0]])
        elif sh == (N, 3):
            sp = np.array(sig)
            sig = np.append(sig, np.zeros((N, 3)), axis=1)
        elif sh == (6,):
            N = 1
            sp = sig_princ(sig)[0]
            sp = np.array([sp])
            sig = np.array([sig])
        elif sh == (N, 6):
            sp = sig_princ(sig)[0]
        else:
            print('*** calc_seq: N={}, sh={}, caller={}'.format(N, sh, sys._getframe().f_back.f_code.co_name))
            raise TypeError('Unknown format of stress in calc_seq')

        # Step 2: call subroutines or evaluate von Mises/J2 equiv stress
        if self.tresca:
            # calculate Tresca equiv. stress
            seq = np.amax(sp, axis=1) - np.amin(sp, axis=1)
        elif self.barlat:
            # calculate Baralat equiv. stress
            seq = np.zeros(N)
            for i in range(N):
                seq[i] = self.calc_seqB(sig[i, :])
        else:
            # calculate J2 or Hill equiv. stress
            I1 = np.sum(sp, axis=1) / 3.  # hydrostatic stress as 1st invariant
            if self.sy is None:
                # elastic material
                hp = np.ones(3)
                d0 = 0.
            else:
                hp = self.hill
                d0 = self.drucker

            # consider anisotropy in flow behavior in second invariant in Hill-type approach
            if self.hill_6p:
                # equiv. stress for full 6-parameter Hill model
                I2 = hp[0] * np.square(sig[:, 0] - sig[:, 1]) + \
                     hp[1] * np.square(sig[:, 1] - sig[:, 2]) + \
                     hp[2] * np.square(sig[:, 2] - sig[:, 0]) + \
                     6. * hp[3] * np.square(sig[:, 3]) + \
                     6. * hp[4] * np.square(sig[:, 4]) + \
                     6. * hp[5] * np.square(sig[:, 5])
                I2 *= 0.5
                self.msg['equiv'] = '6-parameter Hill, full Voigt stress'
                # print('Full stress', np.sqrt(I2))
            else:
                # standard: equiv. stress based on princ. stresses with 3-parameter Hill model
                # calculate Hill or J2 equiv. stress (latter is default, all Hill parameters = 1)
                d12 = sp[:, 0] - sp[:, 1]
                d23 = sp[:, 1] - sp[:, 2]
                d31 = sp[:, 2] - sp[:, 0]
                I2 = 0.5 * (hp[0] * np.square(d12) + hp[1] * np.square(d23) + hp[2] * np.square(d31))
                self.msg['equiv'] = '3-parameter Hill'
            # eqiv stress including hydrostatic stress for tension-compression asymmetry
            seq = np.sqrt(I2) + d0 * I1  # generalized eqiv. stress
        if N == 1:
            seq = seq[0]
        return seq

    def calc_seqB(self, sv):
        """Calculate equivalent stress based on Yld2004-18p yield function 
        proposed by Barlat et al, Int. J. Plast. 21 (2005) 1009
        
        Parameters
        ----------
        sv : (6,) array
            Voigt stress tensor
            
        Returns
        -------
        seq : float
            Equivalent stress
        """
        sd = sig_dev(sv)
        st1 = self.Bar_m1 @ sd  # first linearly transformed stress deviator s_tilda_'
        st2 = self.Bar_m2 @ sd  # second linearly transformed stress deviator s_tilda_''
        Stp1 = sig_princ(st1)[0]  # principal stress of transformed stress
        Stp2 = sig_princ(st2)[0]
        a = self.barlat_exp
        seq = np.abs(Stp1[0] - Stp2[0]) ** a + np.abs(Stp1[0] - Stp2[1]) ** a + np.abs(Stp1[0] - Stp2[2]) ** a + \
              np.abs(Stp1[1] - Stp2[0]) ** a + np.abs(Stp1[1] - Stp2[1]) ** a + np.abs(Stp1[1] - Stp2[2]) ** a + \
              np.abs(Stp1[2] - Stp2[0]) ** a + np.abs(Stp1[2] - Stp2[1]) ** a + np.abs(Stp1[2] - Stp2[2]) ** a
        seq = (0.25 * seq) ** (1. / a)
        return seq

    def calc_fgrad(self, sig, epl=None, seq=None, ana=False):
        """Calculate gradient to yield surface. Three different methods can be used: (i) analytical gradient to Hill-like yield
        function (default if no ML yield function exists - ML_yf=False), (ii) gradient to ML yield function (default if ML yield
        function exists - ML_yf=True; can be overwritten if ana=True), (iii) ML gradient fitted seperately from ML yield function
        (activated if ML_grad=True and ana=False)

        Parameters
        ----------
        sig : (sdim,) or (N,sdim) array
            Stress value (Pricipal stress or full stress tensor)
        epl : (sdim,) array
            Plastic strain tensor (optional, default = 0)
        seq : float or (N,) array
            Equivalent stresses (optional)
        ana : Boolean
            Indicator if analytical solution should be used, rather than ML yield fct (optional, default: False)

        Returns
        -------
        fgrad : (sdim,), (N,sdim) array
            Gradient to yield surface at given position in stress space, same dimension as sdim
        """
        if epl is None:
            epl = np.zeros(self.sdim)
        N = len(sig)
        sh = np.shape(sig)
        if sh == (3,) or sh == (6,):
            N = 1  # sig is vector of principal stresses
            sig = np.array([sig])
        elif sh != (N, self.sdim):
            raise ValueError('Unknown format of stress in calc_fgrad')
        fgrad = np.zeros((N, self.sdim))
        if self.ML_grad and not ana:
            # use SVR fitted to gradient
            sig = sig / self.sy
            fgrad[:, 0] = self.svm_grad0.predict(sig) * self.gscale[0]
            fgrad[:, 1] = self.svm_grad1.predict(sig) * self.gscale[1]
            fgrad[:, 2] = self.svm_grad2.predict(sig) * self.gscale[2]
            self.msg['gradient'] = 'SVR gradient'
        elif self.ML_yf and not ana:
            # use gradient of SVC yield fct. in stress space
            # gradient of SVC kernel function w.r.t. feature vector
            def grad_rbf(x, xp):
                # calculate gradient of radial basis function kernel
                # x.shape=(Ndof,)
                # xp.shape=(Nsv,Ndof)
                # grad.shape=(Nsv,Ndof)
                hv = x - xp
                hh = np.sum(hv * hv, axis=1)  # ||x-x'||^2=sum_i(x_i-x'_i)^2
                k = np.exp(-self.gam_yf * hh)
                arg = -2. * self.gam_yf * hv
                grad = k[:, None] * arg
                return grad

            def Jac(sig):
                # define Jacobian of coordinate transformation
                J = np.ones((3, 3))
                dev = sig_dev(sig)  # deviatoric princ. stress
                vn = np.linalg.norm(dev) * np.sqrt(1.5)  # norm of stress vector
                if vn > 0.1:
                    # calculate Jacobian only if sig>0
                    dseqds = 3. * dev / vn
                    J[:, 2] /= 3.
                    J[:, 0] = dseqds
                    dsa = np.dot(sig, a_vec)
                    dsb = np.dot(sig, b_vec)
                    sc = dsa + 1j * dsb
                    z = -1j * ((a_vec + 1j * b_vec) / sc - dseqds / vn)
                    J[:, 1] = np.real(z)
                return J

            x = np.zeros((N, self.Ndof))
            if self.sdim == 3:
                x[:, 0] = sig_eq_j2(sig) / self.scale_seq - 1.
                x[:, 1] = sig_polar_ang(sig) / np.pi
            else:
                x[:, 0:6] = sig_dev(sig)[:, 0:6] / self.scale_seq
            if self.whdat:
                x[:, self.ind_wh:self.ind_wh + self.sdim] = epl / self.scale_wh
            if self.txdat:
                ih = self.ind_tx
                x[:, ih:ih + self.Nset] = [self.tx_cur[i] / self.scale_text[i] - 1. for i in range(self.Nset)]
            dc = self.svm_yf.dual_coef_[0, :]
            sv = self.svm_yf.support_vectors_
            hk = np.zeros(self.sdim)
            for i in range(N):
                hh = grad_rbf(x[i, :], sv)
                dKdx = np.sum(dc[:, None] * hh, axis=0)
                if self.sdim == 3:
                    fgrad[i, :] = Jac(sig[i, :]) @ np.array([1, dKdx[1], 0])
                else:
                    fgrad[i, 0:6] = dKdx[0:6] / self.scale_seq
                if self.whdat:
                    hk -= dKdx[self.ind_wh:self.ind_wh + self.sdim] * self.scale_seq / self.scale_wh
            self.khard = np.sum(hk) / N  # multiply with matrix (d_eps_eq/d_eps)^-1 instead of summation
            self.msg['gradient'] = 'gradient to ML_yf'
        else:
            # calculate analytical gradient based on the active material formulation 
            # standard: Hill definition of equiv. stress, which contains isotropic J2 equiv. stress
            # as special case.
            # currently implemented: J2, 3-parameter Hill (only principal stresses), 6-parameter Hill full stress tensor
            # no gradient yet for Barlat and Tresca, numerical gradient
            if self.barlat:
                raise ValueError('calc_fgrad: analytical gradient for Barlat not implemented')
            if self.tresca:
                raise ValueError('calc_fgrad: analytical gradient for Tresca not implemented')
            h0 = self.hill[0]
            h1 = self.hill[1]
            h2 = self.hill[2]
            d3 = self.drucker / 3.
            if seq is None:
                seq = self.calc_seq(sig)
            sig = sig_dev(sig)
            fgrad[:, 0] = ((h0 + h2) * sig[:, 0] - h0 * sig[:, 1] - h2 * sig[:, 2]) / (2. * seq) + d3
            fgrad[:, 1] = ((h1 + h0) * sig[:, 1] - h0 * sig[:, 0] - h1 * sig[:, 2]) / (2. * seq) + d3
            fgrad[:, 2] = ((h2 + h1) * sig[:, 2] - h2 * sig[:, 0] - h1 * sig[:, 1]) / (2. * seq) + d3
            if self.sdim == 6:
                h3 = self.hill[3]
                h4 = self.hill[4]
                h5 = self.hill[5]
                fgrad[:, 3] = 3. * h3 * sig[:, 3] / seq
                fgrad[:, 4] = 3. * h4 * sig[:, 4] / seq
                fgrad[:, 5] = 3. * h5 * sig[:, 5] / seq
                if h0 == h1 == h2 == h3 == h4 == h5 == 1.:
                    label = 'analytical, J2 isotropic, full stress'
                else:
                    label = 'analytical, 6-parameter Hill, full stress'
            else:
                if h0 == h1 == h2 == 1.:
                    label = 'analytical, J2 isotropic, princ. stress'
                else:
                    label = 'analytical, 3-parameter Hill, princ. stress'
            self.msg['gradient'] = label
        if N == 1:
            fgrad = fgrad[0, :]
        return fgrad

    def get_sflow(self, epl):
        """Calculate an estimate of the scalar flow stress (strength) of the material
        for a given plastic strain.

        NOTE: Currently assumes only linear isotropic hardening with the current hardening rate,
        does not include texture information and needs to be adapted to data contained in ms.param
        
        Parameters
        ----------
        epl : float or (sdim,) array
            Current value of equiv. plastic strain (float) or plastic strain tensor
            
        Yields
        ------
        sflow : float
            Average flow stress"""

        # if self.msparam is None:
        if type(epl) in (float, np.float64):
            peeq = epl
        else:
            peeq = eps_eq(epl)

        sflow = self.sy + peeq * self.khard
        '''else:
            sm = np.sum(self.tx_cur)
            if sm < 1.e-3:
                wght = np.ones(self.Nset) / self.Nset
            else:
                wght = self.tx_cur / sm
            sflow = 0.
            for i, ms in enumerate(self.msparam):
                sflow += np.interp(peeq + self.epc, ms['work_hard'], ms['sy_av'][self.ms_index[i], :]) * wght[i]'''
        return sflow

    def epl_dot(self, sig, epl, Cel, deps):
        """Calculate plastic strain increment relaxing stress back to yield locus;
        Reference: M.A. Crisfield, Non-linear finite element analysis of solids and structures,
        Chapter 6, Eqs. (6.4), (6.8) and (6.17)

        Parameters
        ----------
        sig : (6,)-array
            Voigt stress tensor
        epl : (6,)-array
            Voigt plastic strain tensor
        Cel : (6,6) array
            Elastic stiffnes tensor
        deps: Voigt tensor
            Strain increment from predictor step

        Returns
        -------
        pdot : Voigt tensor
            Plastic strain increment
        """
        # peeq = eps_eq(epl)  # equiv. plastic strain
        yfun = self.calc_yf(sig + Cel @ deps, epl=epl)
        # for DEBUGGING
        '''yf0  = self.calc_yf(sig, peeq=peeq)
        if yf0<-ptol and yfun>ptol and peeq<1.e-5:
            if self.ML_yf:
                ds = Cel@deps
                yfun = self.ML_full_yf(sig+ds)
            print('*** Warning in epl_dot: Step crosses yield surface')
            print('sig, epl, deps, yfun, yf0, peeq,caller', sig, epl, deps, yfun, yf0, peeq, sys._getframe().f_back.f_code.co_name)
            yfun=0. # catch error that can be produced for anisotropic materials'''
        if (yfun <= yf_tolerance):
            pdot = np.zeros(6)
            # print('WARNING: Test for small stresses will be depracted in next version')
        else:
            if self.sdim == 3:
                a = np.zeros(6)
                a[0:3] = self.calc_fgrad(sig_princ(sig)[0], epl=epl)
            else:
                a = self.calc_fgrad(sig, epl=epl)
            hh = a.T @ Cel @ a + self.khard
            lam_dot = a.T @ Cel @ deps / hh  # deps must not contain elastic strain components
            pdot = lam_dot * a
        return pdot

    def C_tan(self, sig, Cel, epl=None):
        """Calculate tangent stiffness relaxing stress back to yield locus;
        Reference: M.A. Crisfield, Non-linear finite element analysis of solids and structures,
        Chapter 6, Eqs. (6.9) and (6.18)

        Parameters
        ----------
        sig : Voigt tensor
            Stress
        Cel : (6,6) array
            Elastic stiffness tensor used for predictor step
        epl : (sdim,) array
            Equivalent plastic strain tensor (optional, default: 0.)

        Returns
        -------
        Ct : (6,6) array
            Tangent stiffness tensor
        """
        if epl is None:
            epl = np.zeros(self.sdim)
        if self.sdim == 3:
            a = np.zeros(6)
            a[0:3] = self.calc_fgrad(sig_princ(sig)[0], epl=epl)
        else:
            a = self.calc_fgrad(sig, epl=epl)
        hh = a.T @ Cel @ a + self.khard
        ca = Cel @ a
        Ct = Cel - np.kron(ca, ca).reshape(6, 6) / hh
        return Ct

    # ==============================================================
    # subroutines for ML flow rule, training
    # ==============================================================
    def setup_yf_SVM(self, x, y_train, x_test=None, y_test=None, C=15., gamma=2.5,
                     fs=0.1, plot=False, cyl=False, gridsearch=False, cvals=None, gvals=None):
        """
        Generic function call to setup and train the SVM yield function, for details see the specific functions
        setup_yf_SVM_6D and setup_yf_SVM_3D.
        """
        if self.sdim == 3:
            train_sc, test_sc = self.setup_yf_SVM_3D(x, y_train, x_test=x_test, y_test=y_test,
                                                     C=C, gamma=gamma, fs=fs, plot=plot, cyl=cyl,
                                                     gridsearch=gridsearch, cvals=cvals, gvals=gvals)
        else:
            train_sc, test_sc = self.setup_yf_SVM_6D(x, y_train, x_test=x_test, y_test=y_test,
                                                     C=C, gamma=gamma, plot=plot,
                                                     gridsearch=gridsearch, cvals=cvals, gvals=gvals)
        return train_sc, test_sc

    def setup_yf_SVM_6D(self, x, y_train, x_test=None, y_test=None, C=15., gamma=2.5, plot=False,
                        gridsearch=False, cvals=None, gvals=None):
        """Initialize and train Support Vector Classifier (SVC) as machine learning (ML) yield function. Training and 
        test data (features) are accepted as either 3D principal stresses or cylindrical stresses, but principal 
        stresses will be converted to cylindrical stresses, such that training is always performed in cylindrical 
        stress space, with equiv. stress at yield onset and polar angle as degrees of freedom. Graphical output on 
        the trained SVC yield function is possible. 


        Parameters
        ----------
        x   :  (N,self.Ndof) array
            Training data in form of deviatoric Voigt stresses, components s1-s6), s0=-s1-s2.
            Additional DOF for work hardening and texture if considered.
        y_train : 1d-array
            Result vector for training data (same size as x)
        x_test  : (N,self.Ndof) array
            Test data either as Cartesian princ. stresses (N,3) or cylindrical stresses (N,2) (optional)
        y_test
            Result vector for test data (optional)
        C  : float
            Parameter for training of SVC (optional, default: 10)
        gamma  : float
            Parameter for kernel function of SVC (optional, default: 1)
        plot : Boolean
            Indicates if plot of decision function should be generated (optional, default: False)
        gridsearch : Boolean
            Perform grid search to optimize hyper parameters of ML flow rule (optional, default: False)
        cvals : array
            Values for SVC training parameter C in gridsearch (optional, default: None)
        gvals: array
            Values for SVC parameter gamma in gridsearch (optional, default: None)

        Returns
        -------
        train_sc : float
            Training score
        test_sc  : float
            test score
        """
        # calculate proper scaling factor and scale data in input vector into range [-1,+1] for all columns
        print('Using {} full Voigt yield stresses for training.'.format(x.shape))
        assert self.sdim == 6
        self.gam_yf = gamma
        self.C_yf = C
        if self.msparam is None:
            self.scale_seq = self.sy
        else:
            # calculate scaling factors need for SVC training from microstructure parameters
            self.scale_seq = 0.
            self.scale_wh = 0.
            self.scale_text = np.zeros(self.Nset)
            for i in range(self.Nset):
                self.scale_seq += self.msparam[i]['sy_av'] / self.Nset
                self.scale_wh += (self.msparam[i]['peeq_max'] - self.epc) / self.Nset
                self.scale_text[i] = np.average(self.msparam[i]['texture'])
        N = len(x)
        X_train = np.zeros((N, self.Ndof))
        X_train[:, 0:6] = x[:, 0:6] / self.scale_seq
        if self.whdat:
            X_train[:, self.ind_wh:self.ind_wh + 6] = x[:, self.ind_wh:self.ind_wh + 6] / self.scale_wh
            print('Using work hardening data "%s" for training: %i data sets up to PEEQ=%6.3f'
                  % (self.msparam[0]['ms_type'], self.msparam[0]['Npl'], self.msparam[0]['peeq_max']))
        if self.txdat:
            ih = self.ind_tx
            for i in range(self.Nset):
                X_train[:, ih + i] = x[:, ih + i] / self.scale_text[i] - 1.
                print(
                    'Using texture data "%s" for training: %i data sets with texture_parameters in range [%4.2f,%4.2f]'
                    % (self.msparam[i]['ms_type'], self.msparam[i]['Ntext'], self.msparam[i]['texture'][0],
                       self.msparam[i]['texture'][-1]))

        # coordinate transformation for test data
        if x_test is not None:
            Ntest = len(x_test)
            X_test = np.zeros((Ntest, self.Ndof))
            X_test[:, 0:6] = x_test[:, 0:6] / self.scale_seq
            if self.whdat:
                X_test[:, self.ind_wh:self.ind_wh + 6] = x_test[:, self.ind_wh:self.ind_wh + 6] / self.scale_wh
            if self.txdat:
                ih = self.ind_tx
                for i in range(self.Nset):
                    X_test[:, ih + i] = x_test[:, ih + i] / self.scale_text[i] - 1.
        # define and fit SVC
        if gridsearch:
            print('The hyperparameter optimization with Gridsearch to find best C and gamma...')
            # define search grid and add user parameters if not present in grid
            if cvals is None:
                cvals = [4, 6, 8, 10, 15, 20]
                if C not in cvals:
                    cvals.append(C)
            if gvals is None:
                gvals = [1, 1.5, 2, 2.5, 3]
                if gamma not in gvals:
                    gvals.append(gamma)
            param_grid = {'C': cvals, 'gamma': gvals}
            grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            print('The best hyperparameters are:', grid.best_params_)
            self.gam_yf = grid.best_params_["gamma"]
            self.C_yf = grid.best_params_["C"]
            self.svm_yf = svm.SVC(kernel='rbf', C=self.C_yf, gamma=self.gam_yf)
            self.svm_yf.fit(X_train, y_train)
            print('Original values: C={}, gamma={}'.format(C, gamma))
        else:
            self.svm_yf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
            self.svm_yf.fit(X_train, y_train)
        self.ML_yf = True
        # calculate scores
        train_sc = 100 * self.svm_yf.score(X_train, y_train)
        if x_test is None:
            test_sc = None
        else:
            test_sc = 100 * self.svm_yf.score(x_test, y_test)
        # create plot if requested
        if plot:
            print('Plot of extended training data for SVM classification in 2D cylindrical stress space')
            xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 50), np.linspace(-1.2, 1.2, 50))
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
            if self.Ndof == 2:
                feat = np.c_[yy.ravel(), xx.ravel()]
            elif self.Ndof == 3:
                feat = np.c_[yy.ravel(), xx.ravel(), np.ones(2500) * self.scale_wh]
            else:
                feat = np.c_[yy.ravel(), xx.ravel(), np.ones(2500) * self.scale_wh, np.ones(2500) * self.scale_text]
            Z = self.svm_yf.decision_function(feat)
            self.plot_data(Z, ax, xx, yy, c='black')
            ax.scatter(X_train[:, 1], X_train[:, 0], s=10, c=y_train, cmap=plt.cm.Paired)
            ax.set_title('extended SVM yield function in training')
            ax.set_xlabel(r'$\theta/\pi$')
            ax.set_ylabel(r'$\sigma_{eq}/\sigma_y$')
            plt.show()
        return train_sc, test_sc

    def setup_yf_SVM_3D(self, x, y_train, x_test=None, y_test=None, C=10.,
                        gamma=2., fs=0.1, plot=False, cyl=False,
                        gridsearch=False, cvals=None, gvals=None):
        """Initialize and train Support Vector Classifier (SVC) as machine
        learning (ML) yield function. Training and test data (features) are
        accepted as either 3D principal stresses or cylindrical stresses, but
        principal stresses will be converted to cylindrical stresses, such
        that training is always performed in cylindrical stress space, with
        equiv. stress at yield onset and polar angle as degrees of freedom. 
        Graphical output on the trained SVC yield function is possible.


        Parameters
        ----------
        x   :  (N,2) or (N,3) array
            Training data either as Cartesian princ. stresses (N,3) or
            cylindrical stresses (N,2)
        cyl : Boolean
            Indicator for cylindrical stresses if x is has shape (N,3)
        y_train : 1d-array
            Result vector for training data (same size as x)
        x_test  : (N,2) or (N,3) array
            Test data either as Cartesian princ. stresses (N,3) or cylindrical
            stresses (N,2) (optional)
        y_test
            Result vector for test data (optional)
        C  : float
            Parameter for training of SVC (optional, default: 10)
        gamma  : float
            Parameter for kernel function of SVC (optional, default: 1)
        fs  : float
            Parameters for size of periodic continuation of training data
            (optional, default:0.1)
        plot : Boolean
            Indicates if plot of decision function should be generated
            (optional, default: False)
        gridsearch : Boolean
            Perform grid search to optimize hyperparameters of ML flow rule
            (optional, default: False)
        cvals : array
            Values for SVC training parameter C in gridsearch (optional, default: None)
        gvals: array
            Values for SVC parameter gamma in gridsearch (optional, default: None)

        Returns
        -------
        train_sc : float
            Training score
        test_sc  : float
            test score
        """
        # transformation of princ. stress into cyl. coordinates
        self.gam_yf = gamma
        self.C_yf = C
        assert self.sdim == 3
        if self.msparam is None:
            self.scale_seq = self.sy
        else:
            # calculate scaling factors need for SVC training from microstructure parameters
            self.scale_seq = 0.
            self.scale_wh = 0.
            self.scale_text = np.zeros(self.Nset)
            for i in range(self.Nset):
                self.scale_seq += self.msparam[i]['sy_av'] / self.Nset
                self.scale_wh += (self.msparam[i]['peeq_max'] - self.epc) / self.Nset
                self.scale_text[i] = np.average(self.msparam[i]['texture'])
        N = len(x)
        X_train = np.zeros((N, self.Ndof))
        if not cyl:
            # princ. stresses
            X_train[:, 0] = sig_eq_j2(x[:, 0:3]) / self.scale_seq - 1.
            X_train[:, 1] = sig_polar_ang(x[:, 0:3]) / np.pi
            print('Converting principal stresses to cylindrical stresses for training')
        else:
            # cylindrical stresses
            X_train[:, 0] = x[:, 0] / self.scale_seq - 1.
            X_train[:, 1] = x[:, 1] / np.pi
            print('Using cylindrical stresses for training')
        if self.whdat:
            X_train[:, self.ind_wh] = x[:, self.ind_wh + 1] / self.scale_wh - 1.
            print('Using work hardening data "%s" for training up to PEEQ=%6.3f'
                  % (self.msparam[0]['ms_type'], self.msparam[0]['peeq_max']))
        if self.txdat:
            ih = self.ind_tx + 1
            for i in range(self.Nset):
                X_train[:, ih + i] = x[:, ih + i] / self.scale_text[i] - 1.
                print(
                    'Using texture data "%s" for training: %i data sets with texture_parameters in range [%4.2f,%4.2f]'
                    % (self.msparam[i]['ms_type'], self.msparam[i]['Ntext'], self.msparam[i]['texture'][0], \
                       self.msparam[i]['texture'][-1]))
        # copy left and right borders to enforce periodicity in theta
        indr = np.nonzero(X_train[:, 1] > 1. - fs)
        indl = np.nonzero(X_train[:, 1] < fs - 1.)
        Xr = X_train[indr]
        Xl = X_train[indl]
        Xr[:, 1] -= 2.  # shift angle theta
        Xl[:, 1] += 2.
        Xh = np.append(Xr, Xl, axis=0)
        yh = np.append(y_train[indr], y_train[indl], axis=0)
        X_train = np.append(X_train, Xh, axis=0)
        y_train = np.append(y_train, yh, axis=0)
        # coordinate transformation for test data
        if x_test is not None:
            Ntest = len(x_test)
            X_test = np.zeros((Ntest, self.Ndof))
            if not cyl:
                X_test[:, 0] = sig_eq_j2(x_test) / self.scale_seq - 1.
                X_test[:, 1] = sig_polar_ang(x_test) / np.pi
            else:
                X_test[:, 0] = x_test[:, 0] / self.scale_seq - 1.
                X_test[:, 1] = x_test[:, 1] / np.pi
            if self.whdat:
                X_test[:, self.ind_wh] = x_test[:, self.ind_wh + 1] / self.scale_wh - 1.
            if self.txdat:
                ih = self.ind_tx + 1
                for i in range(self.Nset):
                    X_test[:, ih + i] = x_test[:, ih + i] / self.scale_text[i] - 1.
        # define and fit SVC
        if gridsearch:
            print('The hyperparameter optimization with Gridsearch to find best C and gamma...')
            # define search grid and add user parameters if not present in grid
            if cvals is None:
                cvals = [4, 6, 8, 10, 15, 20]
                if not C in cvals:
                    cvals.append(C)
            if gvals is None:
                gvals = [1, 1.5, 2, 2.5, 3]
                if not gamma in gvals:
                    gvals.append(gamma)
            param_grid = {'C': cvals, 'gamma': gvals}
            grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            print('The best hyperparameters are:', grid.best_params_)
            self.gam_yf = grid.best_params_["gamma"]
            self.C_yf = grid.best_params_["C"]
            self.svm_yf = svm.SVC(kernel='rbf', C=self.C_yf, gamma=self.gam_yf)
            self.svm_yf.fit(X_train, y_train)
            print('Original values: C={}, gamma={}'.format(C, gamma))
        else:
            self.svm_yf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
            self.svm_yf.fit(X_train, y_train)
        self.ML_yf = True

        # calculate scores
        train_sc = 100 * self.svm_yf.score(X_train, y_train)
        if x_test is None:
            test_sc = None
        else:
            test_sc = 100 * self.svm_yf.score(X_test, y_test)
        # create plot if requested
        if plot:
            print('Plot of extended training data for SVM classification in 2D cylindrical stress space')
            xx, yy = np.meshgrid(np.linspace(-1. - fs, 1. + fs, 50), np.linspace(-1., 1., 50))
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
            if self.Ndof == 2:
                feat = np.c_[yy.ravel(), xx.ravel()]
            elif self.Ndof == 3:
                feat = np.c_[yy.ravel(), xx.ravel(), np.ones(2500) * self.scale_wh]
            else:
                feat = np.c_[yy.ravel(), xx.ravel(), np.ones(2500) * self.scale_wh, np.ones(2500) * self.scale_text]
            Z = self.svm_yf.decision_function(feat)
            self.plot_data(Z, ax, xx, yy, c='black')
            ax.scatter(X_train[:, 1], X_train[:, 0], s=10, c=y_train, cmap=plt.cm.Paired)
            ax.set_title('extended SVM yield function in training')
            ax.set_xlabel(r'$\theta/\pi$')
            ax.set_ylabel(r'$\sigma_{eq}/\sigma_y$')
            plt.show()
        return train_sc, test_sc

    def train_SVC(self, C=10, gamma=4, Nlc=36, Nseq=25, fs=0.3, extend=True,
                  mat_ref=None, sdata=None, plot=False, fontsize=16,
                  gridsearch=False, cvals=None, gvals=None):
        """Train SVC for all yield functions of the microstructures provided
        in msparam and for flow stresses to capture work hardening. In first
        step, the training data for each set is generated by creating stresses
        on the deviatoric plane and calculating their catgegorial
        yield function ("-1": elastic, "+1": plastic). Furthermore, axes in
        different dimensions for microstructural features are introduced that
        describe the relation between the different sets.
        
        Parameters
        ----------
        C     : float
            Parameter needed for training process, larger values lead to more
            flexibility (optional, default: 10)
        gamma : float
            Parameter of Radial Basis Function of SVC kernel, larger values,
            lead to faster decay of influence of individual 
            support vectors, i.e., to more short ranged kernels
            (optional, default: 4)
        Nlc   : int 
            Number of load cases to be considered, will be overwritten if
            material has microstructure 
            information (optional, default: 36)
        Nseq  : int
            Number of training and test stresses to be generated in elastic
            regime, same number will be 
            produced in plastic regime (optional, default: 25)
        fs : float
            Parameter to ensure peridicity of yield function wrt. theta
        extend : Boolean
            Indicate whether training data should be extended further into
            plastic regime (optional, default: True)
        mat_ref : object of class ``Material``
            reference material needed to calculate yield function if only N is
            provided (optional, ignored if sdata is given)
        sdata: (N, sdim) array
            List of Cartsian stresses lying on yield locus. Based on these
            yield stresses, training data in entire deviatoric stress space
            is created (optional, f no data in self.msparam is given, either
                        sdata or N and mat_ref must be provided)
        plot  : Boolean
            Indicate if graphical output should be performed
            (optional, default: False)
        fontsize : int
            Fontsize for graph annotations (optional, default: 16)
        gridsearch : Boolean
            Perform grid search to optimize hyperparameters of ML flow rule
            (optional, default: False)
        """
        print('\n---------------------------\n')
        print('SVM classification training')
        print('---------------------------\n')
        # augment raw data and create result vector (yield function) for all
        # data on work hardening and textures
        if self.msparam is None:
            Npl = 1
            Ntext = 1
            if sdata is None:
                # create regular pattern of stresses in sdim-dimensional stress
                # space based on reference material
                if mat_ref is None:
                    raise ValueError(
                        'create_data_sig: Neither sdata nor mat_ref are provided, cannot generate training data')
                # define material parameters otherwise defines in material.plasticity
                if mat_ref.CV is None:
                    self.elasticity(C11=mat_ref.C11, C12=mat_ref.C12, C44=mat_ref.C44)
                else:
                    self.elasticity(CV=mat_ref.CV)
                self.plasticity(sy=mat_ref.sy, sdim=mat_ref.sdim)
                xt, yt = self.create_sig_data(N=Nlc, mat_ref=mat_ref, Nseq=Nseq, extend=extend)
                print('Training data created from reference material', mat_ref.name, ', with', Nlc, 'load cases.')
            else:
                # based on given yield stresses
                Nlc = len(sdata[:, 0])
                seq = sig_eq_j2(sdata)
                self.plasticity(sy=np.mean(seq), sdim=len(sdata[0, :]))
                xt, yt = self.create_sig_data(sdata=sdata, Nseq=Nseq, extend=extend)
                print('Training data created from {}-dimensional yield stresses with {} load cases.' \
                      .format(self.sdim, Nlc))
            self.Ndof = 2 if self.sdim == 3 else 6
        else:
            '''WARNING: There are no more hardening levels, Npl, epc in undefined !!!'''
            Nlc = self.msparam[0]['Nlc']
            Npl = self.msparam[0]['Npl']
            Ntext = self.msparam[0]['Ntext']
            if extend:
                Ne = 4
            else:
                Ne = 0
            N0 = Nlc * (2 * Nseq + Ne)  # total number of training data points per level of PEEQ for each microstructure
            if self.txdat:
                Nt = self.Nset * Ntext * Npl * N0  # total number of training data points
            else:
                Nt = Ntext * Npl * N0
            if self.sdim == 3:
                dtrain = 3  # dimension of training data (Ndof+1 for sdim==3)
                if self.whdat:
                    iwh = dtrain
                    dtrain += 1
                if self.txdat:
                    itx = dtrain
                    dtrain += self.Nset
            else:
                dtrain = self.Ndof  # dimension of training data (Ndof for sdim==6)
                if self.whdat:
                    iwh = self.ind_wh
                if self.txdat:
                    itx = self.ind_tx
            xt = np.zeros((Nt, dtrain))
            yt = np.zeros(Nt)
            for m, ms in enumerate(self.msparam):
                for k in range(Ntext):  # loop over all textures
                    for j in range(Npl):  # loop over work hardening levels for each texture
                        # create training data in entire deviatoric stress plane from raw data
                        # training data generated here is unscaled
                        # work hardening parameters are corrected for offset
                        sig_train, yf_train = self.create_sig_data(sdata=ms['flow_stress'][k, j, :, :],
                                                                   sflow=ms['flow_seq_av'][k, j], Nseq=Nseq,
                                                                   extend=True)
                        i0 = (j + k * Npl + m * Ntext) * N0
                        i1 = i0 + N0
                        xt[i0:i1, 0:self.sdim] = sig_train
                        if self.whdat:
                            # Add DOF for work hardening parameter, corrected for offset
                            xt[i0:i1, iwh] = ms['work_hard'][j] - self.epc
                        if self.txdat:
                            # Add DOF for textures
                            xt[i0:i1, itx + m] = ms['texture'][k]
                        yt[i0:i1] = yf_train
            print(
                '%i training data sets created from %i microstructures, with %i load cases each' % (Nt, self.Nset, Nlc))

        if np.any(np.abs(yt) <= 0.99):
            warnings.warn(
                'train_SVC: result vector for yield function contains more categories than "-1" and "+1". Will result in higher dimensional SVC.')
        # Train SVC with data from all microstructures in data
        if self.sdim == 3:
            # assuming Cartesian stresses
            train_sc, test_sc = \
                self.setup_yf_SVM_3D(xt, yt, C=C, gamma=gamma, fs=0.3,
                                     plot=False, gridsearch=gridsearch,
                                     cvals=cvals, gvals=gvals)
        else:
            train_sc, test_sc = self.setup_yf_SVM_6D(xt, yt, C=C, gamma=gamma,
                                                     gridsearch=gridsearch)

        print(self.svm_yf)
        print("Training set score: {} %".format(train_sc))

        if plot:
            '''WARNING: untested for 6D structure of msparam !!!'''
            # plot ML yield loci with reference and test data
            print('Plot ML yield loci with reference curve and test data')
            if self.whdat:
                print('Initial yield locus plotted together with flow stresses for PEEQ in range [%6.3f,%6.3f]'
                      % (self.msparam[0]['work_hard'][0], self.msparam[0]['work_hard'][-1]))
            if self.txdat:
                print('Initial yield locus plotted for texture parameter in range [%6.3f,%6.3f]'
                      % (self.msparam[0]['texture'][0], self.msparam[0]['texture'][-1]))
                Npl = 1  # only plot initial yield surface

            ncol = 2
            Npl = 4
            nrow = int(Npl * Ntext / ncol + 0.95)
            plt.figure(figsize=(20, 8 * nrow))
            plt.subplots_adjust(hspace=0.3)
            theta = np.linspace(-np.pi, np.pi, 36)
            work_hard = np.linspace(self.epc, self.msparam[0]['peeq_max'], Npl)
            for k in range(Ntext):
                self.set_texture(self.msparam[0]['texture'][k], verb=False)
                for j in range(0, Npl, np.maximum(1, int(Npl / 4))):
                    # to-do: x_test and y_test should be setup properly above!!!
                    ind = list(range((j + k * Npl) * N0, (j + k * Npl + 1) * N0, int(0.5 * N0 / Nlc)))
                    x_test = sig_princ2cyl(xt[ind, 0:self.sdim])
                    y_test = yt[ind]
                    ind = np.argsort(x_test[:, 1])  # sort dta points w.r.t. polar angle
                    x_test = x_test[ind, :]
                    y_test = y_test[ind]
                    peeq = work_hard[j] - self.epc
                    sflow = self.get_sflow(peeq)
                    iel = np.nonzero(y_test < 0.)[0]
                    ipl = np.nonzero(np.logical_and(y_test >= 0., x_test[:, 0] < sflow * 1.5))[0]
                    plt.subplot(nrow, ncol, j + k * Npl + 1, projection='polar')
                    plt.gca().plot(x_test[ipl, 1], x_test[ipl, 0], 'r.', label='test data above yield point')
                    plt.gca().plot(x_test[iel, 1], x_test[iel, 0], 'b.', label='test data below yield point')
                    if self.msparam is not None:
                        syc = sig_princ2cyl(self.msparam[0]['flow_stress'][k, j, :, :])
                        ind = np.argsort(syc[:, 1])
                        plt.gca().plot(syc[ind, 1], syc[ind, 0], '-c',
                                       label='reference yield locus')
                    # ML yield fct: find norm of princ. stess vector lying on yield surface
                    snorm = sig_cyl2princ(np.array([sflow * np.ones(36) * np.sqrt(1.5), theta]).T)
                    x1 = fsolve(self.find_yloc, np.ones(36), args=(snorm, peeq), xtol=1.e-5)
                    sig = snorm * x1[:, None]
                    s_yld = sig_eq_j2(sig)
                    plt.gca().plot(theta, s_yld, '-k', label='ML yield locus', linewidth=2)
                    if self.msparam is None:
                        plt.title = self.name
                    else:
                        plt.title(
                            'Flow stress, PEEQ=' + str(work_hard[j].round(decimals=4)) + ', TP='
                            + str(self.msparam[0]['texture'][k].round(decimals=2)), fontsize=fontsize)
                    # plt.xlabel(r'$\theta$ (rad)', fontsize=fontsize-2)
                    # plt.ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=fontsize-2)
                    plt.legend(loc=(.95, 0.85), fontsize=fontsize - 2)
                    plt.tick_params(axis="x", labelsize=fontsize - 4)
                    plt.tick_params(axis="y", labelsize=fontsize - 4)
            plt.show()

    def create_sig_data(self, N=None, mat_ref=None, sdata=None, Nseq=12, sflow=None,
                        offs=0.01, extend=False, rand=False):
        """Function to create consistent data sets on the deviatoric stress plane
        for training or testing of ML yield function. Either the number "N" of raw data points, i.e. load angles, to be 
        generated and a reference material "mat_ref" has to be provided, or a list of raw data points "sdata" with 
        Cartesian stress tensors lying on the yield surface serves as input. Based on the raw data, stress tensors 
        from the yield locus are distributed into the entire deviatoric space, by linear downscaling into the elastic 
        region and upscaling into the plastic region. Data is created in form of cylindrical stresses that lie densly 
        around the expected yield locus and more sparsely in the outer plastic region.

        Parameters
        ----------
        N    : int
            Number of load cases (polar angles) to be created (optional,
            either N and mat_ref or sdata must be provided)
        mat_ref : object of class ``Material``
            reference material needed to calculate yield function if only N is provided (optional, ignored if sdata is given)
        sdata: (N, sdim) array
            List of Cartesian stress tensors lying on yield locus. 
            Based on these yield stresses, training data in entire deviatoric stress space is created
            (optional, either sdata or N and mat_ref must be provided)
        Nseq : int
            Number of training stresses to be generated in the range 'offs' to the yield strength (optional, default: 12)
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
        st : (M, sdim) array
            Cartesian training stresses, M = N (2 Nseq + Nextend)
        yt : (M,) array
            Result vector of categorial yield function (-1 or +1) for supervised training
        """
        if sflow is None:
            sflow = self.sy
        if sdata is None:
            if mat_ref is None:
                raise ValueError(
                    'create_data_sig: Neither sdata nor mat_ref are provided, cannot generate training data')
            # create regular pattern of stresses in sdim-dimensional stress space
            if self.sdim == 3:
                if N is None:
                    warnings.warn('create_sig_data: Neither N not theta provided. Continuing with N=36 (sdim=3)')
                    N = 36
                if not rand:
                    theta = np.linspace(-np.pi, np.pi, N)
                else:
                    theta = 2. * (np.random.rand(N) - 0.5) * np.pi
                sc = np.ones((N, 2))
                sc[:, 1] = theta
                su = sig_cyl2princ(sc)
            else:
                if N is None:
                    warnings.warn('create_sig_data: Neither N not theta provided. Continuing with N=300 (sdim=6)')
                    N = 300
                n3 = int(N / 3)
                n6 = N - n3
                su = sig_dev(load_cases(n3, n6))
            x1 = fsolve(mat_ref.find_yloc, np.ones(N) * mat_ref.sy, args=(su), xtol=1.e-5)
            sdata = sig_dev(su * x1[:, None])  # yield stress tensors representing ground truth
        else:
            # read stress data as seeding points for generation of further training stresses in entire 
            # sdim-dimensional stress space
            i = len(sdata)
            if (N is not None) and (N != i):
                warnings.warn(f'create_sig_data: N and dimension of sdata do not agree. Continuing with N ={i}')
            if mat_ref is not None:
                warnings.warn('create_sig_data: using sdata for training, ignoring mat_ref')
            N = i
            sdata = sig_dev(sdata)  # make sure training stresses are purely deviatoric
        seq = np.linspace(offs, 0.95, Nseq)
        seq = np.append(seq, np.linspace(1.05, 2., Nseq))
        if extend:
            # add training points in plastic regime to avoid fallback of SVC decision fct. to zero
            seq = np.append(seq, np.array([2.4, 3., 4., 5.]))
        Nd = len(seq)  # number of training stresses per load case
        st = np.zeros((N * Nd, self.sdim))  # input vector with training stresses
        yt = np.zeros(N * Nd)  # result vector for supervised learning
        for i in range(Nd):
            j0 = i * N
            j1 = (i + 1) * N
            # scale sdata into elastic regime (seq<1) or into plastic regime (seq>=1)
            st[j0:j1, :] = sdata[:, 0:self.sdim] * seq[i]
            yt[j0:j1] = -1. if i < Nseq else +1.
        return st, yt

    def setup_fgrad_SVM(self, X_grad_train, y_grad_train, C=10., gamma=0.1):
        """Inititalize and train SVM regression for gradient evaluation

        Parameters
        ----------
        X_grad_train : (N,3) array
        y_grad_train : (N,) array
        C     : float
            Paramater for training of Support Vector Regression (SVR) (optional, default:10)
        gamma : float
            Parameter for kernel of SVR (optional, default: 0.1)
        """
        # define support vector regressor parameters
        self.svm_grad0 = svm.SVR(C=C, cache_size=3000, coef0=0.0, degree=3, epsilon=0.01, gamma=gamma,
                                 kernel='rbf', max_iter=-1, shrinking=True, tol=0.0001, verbose=False)
        self.svm_grad1 = svm.SVR(C=C, cache_size=3000, coef0=0.0, degree=3, epsilon=0.01, gamma=gamma,
                                 kernel='rbf', max_iter=-1, shrinking=True, tol=0.0001, verbose=False)
        self.svm_grad2 = svm.SVR(C=C, cache_size=3000, coef0=0.0, degree=3, epsilon=0.01, gamma=gamma,
                                 kernel='rbf', max_iter=-1, shrinking=True, tol=0.0001, verbose=False)

        # fit SVM to training data
        X_grad_train = X_grad_train / self.sy
        gmax = np.amax(y_grad_train, axis=0)
        gmin = np.amin(y_grad_train, axis=0)
        self.gscale = gmax - gmin
        y_grad_train0 = y_grad_train[:, 0] / self.gscale[0]
        y_grad_train1 = y_grad_train[:, 1] / self.gscale[1]
        y_grad_train2 = y_grad_train[:, 2] / self.gscale[2]
        self.svm_grad0.fit(X_grad_train, y_grad_train0)
        self.svm_grad1.fit(X_grad_train, y_grad_train1)
        self.svm_grad2.fit(X_grad_train, y_grad_train2)
        self.ML_grad = True

    def export_MLparam(self, sname, source=None, file=None, path='../../models/',
                       descr=None, param=None):
        """The parameters of the trained Ml flow rule (support vectors, dual 
        coefficients, offset and scaling parameters) are written to a csv file
        that is readable to Abaqus (8 numbers per line).
        
        Parameters
        ----------
        sname : str
            Name of script that created this material
        source : str
            Source of parameters (optional, default: None)
        file : str
            Trunk of filename to which CSV flies are written (optional, default: None)
        path : str
            Path to which files are written (optional: default: '../../models/')
        descr : list
            List of names of model parameters used for generating this ML material (optional, default: [])
        param : list
            List of values of parameters used for generating this ML material (optional, default: []);
            descr and param must be of the same size
            
            
        Yields
        ------
        CSV file with name path+file+'-svm.csv' containing 
        support vectors, dual coefficients, and (Ndof, elastic parameters, offset, 
        gamma value, scaling factors) in Abaqus format; and
        JSON file with name path+file+'-svm_meta.json' containing meta data in given format.
        """
        from pkg_resources import get_distribution
        from json import dump
        from datetime import date

        if not self.ML_yf:
            raise AttributeError('export_MLparam: No ML flow rule defined.')
        if self.msparam is None:
            self.Nset = 1
            self.epc = 0.
            self.scale_wh = 1.
            self.scale_text = [1.]
        if self.Nset > 9:
            raise ValueError('export_MLparam: Too many sets to export.')
        if (descr is not None and param is not None) and len(descr) != len(param):
            raise ValueError('Lists for descr and param must have the same lengths.')
        if file is None:
            file = 'abq_' + self.name
        if path[-1] != '/':
            path += '/'
        file = path + file

        # write parameters of trained SVC to file readable to Abaqus
        dc = self.svm_yf.dual_coef_[0]  # dual coefficients
        nsv = len(dc)  # number of support vectors
        nlin = int((nsv * (self.Ndof + 1) + 30) / 8) + 1
        Ndata = nlin * 8  # Number of data points to write
        props = np.zeros(Ndata)
        props[0] = nsv
        props[1] = self.Ndof
        props[2] = self.C11
        props[3] = self.C12
        props[4] = self.C44
        props[5] = self.svm_yf.intercept_[0]
        props[6] = self.gam_yf
        props[7] = self.epc
        props[8] = self.scale_seq
        props[9] = self.scale_wh
        if self.CV is None:
            props[10:16] = -1
        else:
            props[10] = self.CV[1, 1]
            props[11] = self.CV[2, 2]
            props[12] = self.CV[0, 2]
            props[13] = self.CV[1, 2]
            props[14] = self.CV[4, 4]
            props[15] = self.CV[5, 5]
        props[16] = self.Nset
        props[16:16 + self.Nset] = self.scale_text
        props[29:29 + nsv] = dc
        nl = (self.Ndof + 1) * nsv + 29  # last entry of support vectors
        props[29 + nsv:nl] = self.svm_yf.support_vectors_.flatten()
        np.savetxt(file + '-svm.csv', props.reshape((nlin, 8)), delimiter=', ', newline='\n')

        # parameters for metadata
        today = str(date.today())  # date
        owner = getpass.getuser()  # username
        sys_info = platform.uname()  # system information
        if descr is None:
            descr = []
        if param is None:
            param = []
        descr.extend(['Ndata', 'gamma', 'C'])
        param.extend([Ndata, self.gam_yf, self.C_yf])

        # Create metadata
        meta = {
            "Info": {
                "Owner": owner,
                "Institution": "ICAMS, Ruhr University Bochum, Germany",
                "Date": today,
                "Description": "SVC-parameters for plasticity model",
                "Method": "Support Vector Classification",
                "System": {
                    "sysname": sys_info[0],
                    "nodename": sys_info[1],
                    "release": sys_info[2],
                    "version": sys_info[3],
                    "machine": sys_info[4]},
            },
            "Model": {
                "Creator": "pylabfea",
                "Version": get_distribution('pylabfea').version,
                "Repository": "https://github.com/AHartmaier/pyLabFEA.git",
                "Input": source,
                "Script": sname,
                "Names": descr,
                "Parameters": param
            },
            "Data": {
                "Class": 'SVC_parameters',
                "Type": 'CSV',
                "File": file + '-svm.csv',
                "Separator": ',',
                "Header": None,
                "Format": (nlin, 8),
                "Names": ['nsv', 'nsd', 'C11', 'C12', 'C44', 'rho', 'gamma', 'epc',
                          'scale_seq', 'scale_wh', 'C22', 'C33', 'C13', 'C23',
                          'C55', 'C66', 'Nset', 'scale_text[0:Nset]',
                          'dual_coef[0:nsv]', 'sup_vec[0:nsv,0:nsd]'],
                "Units": {
                    'Stress': 'MPa',
                    'Strain': 'None',
                    'Disp': 'mm',
                    'Force': 'N'}
            }
        }
        with open(file + '-svm_meta.json', 'w') as fp:
            dump(meta, fp, indent=2)

    def pckl(self, name=None, path='../../materials/'):
        """Write material into pickle file. Usefull for materials with trained machine 
        learning flow rules to avoid time-consuming re-training.
        
        
        Parameters
        ----------
        name : string (optional, default: None)
            File name for pickled material. The default is None, in which case 
            the filename will be the material name + '.pckl'. 
        path : string
            Path to location for pickles

        Returns
        -------
        None.

        """
        if name is None:
            name = 'mat_' + self.name + '.pkl'
        if path[-1] != '/':
            path += '/'
        with open(path + name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return

    # =========================================================
    # subroutines for material definitions
    # =========================================================
    def elasticity(self, C11=None, C12=None, C44=None,  # standard parameters for crystals with cubic symmetry
                   CV=None,  # user specified Voigt matrix
                   E=None, nu=None):  # parameters for isotropic material
        """Define elastic material properties

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
            
        Returns
        -------
        None.
        """
        if E is not None:
            if nu is None:
                raise ValueError('Error: Inconsistent definition of material parameters: Only E provided')
            if (C11 is not None) or (C12 is not None) or (C44 is not None):
                raise ValueError('Error: Inconsistent definition of material parameters: E provided together with C_ij')
            hh = E / ((1. + nu) * (1. - 2. * nu))
            self.C11 = (1. - nu) * hh
            self.C12 = nu * hh
            self.C44 = (0.5 - nu) * hh
            self.E = E
            self.nu = nu
            self.CV = None
        elif C11 is not None:
            if nu is not None:
                raise ValueError(
                    'Error: Inconsistent definition of material parameters: nu provided together with C_ij')
            if (C12 is None) or (C44 is None):
                raise ValueError('Error: Inconsistent definition of material parameters: C_12 or C_44 values missing')
            self.C11 = C11
            self.C12 = C12
            self.C44 = C44
            self.nu = C12 / (C11 + C12)
            self.E = 2 * C44 * (1 + self.nu)  # only for isotropy, might be used for plane stress models
            self.CV = None
            # warnings.warn('elasticity: E and nu calculated from anisotropic elastic parameters')
        elif CV is not None:
            self.CV = np.array(CV)
            self.C11 = self.CV[0, 0]
            self.C12 = self.CV[0, 1]
            self.C44 = self.CV[3, 3]
            self.nu = self.C12 / (self.C11 + self.C12)
            self.E = 2 * self.C44 * (1 + self.nu)  # only for isotropy, might be used for plane stress models
            # warnings.warn('elasticity: E and nu calculated from anisotropic elastic parameters')
        else:
            raise ValueError('elasticity: Inconsistent definition of material parameters')

    def plasticity(self, sy=None, sdim=6, drucker=0., khard=0.,
                   tresca=False,
                   barlat=None, barlat_exp=None,
                   hill=None, hill_3p=None, hill_6p=None, rv=None):
        """Define plastic material parameters; anisotropic Hill-like and Drucker-like
        behavior is supported

        Parameters
        ----------
        sy   : float
            Yield strength
        hill : (3,) or (6,) array
            Parameters for Hill-like orthotropic anisotropy (optional, default: isotropy)
        drucker : float
            Parameter for Drucker-like tension-compression asymmetry (optional, default: 0)
        khard: float
            Linear strain hardening slope (ds/de_p) (optional, default: 0)
        tresca : Boolean
            Indicate if Tresca equivalent stress should be used (optional, default: False)
        barlat : (18,) array
            Array with parameters for Barlat Yld2004-18p yield function (optional)
        barlat_exp : int
            Exponent for Barlat Yld2004-18p yield function (optional)
        hill_3p : Boolean
            Indicate if 3-parameter Hill model shall be applied (optional, default: None)
            Will be set True automatically if 3 Hill paramaters != 1 are provided 
        hill_6p : Boolean
            Indicate if 6-parameter Hill model shall be applied (optional, default: None)
            Will be set True automatically if 6 Hill parameters are provided
        rv  : (6,) array
            Parameters for anisotropic flow aspect ratios that can be given alternatively to Hill
            parameters (optional)
        sdim : int
            Dimensionality of stress tensor to be used for plasticity, must be either 3 
            (only principal stresses are considered) or 6 (full stress tensor is considered),
            (optional, default: 6)
        """
        self.sy0 = sy  # store initial yield strength of material
        self.sy = sy  # current yield strength (may be modified by texture)
        self.khard = khard  # strain hardening slope (d flow stress / d plastic strain)
        self.drucker = drucker  # Drucker-Prager parameter: weight of hydrostatic stress
        if sdim != 3 and sdim != 6:
            raise ValueError('{} in plasticity: sdim must be either 3 or 6'.format(self.name))
        else:
            if self.sdim is not None and self.sdim != sdim:
                print('plasticity: Parameter sdim is changed. New value:', sdim)
            self.sdim = sdim
        if hill is None and rv is None:
            hill = np.ones(self.sdim)
        elif hill is None:
            hill = np.ones(self.sdim)
            if len(rv) != self.sdim:
                raise ValueError(f'plasticity: wrong dimension of yield stress ratios, must be {sdim}')
            rinv = 1./np.array(rv)
            hill[0] = rinv[0]**2 + rinv[1]**2 - rinv[2]**2
            hill[1] = rinv[1]**2 + rinv[2]**2 - rinv[0]**2
            hill[2] = rinv[2]**2 + rinv[0]**2 - rinv[1]**2
            if self.sdim == 6:
                hill[3] = rinv[3]**2
                hill[4] = rinv[4]**2
                hill[5] = rinv[5]**2
        elif hill is not None and rv is not None:
            warnings.warn('plasticity: Both, hill and rv, have been provided. Using Hill parameters.')
        lh = len(hill)
        if hill_6p is None and hill_3p is None:
            # determine if 3 or 6 Hill parameters are provided
            hill_6p = (lh == 6)
            hill_3p = not hill_6p
            if hill_3p and (hill[0] == 1.) and (hill[1] == 1.) and (hill[2] == 1.):
                hill_3p = False
        if hill_6p and lh != 6:
            raise ValueError('plasticity: When hill_6p is set True, 6 Hill parameters must be provided')
        if hill_3p and lh != 3:
            raise ValueError('plasticity: When hill_3p is set True, only 3 Hill parameters can be provided')
        if hill_6p and sdim == 3:
            warnings.warn('plasticity: 6 Hill parameters are provided, but sdim=3; ignoring shear parameters')
            hill_6p = False
            hill_3p = True
            hill = hill[0:3]
        if hill_3p and sdim == 6:
            print('Material', self.name)
            warnings.warn('plasticity: 3 Hill parameters are provided, but sdim=6; shear parameters set to 1')
            hill_3p = False
            hill_6p = True
            hill.extend([1., 1., 1.])
        if sdim == 6 and lh == 3:
            hill.extend([1., 1., 1.])
        self.hill_6p = hill_6p
        self.hill_3p = hill_3p
        self.hill = np.array(hill)  # Hill anisotropic parameters
        # set Trseca flag
        if tresca is None:
            tresca = False
        self.tresca = tresca
        if barlat is not None:
            self.barlat = True
            self.Bar_m1 = np.array([[0., -barlat[0], -barlat[1], 0., 0., 0.],
                                    [-barlat[2], 0., -barlat[3], 0., 0., 0.],
                                    [-barlat[4], -barlat[5], 0., 0., 0., 0.],
                                    [0., 0., 0., barlat[6], 0., 0.],
                                    [0., 0., 0., 0., barlat[7], 0.],
                                    [0., 0., 0., 0., 0., barlat[8]]])

            #  Cdouble dash matrix
            self.Bar_m2 = np.array([[0., -barlat[9], -barlat[10], 0., 0., 0.],
                                    [-barlat[11], 0., -barlat[12], 0., 0., 0.],
                                    [-barlat[13], -barlat[14], 0., 0., 0., 0.],
                                    [0., 0., 0., barlat[15], 0., 0.],
                                    [0., 0., 0., 0., barlat[16], 0.],
                                    [0., 0., 0., 0., 0., barlat[17]]])
            self.barlat_exp = barlat_exp
        else:
            self.barlat = False

    def from_data(self, param):
        """Define material properties from data sets generated in module `Data`:
        contains data on elastic and plastic behavior, including work hardening,
        for different crystallographic textures. Possible to extend to grain sizes,
        grain shapes and porosities. Will invoke definition of elastic and plastic
        parameters by calls to the methods `Material.elasticity` and `Material.plasticity`
        with the parameters provided in the data set.
        Also initializes current texture to first one in list and resets work hardening
        parameters.

        Parameters
        ----------
        param : list of directories
            `Data.mat_param` directories containing material data sets
        """

        # import dictionaries with all microstructure parameters resulting from data module
        self.msparam = np.array(param, ndmin=1)
        self.Nset = len(self.msparam)  # number of microstructures in material definition
        self.whdat = self.msparam[0]['wh_data']  # flag if work hardening data exists
        Nlc = self.msparam[0]['Nlc']
        Ntext = self.msparam[0]['Ntext']
        self.txdat = False if Ntext == 1 else True  # texture variations exist
        if self.sdim is None:
            self.sdim = self.msparam[0]['sdim']
        elif self.sdim != self.msparam[0]['sdim']:
            self.sdim = self.msparam[0]['sdim']
            warnings.warn('from_data: Microstructure has changed definition of sdim. New value={}'.format(self.sdim))
        if self.sdim != 3 and self.sdim != 6:
            raise ValueError('Value of sdim must be either 3 or 6')
        self.epc = self.msparam[0]['epc']
        for i in range(1, self.Nset):
            h1 = self.msparam[i]['Nlc'] != Nlc
            h3 = self.msparam[i]['Ntext'] != Ntext
            h4 = self.msparam[i]['sdim'] != self.sdim
            if h1 or h3 or h4:
                print('Error: Structure of data set #', i, ' is inconsistent:', Nlc, Ntext, self.sdim, h1, h3, h4)
                raise ValueError('Inconsistent data structure')
                # Conditions can be relaxed by modifying train_SVC

        # determine number of dof for feature vector
        self.Ndof = 2 if self.sdim == 3 else 6
        if self.whdat:
            self.ind_wh = self.Ndof  # starting index for dof associated with work hardening
            self.Ndof += 6  # add dof's for work hardening parameter if data exists
        if self.txdat:
            self.ind_tx = self.Ndof  # starting index for dof associated with textures
            self.Ndof += self.Nset  # add dof's for textures

        # assign average properties to material and initialize texture and work-hardening
        self.elasticity(E=self.msparam[0]['E_av'], nu=self.msparam[0]['nu_av'])
        self.plasticity(sy=self.msparam[0]['sy_av'], sdim=self.sdim)
        tp = np.zeros(self.Nset)
        tp[0] = 1.
        self.set_texture(tp)

    def from_MLparam(self, name, path='../../models/'):
        """Define material properties from parameters of trained machine learning 
        models that have been written with `Material.export_MLparam`.
        Will invoke definition of elastic parameters by calls to the methods 
        `Material.elasticity` with the parameters provided in the data set. 
        Also initializes current texture to first one in list and resets work hardening
        parameters.

        Parameters
        ----------
        name : string
            Name of parameter files (`name`.csv file and metadata file `name_meta.json`)
        path : string
            Path in which files are stored (optional, default: '../../models/')            
        """
        raise ModuleNotFoundError('Import from ML parameters not yet implemented.')

    def set_texture(self, current, verb=False):
        """Set parameters for current crystallographic texture of material as defined in microstructure.
        
        Parameters
        ----------
        current : float or list
            Mixture parameter for microstructures in range [0,1] for each defined microstructure, indicates the 
            intensity of the given microstructure. The sum of all mixture parameters must be <=1, the rest will be set 
            to random texture. Must have same dimension as material.msparam.
        verb : Boolean
            Be verbose 
            
        Yields
        ------
        Material.tx_cur : list
            Current value of microstructural mixture parameter for active microstructure. Has same dimension as material.msparam
        Material.sy  : float
            Yield stength is redefined accoring to texture parameter
        Material.khard : float
            Work hardening parameter is redefined according to texture parameter
        Material.epc : float
            Critical PEEQ for which onset of plastic deformation is definied in data
        """
        self.tx_cur = np.array(current, ndmin=1)
        sm = np.sum(self.tx_cur)  # sum of mixture parameters
        if sm > 1. or sm < 0.:
            print('Error: Microstructure parameters out of range:', sm, current)
            raise ValueError('set_texture: Bad value for mixture parameter')
        if len(self.tx_cur) != self.Nset:
            print('Error: Microstructure parameters have wrong dimension:', current, self.Nset)
            raise ValueError('set_texture: Wrong dimension of mixture parameter')
        # calculate weight factor for each texture based on its mixture parameter
        if sm < 1.e-3:
            wght = np.ones(self.Nset) / self.Nset
        else:
            wght = self.tx_cur / sm
        self.sy = 0.
        # self.khard = 0.
        index = []
        for i, ms in enumerate(self.msparam):
            hh = ms['texture'] - self.tx_cur[i]
            index.append(np.argmin(np.abs(hh)))
            # redefine plasticity parameters according to texture parameter
            sy = ms['sy_av']  # np.interp(self.tx_cur[i], ms['texture'], ms['flow_seq_av'][:, 0])
            self.sy += sy * wght[i]
            '''if self.whdat:
                #set strain hardening parameters to initial value for selected texture
                ds = ms['flow_seq_av'][index[-1],1] - ms['flow_seq_av'][index[-1],0] # assuming isotropic hardening
                de = ms['work_hard'][1] - ms['work_hard'][0]
                khard   =  ds/de # linear work hardening rate b/w values for w.h. in data
                self.khard += khard*wght[i]'''

        if verb:
            print('New texture parameters: ', self.tx_cur)
            print('Texture mixing: ')
            [print(self.msparam[i]['ms_type'], 'with mixture parameter', self.tx_cur[i]) for i in range(self.Nset)]
            print('Yield strength:', self.sy, 'MPa')
        self.ms_index = index

    # ==============================================================
    # subroutines for post-processing and graphics
    # ==============================================================

    def ellipsis(self, a=1., b=1. / np.sqrt(3.), n=72):
        """Create ellipsis with main axis along 45° axis, used for graphical representation of isotropic yield locus.

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
        """
        t = np.arange(0., 2.1 * np.pi, np.pi / n)
        x = a * np.cos(t) - b * np.sin(t)
        y = a * np.cos(t) + b * np.sin(t)
        return x, y

    def plot_data(self, Z, axs, xx, yy, field=True, c='red'):
        """Plotting data in stress space to visualize yield loci.

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
        """
        # symmetrize Z values
        zmin = np.amin(Z)
        zmax = np.amax(Z)
        if (-zmin < zmax):
            Z[np.nonzero(Z > -zmin)] = -zmin
        else:
            Z[np.nonzero(Z < -zmax)] = -zmax
        Z = Z.reshape(xx.shape)

        # display data
        if field:
            axs.imshow(Z, interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
                       origin='lower', cmap=plt.cm.PuOr_r)
        contour = axs.contour(xx, yy, Z, levels=[0], linewidths=2,
                              linestyles='solid', colors=c)
        line = contour.collections
        return line

    def plot_yield_locus(self, fun=None, label=None, data=None, trange=1.e-2, peeq=0.,
                         xstart=None, xend=None, axis1=[0], axis2=[1], iso=False, ref_mat=None,
                         field=False, Nmesh=100, file=None, fontsize=20, scaling=True):
        """Plot different cuts through yield locus in 3D principal stress space.

        Parameters
        ----------
        fun   : function handle
            Yield function to be plotted (optional, default: own yield function)
        label : str
            Label for yield function (optional, default: own name)
        data  : (N,3) array
            principal stress data to be used for scatter plot (optional)
        trange : float
            Cut-off for data to be plotted on slice (optional, default: 1.e-2)
        peeq   : float
            Level of plastic strain for which yield locus is plotted (isotropic hardening)
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
        scaling  : Boolean
            Scale stress with yield strength (optional, default: True)

        Returns
        -------
        axs : pyplot axis handle
            Axis of the plot
        """
        if xstart is None:
            if scaling:
                xstart = -2.
            else:
                xstart = -2. * self.sy
        if xend is None:
            if scaling:
                xend = 2.
            else:
                xend = 2. * self.sy
        xx, yy = np.meshgrid(np.linspace(xstart, xend, Nmesh),
                             np.linspace(xstart, xend, Nmesh))
        Nm2 = Nmesh * Nmesh
        Nc = len(axis1)
        if len(axis2) != Nc:
            sys.exit('Error in plot_yield_locus: mismatch in dimensions of ax1 and ax2')

        if Nc == 1:
            fs = (10, 8)
            fontsize *= 4 / 5
            plt.subplots_adjust(wspace=0.3)
        else:
            fs = (20, 5)
        fig, axs = plt.subplots(nrows=1, ncols=Nc, figsize=fs)
        fig.subplots_adjust(hspace=0.3)

        # loop over subplots in axis1 and axis2
        for j in range(Nc):
            if Nc == 1:
                ax = axs
            else:
                ax = axs[j]
            lines = []
            labels = []
            # select slice in 3D stress space
            s1 = None
            s2 = None
            s3 = None
            # first axis
            if axis1[j] == 0:
                s1 = xx.ravel()
                title = r'$\sigma_1$'
                if scaling:
                    xlab = r'$\sigma_1 / \sigma_y$'
                else:
                    xlab = r'$\sigma_1$ (MPa)'
            elif axis1[j] == 1:
                s2 = xx.ravel()
                title = r'$\sigma_2$'
                if scaling:
                    xlab = r'$\sigma_2 / \sigma_y$'
                else:
                    xlab = r'$\sigma_2$ (MPa)'
            elif axis1[j] == 2:
                s3 = xx.ravel()
                title = r'$\sigma_3$'
                if scaling:
                    xlab = r'$\sigma_3 / \sigma_y$'
                else:
                    xlab = r'$\sigma_3$ (MPa)'
            elif axis1[j] == 3:
                s1 = xx.ravel()
                s2 = xx.ravel()
                title = r'$p=\sigma_1=\sigma_2$'
                if scaling:
                    xlab = r'$p / \sigma_y$'
                else:
                    xlab = '$p$ (MPa)'
                ref_mat = False
                axis1[j] = 0
            else:
                warnings.warn('plot_yield_locus: axis1 not defined properly, set to sig_1:{} {}'.format(axis1, j))
                s1 = xx.ravel()
                title = r'$\sigma_1$'
                if scaling:
                    xlab = r'$\sigma_1 / \sigma_y$'
                else:
                    xlab = r'$\sigma_1$ (MPa)'
            # second axis
            if axis2[j] == 0:
                s1 = yy.ravel()
                title += r'-$\sigma_1$ slice'
                if scaling:
                    ylab = r'$\sigma_1 / \sigma_y$'
                else:
                    ylab = r'$\sigma_1$ (MPa)'
            elif axis2[j] == 1:
                s2 = yy.ravel()
                title += r'-$\sigma_2$ slice'
                if scaling:
                    ylab = r'$\sigma_2 / \sigma_y$'
                else:
                    ylab = r'$\sigma_2$ (MPa)'
            elif axis2[j] == 2:
                s3 = yy.ravel()
                title += r'-$\sigma_3$ slice'
                if scaling:
                    ylab = r'$\sigma_3 / \sigma_y$'
                else:
                    ylab = r'$\sigma_3$ (MPa)'
            elif axis2[j] == 3:
                s3 = yy.ravel()
                title += r'-$\sigma_3$ slice'
                if scaling:
                    ylab = r'$\sigma_3 / \sigma_y$'
                else:
                    ylab = r'$\sigma_3$ (MPa)'
                axis2[j] = 2
            else:
                warnings.warn('plot_yield_locus: axis2 not defined properly, set to sig_2: {} {}'.format(axis2, j))
                s2 = yy.ravel()
                title += r'-$\sigma_2$ slice'
                if scaling:
                    ylab = r'$\sigma_2 / \sigma_y$'
                else:
                    ylab = r'$\sigma_2$ (MPa)'
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

            # evaluate yield function to be plotted
            if scaling:
                sf = 1. / self.sy
                sig *= self.sy
            else:
                sf = 1.
            if fun is None:
                Z = self.calc_yf(sig, epl=peeq, pred=True) * sf
            else:
                Z = fun(sig, pred=True) * sf
            if label is None:
                label = self.name
            hl = self.plot_data(Z, ax, xx, yy, field=field)
            lines.extend(hl)
            labels.extend([label])
            # plot reference function if provided
            if ref_mat is not None:
                Z = ref_mat.calc_yf(sig, epl=peeq, pred=True) * sf
                labels.extend([ref_mat.name])
                hl = self.plot_data(Z, ax, xx, yy, field=False, c='black')
                lines.extend(hl)
            # plot ellipsis as reference if requested
            if iso:
                x0, y0 = self.ellipsis()  # reference for isotropic material
                if not scaling:
                    x0 *= self.sy
                    y0 *= self.sy
                hl = ax.plot(x0, y0, '-b')
                lines.extend(hl)
                labels.extend(['isotropic J2'])
            # plot data if provided
            if (data is not None):
                # select data from 3D stress space within range [xstart, xend] and to fit to slice
                dat = np.array(data) * sf
                dsel = np.nonzero(np.logical_and(np.abs(dat[:, si3]) < trange,
                                                 np.logical_and(dat[:, axis1[j]] > xstart, dat[:, axis1[j]] < xend)))
                ir = dsel[0]
                yf = np.sign(self.calc_yf(data[ir, :], epl=peeq))
                ax.scatter(dat[ir, axis1[j]], dat[ir, axis2[j]], s=60, c=yf,
                           cmap=plt.cm.Paired, edgecolors='k')
            ax.legend(lines, labels, loc='upper left', fontsize=fontsize - 4)
            # ax.set_title(title,fontsize=fontsize)
            ax.set_xlabel(xlab, fontsize=fontsize)
            ax.set_ylabel(ylab, fontsize=fontsize)
            hh = 4 if Nc == 1 else 7
            ax.tick_params(axis="x", labelsize=fontsize - hh)
            ax.tick_params(axis="y", labelsize=fontsize - hh)
        # save plot to file if filename is provided
        if file is not None:
            fig.savefig(file + '.pdf', format='pdf', dpi=300)
        return axs

    def calc_properties(self, size=2, Nel=2, verb=False, eps=0.005, min_step=None,
                        sigeps=False, load_cases=['stx', 'sty', 'et2', 'ect']):
        """Use pylabfea.model to calculate material strength and stress-strain data along a given load path.

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
        load_cases : list
            List of load cases to be performed (optional, default: ['stx','sty','et2','ect']);
            'stx': uniaxial tensile yield stress in horizontal (x-)direction;
            'sty': uniaxial tensile yield stress in vertical (y-)direction;
            'et2': plane stress, equibiaxial strain in x and y direction;
            'ect': pure shear strain (x-compression, y-tension), plane stress
        """

        def calc_strength(vbc1, nbc1, vbc2, nbc2, sel):
            fe = Model(dim=2, planestress=True)
            fe.geom([size], LY=size)  # define section in absolute length
            fe.assign([self])  # assign material to section
            fe.bcleft(0.)  # fix lhs nodes in x-direction
            fe.bcbot(0.)  # fix bottom nodes in y-direction
            fe.bcright(vbc1, nbc1)  # define BC in x-direction
            fe.bctop(vbc2, nbc2)  # define BC in y-direction
            fe.mesh(NX=Nel, NY=Nel)  # create mesh
            fe.solve(verb=verb, min_step=min_step)  # solve mechanical equilibrium condition under BC
            seq = self.calc_seq(fe.sgl)  # store time dependent mechanical data of model
            eeq = eps_eq(fe.egl)
            peeq = eps_eq(fe.epgl)
            iys = np.nonzero(peeq < 1.e-2)
            ys = seq[iys[0][-1]]
            self.prop[sel]['ys'] = ys
            self.prop[sel]['seq'] = seq
            self.prop[sel]['eeq'] = eeq
            self.prop[sel]['peeq'] = peeq
            seq = sig_eq_j2(fe.sgl)  # store time dependent mechanical data of model
            eeq = eps_eq(fe.egl)
            peeq = eps_eq(fe.epgl)
            iys = np.nonzero(peeq < 1.e-6)  # take stress at last index of elastic regime
            ys = seq[iys[0][-1]]
            self.propJ2[sel]['ys'] = ys
            self.propJ2[sel]['seq'] = seq
            self.propJ2[sel]['eeq'] = eeq
            self.propJ2[sel]['peeq'] = peeq
            if sigeps:
                self.sigeps[sel]['sig'] = fe.sgl
                self.sigeps[sel]['eps'] = fe.egl
                self.sigeps[sel]['epl'] = fe.epgl
            return

        def calc_stx():
            u1 = eps * size
            calc_strength(u1, 'disp', 0., 'force', 'stx')
            self.prop['stx']['style'] = '-r'
            self.prop['stx']['name'] = 'uniax-x'
            return

        def calc_sty():
            u2 = eps * size
            calc_strength(0., 'force', u2, 'disp', 'sty')
            self.prop['sty']['style'] = '-b'
            self.prop['sty']['name'] = 'uniax-y'
            return

        def calc_et2():
            u1 = 0.4 * eps * size
            u2 = 0.4 * eps * size
            calc_strength(u1, 'disp', u2, 'disp', 'et2')
            self.prop['et2']['style'] = '-k'
            self.prop['et2']['name'] = 'equibiax'
            return

        def calc_ect():
            u1 = -0.8 * eps * size
            u2 = 0.8 * eps * size
            calc_strength(u1, 'disp', u2, 'disp', 'ect')
            self.prop['ect']['style'] = '-m'
            self.prop['ect']['name'] = 'shear'
            return

        # calculate strength and stress strain data along given load paths
        for case in load_cases:
            if case == 'stx':
                calc_stx()  # uniaxial tensile yield stress in horizontal (x-)direction
            elif case == 'sty':
                calc_sty()  # uniaxial tensile yield stress in vertical (y-)direction
            elif case == 'et2':
                calc_et2()  # plane stress, equibiaxial strain in x and y direction
            elif case == 'ect':
                calc_ect()  # pure shear strain (x-compression, y-tension), plane stress
            else:
                warnings.warn('calc_properties: Load case not supported: {}'.format(case))

    def plot_stress_strain(self, Hill=False, file=None, fontsize=14):
        """Plot stress-strain data and print values for strength.

        Parameters
        ----------
        Hill : Boolean
            Decide if data for Hill-type equivalent stress is presented (optional, default: False)
        file : str
            Filename to save plot (optional)
        fontsize : int
            Fontsize for axis annotations (optional, default: 14)
        """
        legend = []
        print('---------------------------------------------------------')
        for sel in self.prop:
            if self.propJ2[sel]['ys'] is not None:
                print('J2 yield stress under', self.prop[sel]['name'], 'loading:',
                      self.propJ2[sel]['ys'].round(decimals=3), 'MPa')
                print('---------------------------------------------------------')
                plt.plot(self.propJ2[sel]['eeq'] * 100., self.propJ2[sel]['seq'], self.prop[sel]['style'])
                legend.append(self.prop[sel]['name'])
        plt.title('Material: ' + self.name, fontsize=fontsize)
        plt.xlabel(r'$\epsilon_\mathrm{eq}$ (%)', fontsize=fontsize)
        plt.ylabel(r'$\sigma^\mathrm{J2}_\mathrm{eq}$ (MPa)', fontsize=fontsize)
        plt.tick_params(axis="x", labelsize=fontsize - 4)
        plt.tick_params(axis="y", labelsize=fontsize - 4)
        plt.legend(legend, loc='lower right', fontsize=fontsize)
        if file is not None:
            plt.savefig(file + 'J2.pdf', format='pdf', dpi=300)
        plt.show()
        if Hill:
            for sel in self.prop:
                if self.prop[sel]['ys'] is not None:
                    print('Hill yield stress under', self.prop[sel]['name'], 'loading:',
                          self.prop[sel]['ys'].round(decimals=3), 'MPa')
                    print('---------------------------------------------------------')
                    plt.plot(self.prop[sel]['eeq'] * 100., self.prop[sel]['seq'], self.prop[sel]['style'])
                    legend.append(self.prop[sel]['name'])
            plt.title('Material: ' + self.name, fontsize=fontsize)
            plt.xlabel(r'$\epsilon_\mathrm{eq}$ (%)', fontsize=fontsize)
            plt.ylabel(r'$\sigma_\mathrm{eq}$ (MPa)', fontsize=fontsize)
            plt.tick_params(axis="x", labelsize=fontsize - 4)
            plt.tick_params(axis="y", labelsize=fontsize - 4)
            plt.legend(legend, loc='lower right', fontsize=fontsize)
            if file is not None:
                plt.savefig(file + 'Hill.pdf', format='pdf', dpi=300)
            plt.show()
        return

    def polar_plot_yl(self, Na=72, cmat=None, data=None, dname='reference', scaling=None,
                      field=False, predict=False, cbar=False, Np=100, file=None, arrow=False,
                      sJ2=False, show=True):
        """Plot yield locus as polar plot in deviatoric stress plane
        
        Parameters
        ----------
        Na : int
            Number of angles on which yield locus is evaluated (optional, default: 72)
        cmat : list of materials
            Materials of which YL is plotted in same plot with same scaling (optional)
        data : (N,3) array
            Array of cylindrical stress added to plot (optional)
        dname : str
            Label for data (optional, default: reference)
        scaling : float
            Scaling factor for stresses (optional)
        field : Boolean
            Field of decision function is plotted, works only together with ML yield function
            (optional, default: False)
        predict : Boolean
            Plot ML prediction (-1,1), otherwise decision fucntion is plotted (optional, default: False)
        Np      : int
            Number of points per axis for field plot (optional, default: 100)
        cbar    : Boolean
            Plot colorbar for field (optional, default: False)
        file  : str
            Name of PDF file to which plot is saved
        arrow : Boolean
            Indicate if arrows for the pricipal stress directions are shown (optional, default: False)
        sJ2   : Boolean
            Indicate that J2 equivalent stress shall be used instead of material definition of 
            equivalent stress (optional, default: False)
            
        """
        if scaling is None:
            sf = 1.
        else:
            sf = 1. / scaling
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_axes([0, 0, 1, 1, ], projection='polar')
        if field and self.ML_yf:
            xx, yy = np.meshgrid(np.linspace(-1., 1., Np), np.linspace(-1, 1., Np))
            if self.Ndof == 2:
                feat = np.c_[yy.ravel(), xx.ravel()]
            elif self.Ndof == 3:
                hh = -np.ones(Np * Np)
                feat = np.c_[yy.ravel(), xx.ravel(), hh]
            else:
                raise ValueError(
                    '"polar_plot_yl" currently does not support texture as degree of freedom for field plots.')
            cmap = plt.cm.get_cmap('PuOr_r')  # 'bwr' and 'PuOr_r' are good choices
            if predict:
                Z = self.svm_yf.predict(feat)
            else:
                Z = self.svm_yf.decision_function(feat)
            'symmetrize Z values'
            zmin = np.amin(Z)
            zmax = np.amax(Z)
            if (-zmin < zmax):
                Z[np.nonzero(Z > -zmin)] = -zmin
            else:
                Z[np.nonzero(Z < -zmax)] = -zmax
            Z = Z.reshape(xx.shape)
            im = ax.pcolormesh(xx * np.pi, (yy + 1.) * self.scale_seq * sf, Z, cmap=cmap,
                               shading='auto')
            if cbar:
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel("yield function (MPa)", rotation=-90)
        # find norm of princ. stess vector lying on yield surface
        theta = np.linspace(0., 2 * np.pi, Na)
        snorm = sig_cyl2princ(np.array([self.sy * np.ones(Na) * np.sqrt(1.5), theta]).T)
        x1 = fsolve(self.find_yloc, np.ones(Na), args=snorm, xtol=1.e-5)
        sig = snorm * np.array([x1, x1, x1]).T
        if sJ2:
            s_yld = sig_eq_j2(sig)
        else:
            s_yld = self.calc_seq(sig)
        ax.plot(theta, s_yld * sf, '-r', linewidth=2, label=self.name)
        if cmat is not None:
            N = len(cmat)
            cmap = plt.cm.get_cmap('copper')
            for i, mat in enumerate(cmat):
                x1 = fsolve(mat.find_yloc, np.ones(Na), args=snorm, xtol=1.e-5)
                sig = snorm * np.array([x1, x1, x1]).T
                if sJ2:
                    s_yld = sig_eq_j2(sig)
                else:
                    s_yld = self.calc_seq(sig)
                ax.plot(theta, s_yld * sf, color=cmap(i / N), linewidth=2, label=mat.name)
        if data is not None:
            ax.plot(data[:, 1], data[:, 0] * sf, '.b', label=dname)
        if arrow:
            dr = self.sy
            drh = 0.08 * dr
            ax.arrow(0, 0, 0, dr, head_width=0.05, width=0.004,
                     head_length=drh, color='r', length_includes_head=True)
            ax.text(-0.12, dr * 0.89, r'$\sigma_1$', color='r', fontsize=22)
            ax.arrow(2.0944, 0, 0, dr, head_width=0.05,
                     width=0.004, head_length=drh, color='r', length_includes_head=True)
            ax.text(2.24, dr * 0.94, r'$\sigma_2$', color='r', fontsize=22)
            ax.arrow(-2.0944, 0, 0, dr, head_width=0.05,
                     width=0.004, head_length=drh, color='r', length_includes_head=True)
            ax.text(-2.04, dr * 0.97, r'$\sigma_3$', color='r', fontsize=22)
        if file is not None:
            plt.legend(loc=(.9, 0.95), fontsize=18)
            plt.savefig(file + '.pdf', format='pdf', dpi=300)
        if show:
            plt.legend(loc=(.9, 0.95), fontsize=18)
            plt.show()
        return ax
