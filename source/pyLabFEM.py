#Module FEM
'''Introduces global functions for mechanical quantities and class ``Model`` that 
contains that attributes and methods needed in FEA. Materials are defined in 
module pyLabMaterial.

uses NumPy, SciPy, MatPlotLib and pyLabMaterial

Version: 1.0 (2020-03-06)
Author: Alexander Hartmaier, ICAMS/Ruhr-University Bochum, March 2020
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)'''
import numpy as np
from scipy.optimize import fsolve
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

'==================================='
'define global methods and variables'
'==================================='
a_vec = np.array([1., -0.5, -0.5])/np.sqrt(1.5)
'''First unit vector spanning deviatoric stress plane (real axis)'''
b_vec = np.array([0.,  0.5, -0.5])*np.sqrt(2)  
'''Second unit vector spanning deviatoric stress plane (imaginary axis)'''

def seq_J2(sprinc):
    '''Calculate J2 equivalent stress from principal stresses
    
    Parameters
    ----------
    sprinc : (3,), (N,3) or (N,6) array
         (3,) or (N,3): Principal stress or list of principal stresses;
         (N,6): Voigt stress
         
    Returns
    -------
    seq : float or (N,) array
        J2 equivalent stresses
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
        print('*** seq_J2: N, sh', N, sh, sys._getframe().f_back.f_code.co_name)
        sys.exit('Error: Unknown format of stress in seq_J2')
    d12 = sprinc[:,0] - sprinc[:,1]
    d23 = sprinc[:,1] - sprinc[:,2]
    d31 = sprinc[:,2] - sprinc[:,0]
    J2  = 0.5*(np.square(d12) + np.square(d23) + np.square(d31))
    seq  = np.sqrt(J2)  # J2 eqiv. stress
    if sh==(3,): 
        seq = seq[0]
    return seq

def polar_ang(sprinc):
    '''Transform principal stresses into polar angle on deviatoric plane spanned by a_vec and b_vec
    
    Parameters
    ----------
    sprinc : (3,), (N,3) or (N,6) array
         (3,), (N,3): Principal stresses; 
         (N,6): Voigt stress
         
    Returns
    -------
    theta : float or (N,) array
        polar angles in deviatoric plane as positive angle between sprinc and a_vec in range [-pi,+p]
    '''
    sh = np.shape(sprinc) 
    N = len(sprinc)
    if sh==(3,):
        N = 1
        sprinc = np.array([sprinc])
    elif sh==(N,6):
        sig = sprinc
        sprinc=np.zeros((N,3))
        for i in range(N):
            sprinc[i,:] = Stress(sig[i,:]).p
    elif sh!=(N,3):
        print('*** polar_angle: N, sh', N, sh, sys._getframe().f_back.f_code.co_name)
        sys.exit('Error: Unknown format of stress in polar_angle')
    hyd = np.sum(sprinc,axis=1)/3.  # hydrostatic component
    dev = sprinc - np.array([hyd, hyd, hyd]).transpose() # deviatoric princ. stress
    vn  = np.linalg.norm(dev,axis=1)  #norms of stress vectors
    dsa = np.dot(dev,a_vec)
    dsb = np.dot(dev,b_vec)
    theta = np.angle(dsa + 1j*dsb)
    if sh==(3,):
        theta=theta[0]
    return theta

def sp_cart(scyl):
    '''convert cylindrical stress into 3D Cartesian vector of deviatoric principle stresses 
    
    Parameters
    ----------
    scyl : (2,), (3,), (N,2) or (N,3) array 
         Cylindrical stress in form (seq, theta, (optional: p))
         
    Returns
    -------
    sprinc : float or (N,) array
        principle deviatoric stress
    '''
    sh = np.shape(scyl) 
    if sh==(2,) or sh==(3,):
        scyl = np.array([scyl])
    seq = scyl[:,0]
    theta = scyl[:,1]
    sprinc = (np.tensordot(np.cos(theta), a_vec, axes=0) + 
              np.tensordot(np.sin(theta), b_vec, axes=0)) \
             *np.sqrt(2./3.)*np.array([seq,seq,seq]).T
    if sh[0]==3:
        p = scyl[:,3]
        sprinc += np.array([p,p,p]).T
    if sh==(2,) or sh==(3,):
        sprinc=sc[0]
    return sprinc

def s_cyl(sprinc, mat=None):
    '''convert principal stress into cylindrical stress vector 
    
    Parameters
    ----------
    sprinc : (3,) or (N,3) array 
         principal stresses
    mat : object of class ``Material``
        Material for Hill-type principal stress (optional) 
         
    Returns
    -------
    sc : (3,) or (N,3) array
        stress in cylindrical coordinates (seq, theta, p)
    '''
    sh = np.shape(sprinc) 
    N = len(sprinc)
    if sh==(3,):
        sprinc = np.array([sprinc])
        N = 1
    sc = np.zeros((N,3))
    if mat is None:
        sc[:,0] = seq_J2(sprinc)
    else:
        sc[:,0] = mat.calc_seq(sprinc)
    sc[:,1] = polar_ang(sprinc)
    sc[:,2] = np.sum(sprinc, axis=1)/3.
    if sh==(3,):
        sc=sc[0]
    return sc

def eps_eq(eps):
    '''Calculate equivalent strain 
    
    Parameters
    ----------
    eps : (3,), (6,), (N,3) or (N,6) array
         (3,) or (N,3): Principal strains;
         (6,) or (N,6): Voigt strains
         
    Returns
    -------
    eeq : float or (N,) array
        equivalent strains
    '''
    sh = np.shape(eps)
    N = len(eps)
    if sh==(6,):
        eeq = np.sqrt(2.*(np.dot(eps[0:3],eps[0:3])+2*np.dot(eps[3:6],eps[3:6]))/3.)
    elif sh==(3,):
        eeq = np.sqrt(2.*np.sum(eps*eps)/3.)
    elif sh==(N,3):
        eeq = np.sqrt(2.*np.sum(eps*eps,axis=1)/3.)
    elif sh==(N,6):
        eeq = np.sqrt(2.*(np.sum(eps[:,0:3]*eps[:,0:3],axis=1)+2*np.sum(eps[:,3:6]*eps[:,3:6],axis=1))/3.)
    else:
        print('*** eps_eq (N,sh): ',N,sh,sys._getframe().f_back.f_code.co_name)
        sys.exit('Error in eps_eq: Format not supported')
    return eeq

'========================='
'define class for stresses'
'========================='
class Stress(object):
    '''Stores and converts Voigt stress tensors into different formats, 
    calculates principle stresses, equivalent stresses and transforms into cylindrical coordinates.
    
    Parameters
    ----------
    sv : list-like object, must be 1D with length 6
        Voigt-stress components
    
    Attributes
    ----------
    voigt, v : 1d-array (size 6)
        Stress tensor in Voigt notation
    tens, t : 3x3 array
        Stress tensor in matrix notation
    princ, p : 1d-array (size 3)
        Principal stresses
    hydrostatic, h : float
        Hydrostatic stress component
    '''
    def __init__(self, sv):
        self.v=self.voigt = np.array(sv)
        #calculate (3x3)-tensorial representation
        self.t=self.tens = np.zeros((3,3))
        self.tens[0,0]=sv[0]
        self.tens[1,1]=sv[1]
        self.tens[2,2]=sv[2]
        self.tens[2,1]=self.tens[1,2]=sv[3]
        self.tens[2,0]=self.tens[0,2]=sv[4]
        self.tens[1,0]=self.tens[0,1]=sv[5]
        #calcualte principal stresses and eigen vectors
        self.princ, self.evec = np.linalg.eig(self.tens)
        self.p=self.princ
        self.h=self.hydrostatic = np.sum(self.p)/3.
            
    def seq(self, mat):
        '''calculate Hill-type equivalent stress, invokes corresponding method of class ``Material``
        
        Parameters
        ----------
        mat: object of class ``Material``
            containes Hill parameters and method needed for Hill-type equivalent stress
        
        Returns
        -------
        seq : float
            equivalent stress of Hill-type
        '''
        seq = mat.calc_seq(self.p)
        return seq
    
    def theta(self):
        '''calculate polar angle in deviatoric plane
        
        Returns
        -------
        ang : float
            polar angle of stress in devitoric plane
        '''
        ang = polar_ang(self.p)
        return ang

    def sJ2(self):
        '''calculate J2 principal stress
        
        Returns
        -------
        sJ2 : float
            equivalent stress
        '''
        sJ2 = seq_J2(self.p)
        return sJ2
    
    def lode_ang(self, X):
        '''Calculate Lode angle:  
        Transforms principal stress space into hydrostatic stress, eqiv. stress, and Lode angle; 
        definition of positive cosine for Lode angle is applied
        
        Parameters
        ----------
        X : either float or object of class ``Material``
            if float: interpreted as equivalent stress
            if ``Material``: used to invoke method of class ``Material`` to calculate equivalent stress
        
        Returns
        -------
        la : float
            Lode angle
        '''
        if type(X) is float:
            seq = X # float-type parameters are interpreted as equiv. stress
        else:
            seq = self.seq(X) # otherwise parameter is Material
        J3 = np.linalg.det(self.tens - self.h*np.diag(np.ones(3)))
        hh = 0.5*J3*(3./seq)**3
        la = np.arccos(hh)/3.
        return la

    
'======================='
'define class for strain'
'======================='
class Strain(object):
    '''Stores and converts Voigt strain tensors into different formats, 
    calculates principle strain and equivalent strain.
    
    Parameters
    ----------
    sv : list-like object, must be 1D with length 6
        Voigt-strain components
        
    Attributes
    ----------
    voigt, v : 1d-array (size 6)
        Strain tensor in Voigt notation
    tens, t : 3x3 array
        Strain tensor in matrix notation
    princ, p : 1d-array (size 3)
        Principal strains
    '''
    def __init__(self, sv):
        self.v=self.voigt = np.array(sv)
        #calculate (3x3)-tensorial representation
        self.t=self.tens = np.zeros((3,3))
        self.tens[0,0]=sv[0]
        self.tens[1,1]=sv[1]
        self.tens[2,2]=sv[2]
        self.tens[2,1]=self.tens[1,2]=sv[3]
        self.tens[2,0]=self.tens[0,2]=sv[4]
        self.tens[1,0]=self.tens[0,1]=sv[5]
        #calcualte principal stresses and eigen vectors
        self.princ, self.evec = np.linalg.eig(self.tens)
        self.p=self.princ
        
    def eeq(self):
        '''Calculate equivalent strain
        
        Returns
        -------
        eeq : float
            Equivalent strain
        '''
        eeq=eps_eq(self.v)
        return eeq
    
'========================='   
'define class for FE model'
'========================='
class Model(object):
    '''Class for finite element model. Defines necessary attributes and methods
    for pre-processing (defining geometry, material assignments, mesh and boundary conditions);
    solution (invokes numerical solver for non-linear problems);
    and post-processing (visualization of results on meshed geometry, homogenization of results into global quantities)
    
    Parameters
    ----------
    dim  : int
        Dimensionality of model (optional, default: 1)
    planestress : Boolean
        Sets plane-stress condition (optional, default: False)
    Attributes
    ----------
    dim   : integer
        Dimensionality of model (1 and 2 are supported in this version)
    planestress : Boolean
        Sets plane-stress condition
    Nsec  : int
        Number of sections (defined in ``geom``)
    LS    : 1-d array
        Absolute lengths of sections (defined in ``geom``)
    lenx  : float
        Size of model in x-direction (defined in ``geom``)
    leny  : float
        Size of model in y-direction (defined in ``geom``, default leny=1)
    thick : float
        Thickness of 2d-model (defined in ``geom``, default thick=1)
    nonlin : Boolean
        Indicates non-linearity of model (defined in ``assign``)
    mat   : list of objects to class Material
        List of materials assigned to sections of model, same dimensions as LS (defined in ``assign``)
    ubot  : float
        Nodal displacements in x-direction for lhs nodes (defined in ``bcbot``)
    uleft : float
        Nodal displacement in x-direction for lhs nodes (defined in ``bcleft``)
    ubcright: Boolean
        True: displacement BC on rhs nodes; False: force BC on rhs nodes (defined in ``bcright``)
    bcr   : float
        Nodal displacements/forces in x-direction for rhs nodes (defined in ``bcright``)
    ubctop: Boolean
        True: displacement BC on top nodes; False: force BC on top nodes (defined in ``bctop``)
    bct   : float 
        Nodal displacements/forces in y-direction for top nodes (defined in ``bctop``)
    Nnode : int
        Total number of nodes in Model (defined in ``mesh``)
    NnodeX, NnodeY : int
        Numbers of nodes in x and y-direction (defined in ``mesh``)
    Nel   : int
        Number of elements (defined in ``mesh``)
    Ndof  : int
        Number of degrees of freedom (defined in ``mesh``)
    npos: 1d-array
        Nodal positions, dimension Ndof, same structure as u,f-arrays: (x1, y1, x2, y1, ...) (defined in ``mesh``)
    noleft, noright, nobot, notop : list 
        Lists of nodes on boundaries (defined in ``mesh``)
    noinner : list
        List of inner nodes (defined in ``mesh``)
    element : list
        List of objects of class Element, dimension Nel (defined in ``mesh``)
    u    : 1d-array
        List of nodal displacements (defined in ``solve``)
    f    : 1d-array
        List of nodal forces (defined in ``solve``)
    sgl  : 1d-array
        Time evolution of global stress (defined in ``solve``)
    egl  : 1d-array
        Time evolution of global total strain (defined in ``solve``)
    epgl : 1d-array
        Time evolution of global plastic strain (defined in ``solve``)
    glob : python dictionary
        Global values homogenized from BC or element solutions (defined in ``calc_global``)
    '''
    
    def __init__(self, dim=1, planestress = False):
        # set dimensions and type of model
        # initialize boundary conditions
        # dim: dimensional of model (currently only 1 or 2 is possible)
        # planestress: (boolean) plane stress condition
        if ((dim!=1)and(dim!=2)):
            exit('dim must be either 1 or 2')
        self.dim = dim
        if planestress and (dim!=2):
            print('Warning: Plane stress only defined for 2-d model')
            planestress = False
        self.planestress = planestress
        self.uleft = None  # create warning if user sets no value
        self.ubot = None  
        self.bct = None
        self.bcr = None
        self.ubctop = False # default free boundary on top
        self.ubcright = False # default free boundary on rhs
        self.nonlin = False # default is linear elastic model
        self.sgl = np.zeros((1,6))  # list for time evolution of global stresses
        self.egl = np.zeros((1,6))  # list for time evolution of global strains
        self.epgl = np.zeros((1,6)) # list for time evolution of global plastic strain
        self.u = None
        self.f = None
        self.glob = {
            'ebc1'   : None,  # global x-strain from BC
            'ebc2'   : None,  # global y-strain from BC
            'sbc1'   : None,  # global x-strain from BC
            'sbc2'   : None,  # global y-strain from BC
            'eps'    : np.zeros(6),  # global strain tensor from element solutions
            'sig'    : np.zeros(6),  # global stress tensor from element solutions
            'epl'    : np.zeros(6)   # global plastic strain tensor from element solutions
        }
    '----------------------'
    'Sub-Class for elements'
    '----------------------'
    class Element(object):
        '''Class for isoparametric elements; supports
        1-d elements with linear or quadratic shape function and full integration;
        2-d quadrilateral elements with linear shape function and full integration
        
        Parameters
        ----------
        model  : object of class ``Model``
            Refers to parent class ``Mode`` to inherit all attributes
        nodes  : list 
            List of nodes belonging to element
        lx     : float
            Element size in x-direction
        ly     : float
            Element size in y-direction
        sect   : int
            Section in which element is located
        mat: object of class ``Material``
            Material card with attributes and methods to be applied in element
                
        Attributes
        ----------
        Model : object of class ``Model``
            Refers to parent object (FE model)
        nodes : list 
            List of nodes of this element
        Lelx  : float
            Size of element in x-direction
        Lely  : float
            Size of element in y-direction
        ngp   : int
            number of Gauss points
        gpx   : 1-d array
            x-locations of Gauss points in element
        gpy   : 1-d array
            y-locations of Gauss points in element
        Bmat  : list
            List of B-matrices at Gauss points
        Vel   : float
            Eolume of element
        Kel   : 2-d array
            Element stiffness matrix
        Jac   : float
            Determinant of Jacobian of iso-parametric formulation
        wght  : 1-d array
            List of weight of each Gauss point in integration
        Sect  : int
            Section of model in which element is located
        Mat   : object of class ``Material``
            Material model to be applied
        eps   : Voigt tensor
            average (total) element strain (defined in ``Model.solve``)
        sig   : Voigt tensor
            average element stress (defined in ``Model.solve``)
        epl   : Voigt tensor
            average plastic element strain (defined in ``Model.solve``)
        CV    : 2d array
            Voigt stiffness matrix of element material
        '''
        'Methods'
        #calc_Bmat: Calculate B matrix at Gauss point
        #calc_quant: calculate element stress, total and plastic strain

        def __init__(self, model, nodes, lx, ly, sect, mat):
            self.Model = model
            self.nodes = nodes
            self.Lelx = lx
            self.Lely = ly
            self.Sect = sect
            self.Mat = mat
            DIM = model.dim
            C11 = mat.C11
            C12 = mat.C12
            C44 = mat.C44 
            #Calculate Voigt stiffness matrix for plane stress and plane strain'
            if (model.planestress):
                hh = mat.E/(1-mat.nu*mat.nu)
                C12 = mat.nu*hh
                C11 = hh
                self.CV = np.array([[C11, C12, 0., 0.,  0.,  0.], \
                   [C12, C11, 0., 0.,  0.,  0.], \
                   [0., 0., 0., 0.,  0.,  0.], \
                   [0.,  0.,  0.,  0., 0.,  0.], \
                   [0.,  0.,  0.,  0.,  0., 0.], \
                   [0.,  0.,  0.,  0.,  0.,  C44]])
            else:
                self.CV = np.array([[C11, C12, C12, 0.,  0.,  0.], \
                   [C12, C11, C12, 0.,  0.,  0.], \
                   [C12, C12, C11, 0.,  0.,  0.], \
                   [0.,  0.,  0.,  C44, 0.,  0.], \
                   [0.,  0.,  0.,  0.,  C44, 0.], \
                   [0.,  0.,  0.,  0.,  0.,  C44]])
            self.elstiff = self.CV
            
            #initialize element quantities
            self.eps = np.zeros(6)
            self.sig = np.zeros(6)
            self.epl = np.zeros(6)
            
            #Calculate element attributes
            self.Vel = lx*ly*model.thick        # element volume
            self.ngp = model.shapefact*DIM**2  # number of Gauss points in element
            self.gpx = np.zeros(self.ngp)
            self.gpy = np.zeros(self.ngp)
            self.Bmat = [None]*self.ngp
            self.wght = 1.  # weight factor for Gauss integration, to be adapted to element type
            self.Jac = self.Vel   # determinant of Jacobian matrix, to be adapted to element type
            
            #Calculate B matrix at all Gauss points
            if (model.shapefact==1):
                if (DIM==1):
                    # integration trivial for 1-d element with linear shape function
                    # because B is constant for all positions in element
                    self.Bmat[0] = self.calc_Bmat()
                elif (DIM==2):
                    # integration for 2-d quadriliteral elements with lin shape fct
                    # reduced integration with one Gauss point in middle of element
                    cpos = np.sqrt(1./3.)   # position of Gauss points
                    self.Jac *= 4.
                    for i in range(self.ngp):
                        sx = (-1)**int(i/2)
                        sy = (-1)**i
                        x = 0.5*(1.+sx*cpos)*self.Lelx
                        y = 0.5*(1.+sy*cpos)*self.Lely
                        self.gpx[i] = x
                        self.gpy[i] = y
                        self.Bmat[i] = self.calc_Bmat(x=x, y=y)
            elif (model.shapefact==2):
                if (DIM==1):
                    # integration for 1-d element with quadratic shape function
                    # Gauss integration
                    cpos = np.sqrt(1./3.)   # position of Gauss points
                    self.wght = 0.5
                    for i in range(self.ngp):
                        sx = (-1)**i
                        x = 0.5*self.Lelx*(1. - sx*cpos)  # positions of Gauss points
                        self.gpx[i] = x
                        self.Bmat[i] = self.calc_Bmat(x=x)
                elif (DIM==2):
                    sys.exit('Error: Quadrilateral elements with quadratic shape function not yet implemented')
            self.calc_Kel()
            
        def calc_Kel(self):
            'Calculate element stiffness matrix by Gaussian integration'
            K0 = [(np.transpose(B) @ self.elstiff @ B) for B in self.Bmat]
            self.Kel = self.Jac*self.wght*sum(K0)
        
        def node_num(self):
            '''Calculate indices of DOF associated with element
            
            Returns
            -------
            ind : list of int
                List of indices
            '''
            ind = []
            for j in self.nodes:
                ind.append(j*self.Model.dim)
                if (self.Model.dim==2):
                    ind.append(j*self.Model.dim + 1)
            return ind
        
        def deps(self):
            '''Calculate strain increment in element
            
            Returns
            -------
            deps : Voigt tensor
                Strain increment
            '''
            deps = 0.
            for B in self.Bmat:
                deps += self.wght*B @ self.Model.du[self.node_num()]
            return deps

        def eps_t(self):
            '''Calculate total strain in element
            
            Returns
            -------
            eps_t : Voigt tensor
                Total strain
            '''
            eps_t = 0.
            for B in self.Bmat:
                eps_t += self.wght*B @ self.Model.u[self.node_num()]
            return eps_t
        
        def dsig(self): 
            '''Calculate stress increment
            
            Returns
            -------
            dsig : Voigt tensor
                Stress increment
            '''
            dsig = self.elstiff @ self.deps()
            return dsig
        
        def depl(self):
            '''Calculate plastic strain increment
            
            Returns
            -------
            depl : Voigt tensor
                Plastic strain increment
            '''
            depl = 0.
            if (self.Mat.sy!=None):
                deps = self.deps()
                sig  = self.sig + self.CV @ deps
                depl = self.Mat.epl_dot(sig, self.epl, self.CV, deps)
            return depl
        
        def calc_Bmat(self, x=0., y=0.):
            '''Calculate B matrix at position x in element
            
            Parameters
            ----------
            x : float 
                absolute x-position in element (optional, default: 0)
            y : float
                absolute y-position in element (optional, default: 0)
                
            Returns
            -------
            B : 6xN array
                matrix (N=dim^2*(SF+1))
            '''
            # (B_aj = L_ak N_kj k=1,...,dim a=1,...,6 j=DOF_el)
            # 1-d, linear: 
            # N11 = 1 - x/Lelx N12 = x/Lelx
            # L11 = d/dx
            # 1-d, quadratic:
            # N11 = 1 - 3x/Lelx+2x^2/Lelx^2
            # N12 = 4x/Lelx*(1-x/Lelx)
            # N13 = -x/Lelx(1-2x/Lelx)
            # L11 = d/dx
            # all other N_ij, L_ij are zero for 1-dim case
            # see documentation for 2-d element

            DIM = self.Model.dim
            N = DIM*DIM*(self.Model.shapefact+1)
            B = np.zeros((6,N)) 
            if (self.Model.shapefact==1):
                # linear shape function
                if (DIM==1):
                    hx = 1./self.Lelx
                    B[0,0] = -hx
                    B[0,1] =  hx
                if (DIM==2):
                    xi1 = 2.*x/self.Lelx - 1.  # scaled x-position within element, xi1 in [-1,1]
                    xi2 = 2.*y/self.Lely - 1.  # scaled y-position
                    hxm = 0.125*(1.-xi1)/self.Lely
                    hym = 0.125*(1.-xi2)/self.Lelx
                    hxp = 0.125*(1.+xi1)/self.Lely
                    hyp = 0.125*(1.+xi2)/self.Lelx
                    B[0,0] = -hym
                    B[0,2] = -hyp
                    B[0,4] =  hym
                    B[0,6] =  hyp
                    B[1,1] = -hxm
                    B[1,3] =  hxm
                    B[1,5] = -hxp
                    B[1,7] =  hxp
                    B[5,0] = -hxm
                    B[5,1] = -hym
                    B[5,2] =  hxm
                    B[5,3] = -hyp
                    B[5,4] = -hxp
                    B[5,5] =  hym
                    B[5,6] =  hxp
                    B[5,7] =  hyp
                    if (self.Model.planestress):
                        # eps3 = -nu (sig1 + sig2)/E
                        hh = self.CV @ B   # sig_(alpha,j)
                        B[2,:] = -self.Mat.nu*(hh[0,:] + hh[1,:])/self.Mat.E
            elif (self.Model.shapefact==2):
                h1 = 1./self.Lelx
                h2 = 4./(self.Lelx*self.Lelx)
                if (DIM==1):
                    B[0,0] = h2*x  - 3.*h1
                    B[0,1] = 4.*h1 - 2.*h2*x
                    B[0,2] = h2*x  - h1
                if (DIM==2):
                    sys.exit('Error: Quadratic shape functions for 2D elements not yet implemented')
            return B
        
    def geom(self, sect, LY=1., LZ=1.):
        '''Specify geometry of FE model and its subdivision into sections;
        for 2-d model a laminate structure normal to x-direction is created; adds attributes to class ``Model``
        
        Parameters
        ----------
        sect  : list 
            List with with absolute length of each section
        LY : float
            Length in y direction (optional, default: 1)
        LZ : float
            Thickness of model in z-direction (optional, default: 1)
        '''
        self.Nsec = len(sect)  # number of sections
        self.LS = np.array(sect)
        self.lenx = sum(sect)  # total length of model
        self.leny = LY
        self.thick = LZ
        
    def assign(self, mats):
        '''Assign each section to an object of class ``Material``
        
        Parameters
        ----------
        mats : list 
            List of materials, must have same dimensions as Model.LS
            
        Attributes
        ----------
        
        '''
        self.mat = mats
        for i in range(len(mats)):
            if mats[i].sy!=None:
                self.nonlin = True   # nonlinear model if at least one material is plastic

    'subroutines to define boundary conditions, top/bottom only needed for 2-d models'
    def bcleft(self, u0):
        '''Define boundary conditions on lhs nodes, always displacement type
        
        Parameters
        ----------
        u0   : float
            Displacement in x direction of lhs nodes
        '''
        self.uleft = u0
        
    def bcright(self, h1, bctype):
        '''Define boundary conditions on rhs nodes, either force or displacement type
        
        Parameters
        ----------
        h0     : float
            Displacement or force in x direction 
        bctype : str
            Type of boundary condition ('disp' or 'force')
        '''
        self.bcr  = h1
        self.namebcr = bctype
        if (bctype=='disp'):
            'type of boundary conditions (BC)'
            self.ubcright = True   # True: displacement BC on rhs node
        elif (bctype=='force'):
            self.ubcright = False   # False: force BC on rhs node
        else:
            print('Unknown BC:', bctype)
            sys.exit()
        
    def bcbot(self, u0):
        '''Define boundary conditions on bottom nodes, always displacement type
        
        Parameters
        ----------
        u0  : float
            Displacement in x direction 
        '''
        self.ubot = u0
        
    def bctop(self, h1, bctype):
        '''Define boundary conditions on top nodes, either force or displacement type
        
        Parameters
        ----------
        h0     : float
            Displacement or force in x direction 
        bctype : str
            Type of boundary condition ('disp' or 'force')
        '''
        self.bct  = h1
        self.namebct = bctype
        if (bctype=='disp'):
            'type of boundary conditions (BC)'
            self.ubctop = True   # True: displacement BC on rhs node
        elif (bctype=='force'):
            self.ubctop = False   # False: force BC on rhs node
        else:
            print('Error: Unknown BC:', bctype)
            sys.exit()
            
    def mesh(self, NX=10, NY=1, SF=1):
        '''Generate mesh based on nodal positions model.npos, depending on degree of shape function; 
        List of elements is initialized with method Model.Element
        
        Parameters
        ----------
        NX : int
            Number of elements in x-direction (optional, default: 10)
        NY : int
            Number of elements in y-direction (optional, default: 1)
        SF : int
            Degree of shape functions: 1=linear, 2=quadratic (optional, default: 1)
        '''
        self.shapefact = SF
        DIM = self.dim
        if (NX < self.Nsec):
            sys.exit('Error: Number of elements is smaller than number of sections')
        if ((NY>1)and(DIM==1)):
            NY = 1
            print('Warning: NY=1 for 1-d model')
        if self.u is not None:
            #print('Warning: solution of previous solution steps is deleted')
            self.u = None
            self.f = None
        self.NnodeX = self.shapefact*NX + 1    # number of nodes along x axis
        self.NnodeY = (DIM-1)*self.shapefact*NY + 1    # number of nodes along y axis
        self.Nel = NX*NY
        self.Nnode = self.NnodeX*self.NnodeY   # total number of nodes
        self.Ndof  = self.Nnode*DIM  # degrees of freedom
        self.npos = np.zeros(self.Ndof)  # position array of nodes
        self.element = [None] * self.Nel  # empty list for elements
        self.noleft  = []  # list of nodes on left boundary
        self.noright = []  # list of nodes on right boundary
        self.nobot   = []  # list of nodes on bottom boundary
        self.notop   = []  # list of nodes on top boundary
        self.noinner = []  # list of inner nodes
        
        'Calculate number of elements per section -- only laminate structure'
        hh  = self.LS / self.lenx      # proportion of segment length to total length of model
        nes = [int(x) for x in np.round(hh*NX)]  # nes gives number of elements per segement in proportion 
        if (np.sum(nes) != NX):  # add or remove elements of largest section if necessary
            im = np.argmax(self.LS)
            nes[im] = nes[im] - np.sum(nes) + NX

        'Define nodal positions and element shapes -- only for laminate structure'
        jstart = 0
        nrow = self.NnodeY
        dy = self.leny / NY
        for i in range(self.Nsec):
            'define nodal positions first'
            ncol = nes[i]*self.shapefact + 1
            dx = self.LS[i] / nes[i]
            nr = np.max([1, nrow-1])
            elstart = np.sum(nes[0:i],dtype=int)*nr
            n1 = (int(elstart/NY)*nrow + int(np.mod(elstart,NY)))*self.shapefact
            for j in range(jstart, ncol):
                for k in range(nrow):
                    inode = j*nrow + k + n1
                    self.npos[inode*DIM] = (j+int(elstart/NY))*dx # x-position of node
                    if (DIM==2):
                        self.npos[inode*DIM+1] = k*dy # y-position of node
                    nin = True
                    if (j==0): 
                        self.noleft.append(inode)
                        nin = False
                    if (k==0):
                        self.nobot.append(inode)
                        nin = False
                    if (k==nrow-1):
                        self.notop.append(inode)
                        nin = False
                    if ((i==self.Nsec-1)and(j==ncol-1)):
                        self.noright.append(inode)
                        nin = False
                    if nin:
                        self.noinner.append(inode)
            'initialize elements'
            for j in range(nes[i]*nr):
                ih = elstart + j   # index of current element
                n1 = (int(ih/NY)*nrow + int(np.mod(ih,NY)))*self.shapefact
                n2 = n1 + self.shapefact
                n3 = n1 + nrow*self.shapefact
                n4 = n3 + self.shapefact
                if (self.shapefact*DIM==1):
                    self.element[ih] = self.Element(self, [n1, n2], dx, dy, i, self.mat[i])  # 1-d, lin shape fct
                elif (self.shapefact*DIM==4):
                    nh = n1 + nrow + 1
                    hh = [n1, n1+1, n2, nh, nh+1, n3, n3+1, n4] # 2-d, quad shape fct
                    self.element[ih] = self.Element(self, hh, dx, dy, i, self.mat[i])
                elif (DIM==2):
                    hh = [n1, n2, n3, n4] # 2-d, lin shape fct
                    self.element[ih] = self.Element(self, hh, dx, dy, i, self.mat[i])
                else:
                    hh = [n1, n1+1, n2] # 1-d, lin shape fct
                    self.element[ih] = self.Element(self, hh, dx, dy, i, self.mat[i])
            jstart = 1

    def setupK(self):
        '''Calculate and assemble system stiffness matrix based on element stiffness matrices
        
        Returns
        -------
        K  : 2d-array
            System stiffness matrix
        '''
        DIM = self.dim
        K = np.zeros((self.Ndof, self.Ndof))  # initialize system stiffness matrix
        for el in self.element:
            'assemble element stiffness matrix into system stiffness matrix'
            for j in range(len(el.nodes)):
                j1 = el.nodes[j]*DIM      # position of ux (1st dof) of left node in u vector
                j2 = j1 + DIM
                je1 = j*DIM
                je2 = je1 + DIM
                for k in range(len(el.nodes)):
                    k1 = el.nodes[k]*DIM
                    k2 = k1 + DIM
                    ke1 = k*DIM
                    ke2 = ke1 + DIM
                    K[j1:j2,k1:k2] += el.Kel[je1:je2,ke1:ke2]
        return K
     
    def solve(self, min_step=None, verb=False):
        '''Solve linear system of equations f = K^-1 . u for mechanical equiliibrium;
        total force on internal nodes is zero;
        stores solution in u, f, element attributes
        
        Parameters
        ----------
        min_step : int
            Minimum number of load steps (optional)
        verb     : Boolean
            Be verbose in text output (optional, default: False)
        '''
        'calculate reduced stiffness matrix according to BC'
        def calc_Kred(ind):
            Kred = np.zeros((len(ind), len(ind)))
            for i in range(len(ind)):
                for j in range(len(ind)):
                    Kred[i,j] = K[ind[i], ind[j]]
            return Kred
        
        'test if stress has converged to yield surface'
        def test_convergence():
            conv = True
            f = []
            # test if yield criterion is exceeded
            for el in self.element:
                if (el.Mat.sy!=None):  # not necessary for elastic material
                    peeq = eps_eq(el.epl+el.depl()) # calculate equiv. plastic strain
                    sig = el.sig+el.dsig()
                    hh = el.Mat.calc_yf(sig, peeq=peeq)
                    f.append(hh)
                    if hh > 1.e-3:
                        el.elstiff = el.Mat.C_tan(sig, el.CV)
                        el.calc_Kel()
                        conv = False
            return conv, f
        
        'calculate scaling factor for load steps'
        def calc_scf(sglob):
            sc_list = [1.]
            for el in self.element:
                # test if yield criterion is exceeded
                if (el.Mat.sy!=None):  # not necessary for elastic material
                    yf0 = el.Mat.calc_yf(el.sig, peeq=0.)  # yield fct. at stat of load step
                    'if element starts in elestic regime load step can only touch yield surface'
                    if  yf0 < -0.15:
                        sref = el.Mat.calc_seq(sglob)   # global equiv. stress at max load step
                        if el.Mat.ML_yf:
                            # for ML categorical yield function get better approximation for predictor step
                            seq = Stress(el.sig).seq(el.Mat)
                            hs = np.zeros(3)
                            if np.abs(max_dbcr)>1.e-6:
                                hs[0] = el.Mat.sy*np.sign(max_dbcr) # construct stress vector for search of yield point
                            if np.abs(max_dbct)>1.e-6:
                                hs[1] = el.Mat.sy*np.sign(max_dbct)
                            x0 = 1.
                            if max_dbcr*max_dbct<0:
                                x0 = 0.5
                            x1, infodict, ier, msg = fsolve(el.Mat.find_yloc, x0, args=hs, xtol=1.e-5, 
                                                                full_output=True)
                            y1 = infodict['fvec']
                            if np.abs(y1)<1.e-5 and x1<3.:
                                # zero of ML yield fct. detected at x1*sy
                                yf0 = seq - x1[0]*el.Mat.calc_seq(hs)
                            else:
                                # zero of ML yield fct. not found: get conservative estimate
                                yf0 = seq - 0.85*el.Mat.sy
                                if (verb):
                                    print('Warning in calc_scf')
                                    print('*** detection not successful. yf0=', yf0,', seq=', seq)
                                    print('*** optimization result (x1,y1,ier,msg):', x1, y1, ier, msg)
                            if yf0 > -0.15:
                                yf0 = -sref
                        sc_list.append(-yf0/sref)
            # select scaling appropriate scaling such that no element crosses yield surface
            scf = np.amin(sc_list)
            if scf<1.e-5:
                if verb:
                    print('Warning: Small load increment in calc_scf: ',scf, sc_list)
                scf = 1.e-5
            return scf
            
        'define BC: modify stiffness matrix for displacement BC, calculate consistent force BC'
        def calc_BC(K, ul, ub, dbcr, dbct):
            # BC on lhs nodes is always x-displacement 
            # apply BC by adding known boundary forces to solution vector
            # to reduce rank of system of equations by 1 (one row eliminated)
            # Paramaters:
            # K : stiffness matrix
            # ul, ub : fixed displacement on lhs and bottom nodes
            # dbcr, dbct : increment of BC on rhs and top noes
            du = np.zeros(self.Ndof)
            df = np.zeros(self.Ndof)
            ind = list(range(self.Ndof))  # list of all nodal DOF, will be shortened according to BC
            for j in self.noleft:
                i = j*self.dim   # postion of x-values of node #j in u/f vector
                ind.remove(i)
                du[i] = ul
                df[ind] -= K[ind,i]*ul
            
            if (self.dim==2):
                # BC on bottom nodes is always y-displacement 
                # apply BC by adding known boundary forces to solution vector
                # to reduce rank of system of equations by 1 (one row eliminated)
                for j in self.nobot:
                    i = j*self.dim + 1   # postion of y-values of node #j in u/f vector
                    ind.remove(i)
                    du[i] = ub
                    df[ind] -= K[ind,i]*ub

            # rhs node BC can be force or displacement
            # apply BC and solve corresponding system of equations
            if (self.ubcright):
                # displacement BC
                # add known boundary force to solution vector to
                # eliminate row Ndof from system of equations
                for j in self.noright:
                    i = j*self.dim
                    ind.remove(i)
                    hh = list(range(self.Ndof))
                    hh.remove(i)
                    du[i] = dbcr
                    df[hh] -= K[i, hh]*dbcr
            else:
                # force bc on rhs nodes
                for j in self.noright:
                    i = j*self.dim
                    hh=1.
                    hy = self.npos[i+1] #y-position of node
                    if hy<1.e-3 or hy>self.leny-1.e-3:
                        hh=0.5  # reduce force on corner nodes
                    df[i] += dbcr*hh
                
            # BC on top nodes can be force or displacement
            # apply BC and solve corresponding system of equations
            if (self.dim==2):
                if (self.ubctop):
                    # displacement BC
                    # add known boundary force to solution vector to
                    # eliminate row Ndof from system of equations
                    for j in self.notop:
                        i = j*self.dim + 1
                        ind.remove(i)
                        du[i] = dbct
                        df[ind] -= K[ind,i]*dbct
                else:
                    # force bc on top nodes
                    for j in self.notop:
                        i = j*self.dim + 1
                        hh=1.
                        hx = self.npos[i-1] #x-position of node
                        if hx<1.e-3 or hx>self.lenx-1.e-3:
                            hh=0.5  # reduce force on corner nodes
                        df[i] += dbct*hh
            return du, df, ind
            
        # test if all necessary BC have been set
        if self.uleft is None:
            self.uleft = 0.
            print('Warning: BC on lhs nodes has been set to 0')
        if self.bcr is None:
            self.bcr = 0.
            self.ubrtop = False
            self.namebcr = 'force'
        if self.dim>1:
            if (self.ubot==None):
                self.ubot = 0.
                print('Warning: BC on bottom nodes has been set to 0')
            if self.bct is None:
                self.bct = 0.
                self.ubctop = False
                self.namebct = 'force'
                
        K = self.setupK()  # assemble system stiffness matrix from element stiffness matrices
        jin = []
        for j in self.noinner:
            jin.append(j*self.dim)
            jin.append(j*self.dim+1)
                
        if self.u is None:
            # declare and initialize solution vectors and boundary conditions
            self.u = np.zeros(self.Ndof)
            self.f = np.zeros(self.Ndof)
            'initialize element quantities'
            for el in self.element:
                el.elstiff = el.CV
                el.eps = np.zeros(6)
                el.sig = np.zeros(6)
                el.epl = np.zeros(6)
            bcr0 = 0.
            bct0 = 0.
        else:
            bcr0 = self.bcr_mem
            bct0 = self.bct_mem
        ul = self.uleft
        ub = self.ubot
        
        'define loop for external load steps (BC subdivision)'
        'during each load step mechanical equilibrium is calculated for sub-step'
        'the tangent stiffness matrix of the last load step is used as initial guess'
        'current tangent stiffness matrix compatible with BC is determined iteratively'
        il = 0
        bc_inc = True
        while bc_inc:
            'define increments for boundary conditions'
            if min_step is None:
                if self.dim==1:
                    max_dbct = None
                else:
                    max_dbct = self.bct - bct0
                max_dbcr = self.bcr - bcr0
            else:
                sc = np.maximum(1, min_step-il)
                if self.dim==1:
                    max_dbct = None
                else:
                    max_dbct = (self.bct-bct0)/sc
                max_dbcr = (self.bcr-bcr0)/sc
            'calculate du and df fulfilling for max. load step consistent with stiffness matrix K'
            dbcr = max_dbcr
            dbct = max_dbct
            self.du, self.df, ind = calc_BC(K, ul, ub, dbcr, dbct) # consider BC for system of equ.  
            Kred = calc_Kred(ind)  # setup K matrix for reduced system acc. to BC
            self.du[ind] = np.linalg.solve(Kred, self.df[ind]) # Solve reduced system of equations
            self.u += self.du
            self.f += K@self.du
            'calculate scaling factor for predictor step in case of non-linear model'
            if self.nonlin:
                self.calc_global()  # calculate global values for solution
                self.u -= self.du
                self.f -= self.df
                sglob = np.array([self.glob['sbc1'], self.glob['sbc2'], 0.])
                scale_bc = calc_scf(sglob) # calculate predictor step to hit the yield surface
                if verb:
                    print('##scaling factor',scale_bc)
                dbcr = max_dbcr*scale_bc  # dbcr >= self.bcr - bcr0
                dbct = max_dbct*scale_bc  # dbct <= self.bct - bct0
                'calculate du and df fulfilling for scaled load step consistent with stiffness matrix K'
                self.du, self.df, ind = calc_BC(K, ul, ub, dbcr, dbct) # consider BC for system of equ.  
                #Kred = calc_Kred(ind)  # setup K matrix for reduced system acc. to BC
                self.du[ind] = np.linalg.solve(Kred, self.df[ind]) # Solve reduced system of equations
                conv, f = test_convergence() # false if yield function >1.e-6; modify elem. stiffness
                i = 0
                if verb:
                    print('**Load step #',il)
                    fres = K@(self.u+self.du)
                    print('yield function=',f,'residual forces on inner nodes=',fres[jin])
                while not conv: 
                    # calculate tangent stiffness matrix
                    K = self.setupK()  #setup BC anew if K changes
                    dbcr -= i*0.1*dbcr
                    dbct -= i*0.1*dbct
                    self.du, self.df, ind = calc_BC(K, ul, ub, dbcr, dbct)
                    Kred = calc_Kred(ind)
                    self.du[ind] = np.linalg.solve(Kred, self.df[ind]) # solve du with current stiffness matrix
                    conv, f = test_convergence()
                    i += 1
                    if verb:
                        print('+++Inner load step #',i)
                        fres = K@(self.u+self.du)
                        print('yield function=',f,'residual forces on inner nodes=',fres[jin])
                    if i>10:
                        print('Warning: No convergence achieved, abandoning')
                        conv = True
                self.u += self.du
                self.f += K@self.du
            'update internal variables with results of load step'
            for el in self.element:
                el.eps = el.eps_t()
                el.epl += el.depl()
                el.sig += el.dsig()
            'update load step'
            il += 1
            bcr0 += dbcr
            hl = np.abs(bcr0-self.bcr)>1.e-6 and np.abs(dbcr)>1.e-9
            if self.dim>1:
                bct0 += dbct
                hr = np.abs(bct0-self.bct)>1.e-6 and np.abs(dbct)>1.e-9
            else:
                hr = False
            bc_inc = hr or hl
            'store time dependent quantities'
            self.calc_global()  # calculate global values for solution
            self.sgl  = np.append(self.sgl, [self.glob['sig']], axis=0)
            self.egl  = np.append(self.egl, [self.glob['eps']], axis=0)
            self.epgl = np.append(self.epgl,[self.glob['epl']], axis=0)
            if verb:
                fres = K@self.du  # residual forces 
                fnorm = np.linalg.norm(fres[jin],1)  # norm of residual nodal forces
                print('Load increment ', il, 'total',self.namebct,'top ',bct0,'/',self.bct,'; last step ',dbct)
                print('Load increment ', il, 'total',self.namebcr,'rhs',bcr0,'/',self.bcr,'; last step ',dbcr)
                print('Yield function after ',i,'steps: f=',f, '; norm(f_node)=',fnorm)
                #print('BC strain: ', np.around([self.glob['ebc1'],self.glob['ebc2']],decimals=5))
                #print('BC stress: ', np.around([self.glob['sbc1'],self.glob['sbc2']],decimals=5))
                print('Global strain: ', np.around(self.glob['eps'],decimals=5))
                print('Global stress: ', np.around(self.glob['sig'],decimals=5))
                print('Global plastic strain: ', np.around(self.glob['epl'],decimals=5))
                print('----------------------------')
        self.bct_mem = bct0
        self.bcr_mem = bcr0

    def bcval(self, nodes):
        '''Calculate average displacement and total force at (boundary) nodes
        
        Parameters
        ----------
        nodes : list
            List of nodes
        '''
        hux = 0.
        huy = 0.
        hfx = 0.
        hfy = 0.
        n = len(nodes)
        for i in nodes:
            hux += self.u[i*self.dim]
            hfx += self.f[i*self.dim]
            if (self.dim==2):
                huy += self.u[i*self.dim+1]
                hfy += self.f[i*self.dim+1]
        return hux/n, huy/n, hfx, hfy
    
    def calc_global(self):
        '''Calculate global quantities and store in Model.glob;
        homogenization done by averaging residual forces (sbc1/2) and displacements (ebc1/2) at boundary nodes 
        or by averaging element quantities (sig, eps, epl)
        '''
        'calculate global values from BC'
        uxl, uyl, fxl, fyl = self.bcval(self.noleft)
        uxr, uyr, fxr, fyr = self.bcval(self.noright)
        self.glob['ebc1'] = (uxr-uxl)/self.lenx
        self.glob['sbc1'] = 0.5*(fxr-fxl)/(self.leny*self.thick)
        if (self.dim==2):
            uxb, uyb, fxb, fyb = self.bcval(self.nobot)
            uxt, uyt, fxt, fyt = self.bcval(self.notop)
            self.glob['ebc2'] = (uyt-uyb)/self.leny
            self.glob['sbc2'] = 0.5*(fyt-fyb)/(self.lenx*self.thick)
        'calculate global values from element solutions'
        sig = np.zeros(6)
        eps = np.zeros(6)
        epl = np.zeros(6)
        for el in self.element:
            sig += el.sig*el.Vel
            eps += el.eps*el.Vel
            epl += el.epl*el.Vel
        Vm = self.lenx*self.leny*self.thick   # Volume of model
        self.glob['sig'] = sig/Vm
        self.glob['eps'] = eps/Vm
        self.glob['epl'] = epl/Vm

    def plot(self, fsel, mag=10, colormap='viridis', cdepth=20, showmesh=True, shownodes=True):
        '''Produce graphical output: draw elements in deformed shape with color 
        according to field variable 'fsel'; uses matplotlib
        
        Parameters
        ----------
        fsel   : str
            Field selector for library field
        mag    : float
            Magnification factor for displacements (optional, default: 10)
        cdepth : int
            Number of colors in colormap (optional, default: 20)
        showmesh : Boolean
            Set/unset plotting of lines for element edges (optional, default: True)
        shownodes: Boolean
            Set/unset plotting of nodes (optional, default: True)
        colormap : str
            Name of colormap to be used (optional, default: viridis)
        '''
        fig, ax = plt.subplots(1)
        cmap = mpl.cm.get_cmap(colormap, cdepth)
        def strain1():
            hh = [el.eps[0]*100 for el in self.element]
            text_cb = 'strain_11 (%)'
            return hh, text_cb
        def stress1():
            hh = [el.sig[0] for el in self.element]
            text_cb = 'stress_11 (MPa)'
            return hh, text_cb
        def strain2():
            hh = [el.eps[1]*100 for el in self.element]
            text_cb = 'strain_22 (%)'
            return hh, text_cb
        def stress2():
            hh = [el.sig[1] for el in self.element]
            text_cb = 'stress_22 (MPa)'
            return hh, text_cb
        def stress_eq():
            hh = [Stress(el.sig).seq(el.Mat) for el in self.element]
            text_cb = 'eqiv. stress (MPa)'
            return hh, text_cb
        def strain_peeq():
            hh = [eps_eq(el.epl)*100 for el in self.element]
            text_cb = 'eqiv. plastic strain (%)'
            return hh, text_cb
        def strain_etot():
            hh = [eps_eq(el.eps)*100 for el in self.element]
            text_cb = "eqiv. total strain (%)"
            return hh, text_cb
        def disp_x():
            hh = [el.eps[0]*self.lenx for el in self.element]
            text_cb = "u_x (mm)"
            return hh, text_cb
        def disp_y():
            hh = [el.eps[1]*self.leny for el in self.element]
            text_cb = "u_y (mm)"
            return hh, text_cb
        field={
            'strain1' : strain1(),
            'stress1' : stress1(),
            'strain2' : strain2(),
            'stress2' : stress2(),
            'seq'     : stress_eq(),
            'peeq'    : strain_peeq(),
            'etot'    : strain_etot(),
            'ux'      : disp_x(),
            'uy'      : disp_y()
        }
        
        'define color value by mapping field value of element to interval [0,1]'
        val, text_cb = field[fsel]
        vmin = np.min(val)
        vmax = np.max(val)
        delta = np.abs(vmax - vmin)
        if delta < 0.1 or delta/vmax < 0.04:
            if np.abs(vmax) < 0.1:
                vmax += 0.05
                vmin -= 0.05
            elif vmax>0.:
                vmax *= 1.02
                vmin *= 0.98
            else:
                vmax *= 0.98
                vmin *= 1.02
            delta = np.abs(vmax - vmin)
        col = np.round((val-vmin)/delta, decimals=5)
        
        'create element plots'
        for el in self.element:
            # draw filled polygon for each element
            if (self.dim==1):
                ih = np.min(el.nodes)       # left node of current element
                jh = np.max(el.nodes)    # right node of current element
                ih1 = ih*self.dim            # position of ux in u vector
                jh1 = jh*self.dim            # position of ux in u vector
                hx1 = self.npos[ih] + mag*self.u[ih1] # x position of left node 
                hx2 = self.npos[jh] + mag*self.u[jh1] # x position of right node
                hh = self.thick*0.5
                hx = [hx1, hx2, hx2, hx1]
                hy = [-hh, -hh, hh, hh]
            else:
                hx = [0., 0., 0., 0.]
                hy = [0., 0., 0., 0.]
                k = [0, 3, 1, 2]
                p = 0
                for ih in el.nodes:
                    j = ih*self.dim
                    hx[k[p]] = self.npos[j] + mag*self.u[j]
                    hy[k[p]] = self.npos[j+1] + mag*self.u[j+1]
                    p += 1
            ax.fill(hx, hy, color=cmap(col[self.element.index(el)]))
            if (showmesh):
                hx.append(hx[0])
                hy.append(hy[0])
                ax.plot(hx, hy, 'k', lw=1)  # plot edges of elements

        'plot nodes'
        if (shownodes):
            hh = self.npos + mag*self.u
            if (self.dim==1):
                hx = hh
                hy = np.zeros(self.Ndof)
            else:
                hx = hh[0:self.Ndof:2]
                hy = hh[1:self.Ndof:2]
            ax.scatter(hx, hy, s=50, c='red', marker='o', zorder=3)
 
        'add colorbar'
        axl = fig.add_axes([1.01, 0.15, 0.04, 0.7])  #[left, bottom, width, height]
        # for use in juypter note book: left = 1.01, for python: left = 0.86
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm,
                               orientation='vertical')
        cb1.set_label(text_cb)
        plt.show()
