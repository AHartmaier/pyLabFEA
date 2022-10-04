#Module pylabfea.mod
'''Module pylabefea.model introduces global functions for mechanical quantities
and class ``Model`` that contains that attributes and methods needed in FEA. 
Materials are defined in module pylabfea.material

uses NumPy, SciPy, MatPlotLib

Version: 4.1 (2022-02-22)
Author: Alexander Hartmaier, ICAMS/Ruhr University Bochum, Germany
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)'''
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pylabfea.basic import Stress, eps_eq, sig_eq_j2, yf_tolerance
from matplotlib import colors, colorbar

# =========================   
# define class for FE model
# =========================

class Model(object):
    '''Class for finite element model. Defines necessary attributes and methods
    for pre-processing (defining geometry, material assignments, mesh and 
    boundary conditions);
    solution (invokes numerical solver for non-linear problems);
    and post-processing (visualization of results on meshed geometry, 
    homogenization of results into global quantities).
    
    *Geometry and sections* can be defined with the methods ``geom`` and ``assign``. 
    There are pre-defined options that make the generation of laminate structures 
    particularly easy. But also more general section can be defined and each section 
    can be associated with a different material.
    
    *Boundary conditions* on left-hand-side and bottom nodes are always assumed
    to be static (typically with values of zero);
    Boundary conditions on right-hand-side and top nodes are considered as load
    steps starting from zero.
    Default boundary conditions are: 
    
      * lhs: fixed in x-direction (ux=0), free in y-direction (fy=0)
      * bot: fixed in y-direction (uy=0), free in x-direction (fx=0)
      * rhs: free (fx=fy=0)
      * top: free (fx=fy=0)
      
    There are pre-defined sets of nodes to top, bottom, left and right boundaries to
    which either force or displacement controlled loads can be applied in x or y-direction
    with the methods ``bctop``, ``bcbot``, ``bcleft``, ``bcright``. Furthermore, it is 
    possible to define boundary conditions for a freely defined set of nodes with the 
    method ``bcnode``. The latter option if useful, to fix only one corner node when 
    laterally free boundaries shall be implemented. It can also be used to simulate 
    loads on parts of boundaries, as it occurs for indentations, etc.
    
    *Visualization* is performed with the method ``plot``, which can display various 
    mechanical quantities on the deformed mesh. Homogenization of boundary loads is
    simply performed with the method ``calc_glob`` by which all global stresses and 
    strains are obtained by averaging over the element values and by summing up the 
    boundary loads for comparison. This is particularly useful when calculating the
    stress-strain behavior of a structure.
    
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
        List of materials assigned to sections of model, same dimensions as LS
        (defined in ``assign``)
    ubcbot: dim-array of Boolean
        True: displacement BC on rhs nodes; False: force BC on rhs nodes
        (defined in ``bcright``)
    bcb  : dim-array
        Nodal displacements in x or y-direction for lhs nodes
        (defined in ``bcbot``)
    ubcleft: dim-array of Boolean
        True: displacement BC on rhs nodes; False: force BC on rhs nodes
        (defined in ``bcright``)
    bcl : dim-array
        Nodal displacement in x or y-direction for lhs nodes
        (defined in ``bcleft``)
    ubcright: dim-array of Boolean
        True: displacement BC on rhs nodes; False: force BC on rhs nodes
        (defined in ``bcright``)
    bcr   : dim-array
        Nodal displacements/forces in x or y-direction for rhs nodes
        (defined in ``bcright``)
    ubctop: Boolean
        True: displacement BC on top nodes; False: force BC on top nodes
        (defined in ``bctop``)
    bct   : dim-array 
        Nodal displacements/forces in x or y-direction for top nodes
        (defined in ``bctop``)
    Nnode : int
        Total number of nodes in Model (defined in ``mesh``)
    NnodeX, NnodeY : int
        Numbers of nodes in x and y-direction (defined in ``mesh``)
    Nel   : int
        Number of elements (defined in ``mesh``)
    Ndof  : int
        Number of degrees of freedom (defined in ``mesh``)
    npos: 1d-array
        Nodal positions, dimension Ndof, same structure as u,f-arrays:
            (x1, y1, x2, y1, ...) (defined in ``mesh``)
    noleft, noright, nobot, notop : list 
        Lists of nodes on boundaries (defined in ``mesh``)
    noinner : list
        List of inner nodes (defined in ``mesh``)
    element : list
        List of objects of class ``Element``, dimension Nel
        (defined in ``mesh``)
    u    : (Ndof,) array
        List of nodal displacements (defined in ``solve``)
    f    : (Ndof,) array
        List of nodal forces (defined in ``solve``)
    sgl  : (N,6) array
        Time evolution of global stress tensor with incremental load steps
        (defined in ``solve``)
    egl  : (N,6) array
        Time evolution of global total strain tensor with incremental load
        steps (defined in ``solve``)
    epgl : (N,6) array
        Time evolution of global plastic strain tensor with incremental load
        steps (defined in ``solve``)
    glob : python dictionary
        Global values homogenized from BC or element solutions, contains the
        elements: 
        'ebc1', 'ebc2', 'sbc1', 'sbc2' : global strain and stress from BC
        (type: float)
        'eps', 'epl', 'sig',  : global strain, plastic strain, and stress 
        tensors homogenized 
        from element solutions (type: Voigt tensor)
        (defined in ``calc_global``)
    '''
    
    def __init__(self, dim=1, planestress = False):
        # set dimensions and type of model
        # initialize boundary conditions
        # dim: dimensional of model (currently only 1 or 2 is possible)
        # planestress: (boolean) plane stress condition
        if ((dim!=1)and(dim!=2)):
            raise ValueError('dim must be either 1 or 2')
        self.dim = dim
        if planestress and (dim!=2):
            warnings.warn('Warning: Plane stress only defined for 2-d model')
            planestress = False
        self.planestress = planestress
        #print('Model initialized')
        self.bcl = np.zeros(dim)
        self.bcb = np.zeros(dim)  
        self.bct = np.zeros(dim)
        self.bcr = np.zeros(dim)
        self.bcn = np.zeros(dim)
        self.noset = None
        self.ubctop = [False, False] # default free boundary on top
        self.ubcright = [False, False] # default free boundary on rhs
        self.ubcleft = [True, False]  # default fixed boundary in x-direction on lhs
        self.ubcbot = [False, True] # default fixed boundary in y-direction on bottom
        self.ubcn = [False, False]  # default free BC for node set
        self.nonlin = False # default is linear elastic model
        self.sgl = np.zeros((1,6))  # list for time evolution of global stresses
        self.egl = np.zeros((1,6))  # list for time evolution of global strains
        self.epgl = np.zeros((1,6)) # list for time evolution of global plastic strain
        self.u = None
        self.f = None
        self.Nnode = None
        #SRM : named module glob exists
        self.glob = {
            'ebc1'   : None,  # global x-strain from BC
            'ebc2'   : None,  # global y-strain from BC
            'sbc1'   : None,  # global x-strain from BC
            'sbc2'   : None,  # global y-strain from BC
            'eps'    : np.zeros(6),  # global strain tensor from element solutions
            'sig'    : np.zeros(6),  # global stress tensor from element solutions
            'epl'    : np.zeros(6)   # global plastic strain tensor from element solutions
        }
    #----------------------
    #Sub-Class for elements
    #----------------------
    class Element(object):
        '''Class for isoparametric elements; supports
        1-d elements with linear or quadratic shape function and full integration;
        2-d quadrilateral elements with linear shape function and full integration
        
        Parameters
        ----------
        model  : object of class ``Model``
            Refers to parent class ``Model`` to inherit all attributes
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
        wght  : 1-d array
            List of weight of each Gauss point in integration
        Vel   : float
            Volume of element
        Jac   : float
            Determinant of Jacobian of iso-parametric formulation
        CV    : (6,6) array
            Voigt stiffness matrix of element material
        elstiff : 2-d array
            Tangent stiffness tensor of element
        Kel   : 2-d array
            Element stiffness matrix
        Mat   : object of class ``Material``
            Material model to be applied
        eps   : Voigt tensor
            average (total) element strain (defined in ``Model.solve``)
        sig   : Voigt tensor
            average element stress (defined in ``Model.solve``)
        epl   : Voigt tensor
            average plastic element strain (defined in ``Model.solve``)
        '''

        def __init__(self, model, nodes, lx, ly, mat):
            self.Model = model
            self.nodes = nodes
            self.Lelx = lx
            self.Lely = ly
            self.Mat = mat
            DIM = model.dim
            #Calculate Voigt stiffness matrix for plane stress and plane strain'
            if mat.CV is None:
                C11 = mat.C11
                C12 = mat.C12
                C44 = mat.C44 
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
            else:
                self.CV = mat.CV
            self.elstiff = self.CV
            
            #initialize element quantities
            self.eps = np.zeros(6)
            self.sig = np.zeros(6)
            self.epl = np.zeros(6)
            
            #initialize quantities for response function
            self.res_sig = None
            self.res_depl = None
            
            #Calculate element attributes
            self.Vel = lx*ly*model.thick        # element volume
            self.ngp = model.shapefact*DIM**2  # number of Gauss points in element
            self.gpx = np.zeros(self.ngp)
            self.gpy = np.zeros(self.ngp)
            self.Bmat = [None]*self.ngp
            self.wght = 1.  # weight factor for Gauss integration, to be adapted to element type
            self.Jac = self.Vel   # determinant of Jacobian matrix, to be adapted to element type
            
            #Initialize dict for convergence stats
            self.stat_nlin = {
                'max_iter'  : 0,
                'max_steps' : 0,
                'max_dstiff': 0.
            }
            #Calculate B matrix at all Gauss points
            if (model.shapefact==1):
                if (DIM==1):
                    # integration trivial for 1-d element with linear shape function
                    # because B is constant for all positions in element
                    self.Bmat[0] = self.calc_Bmat()
                elif (DIM==2):
                    # integration for 2-d quadriliteral elements with lin shape fct
                    # full integration with four Gauss points
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
                    raise NotImplementedError('Error: Quadrilateral elements'+
                         'with quadratic shape function not yet implemented')
            self.calc_Kel()
            
        def calc_Kel(self):
            """
            Calculate element stiffness matrix by Gaussian integration
            """
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
            if (self.Mat.sy==None):
                depl = np.zeros(6)
            else:
                depl = self.Mat.epl_dot(self.sig, self.epl, self.CV,
                                        self.deps())
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
                    raise NotImplementedError('Error: Quadratic shape'+
                         'functions for 2D elements not yet implemented')
            return B
        
    def geom(self, sect=1, LX=None, LY=1., LZ=1.):
        '''Specify geometry of FE model with dimensions ``LX``, ``LY`` and ``LZ`` and 
        its subdivision into a number of sections;
        for 2-d model a laminate structure normal to x-direction can be created easily 
        with the in-built option to pass a list with the absolute lengths of each section
        in the parameter ``sect``. When this parameter is an integer it merely reserves 
        space for sections with arbitrary geometries;
        adds attributes to class ``Model``
        
        Parameters
        ----------
        sect  : list or int
            Either number of sections (int) or list with with absolute length of each section
        LX : float
            Length of model in x-direction (optional, default None in which case LX is
            calculated as the sum of the lengths of all sections)
        LY : float
            Length in y direction (optional, default: 1)
        LZ : float
            Thickness of model in z-direction (optional, default: 1)
        '''
        if type(sect)==list:
            self.Nsec = len(sect)  # number of sections
            self.LS = np.array(sect)
            self.lenx = sum(sect)  # total length of model
        elif type(sect)==int:
            if sect < 1:
                raise ValueError('At least one section must be defined.')
            if LX is None:
                raise ValueError('LX must be given if sect is of type int')
            else:
                self.lenx = LX
            self.Nsec = sect
            self.LS = np.ones(sect)*self.lenx/sect
        else:
            raise TypeError('Sect must be either list or int, not {}'\
                            .format(type(sect)))
        self.leny = LY
        self.thick = LZ
        
    def assign(self, mats):
        '''Assigns an object of class ``Material`` to each section.
        
        Parameters
        ----------
        mats : list 
            List of materials, dimension must be equal to number of sections
            
        Attributes
        ----------
        material.mat : List of material objects
            Internal variable for materials assigned to each section of the 
            geometry
        material.nonlin : bool
            Indicate if material non-linearity must be considered
        '''
        if len(mats)!=self.Nsec:
            raise ValueError('Numer of materials ({}) does not match number of sections ({})'\
                             .format(len(mats), self.Nsec))
        self.mat = mats
        self.nonlin = False
        for mat in mats:
            if mat.sy != None:
                self.nonlin = True   # nonlinear model if at least one material is plastic

    #subroutines to define boundary conditions, top/bottom only needed for 2-d models
    def bcleft(self, val=0., bctype='disp', bcdir='x'):
        '''Define boundary conditions on lhs nodes, either force or
        displacement type; static boundary conditions are assumed for lhs boundary. The 
        default is freezing all x-displacements to zero.
        
        Parameters
        ----------
        val : float
            Displacement or force of lhs nodes in bc_dir direction (optional, default: 0)
        bctype : str
            Type of boundary condition ('disp' or 'force')
            (optional, default: 'disp')
        bcdir : str or int
            Direction of boundary load (optional, default: 'x')
        '''
        if bcdir.lower()=='x' or bcdir==0:
            self.bcl[0] = val
            j = 0
        elif bcdir.lower()=='y' or bcdir==1:
            self.bcl[1] = val
            j = 1
        else:
            raise ValueError('bcleft: Unknown value for direction: {}'\
                             .format(bcdir))
        if (bctype.lower()=='disp'):
            #type of boundary conditions (BC)
            self.ubcleft[j] = True   # True: displacement BC on lhs node
        elif (bctype.lower()=='force'):
            self.ubcleft[j] = False   # False: force BC on lhs node
            if np.abs(val) > 1.e-6:
                raise ValueError('Finite force values at left boundary not supported.')
        else:
            raise ValueError('bcleft: Unknown BC: %s'%bctype)
        
    def bcright(self, val, bctype, bcdir='x'):
        '''Define boundary conditions on rhs nodes, either force or
        displacement type.
        If non-zero, the boundary loads will be incremented step-wise until the given 
        boundary conditions are fulfilled.
        
        Parameters
        ----------
        val     : float
            Displacement or force on rhs nodes in bc_dir direction 
        bctype : str
            Type of boundary condition ('disp' or 'force')
        bcdir : str or int
            Direction of boundary load (optional, default: 'x')
        '''
        if bcdir.lower()=='x' or bcdir==0:
            self.bcr[0] = val
            j = 0
        elif bcdir.lower()=='y' or bcdir==1:
            self.bcr[1] = val
            j = 1
        else:
            raise ValueError('bcright: Unknown value for direction :{}'\
                             .format(bcdir))
        if (bctype.lower()=='disp'):
            # type of boundary conditions (BC)
            self.ubcright[j] = True   # True: displacement BC on rhs node
        elif (bctype.lower()=='force'):
            self.ubcright[j] = False   # False: force BC on rhs node
        else:
            raise TypeError('bcright: Unknown BC: {}'.format(bctype))
        
    def bcbot(self, val=0., bctype='disp', bcdir='y'):
        '''Define boundary conditions on bottom nodes, either force or
        displacement type; static boundary conditions are assumed for bottom boundary. 
        The default is freezing all y-displacements to zero.
        
        Parameters
        ----------
        val  : float
            Displacement in bcdir direction (optional, default: 0)
        bctype : str
            Type of boundary condition ('disp' or 'force')
            (optional, default: 'disp')
        bcdir : str or int
            Direction of boundary load (optional, default: 'y')
        '''
        if self.dim!=2:
            warnings.warn('BC on bottom nodes will be ignoresd for 2D model')
        if bcdir.lower()=='x' or bcdir==0:
            self.bcb[0] = val
            j = 0
        elif bcdir.lower()=='y' or bcdir==1:
            self.bcb[1] = val
            j = 1
        else:
            raise ValueError('bcleft: Unknown value for direction: {}'\
                             .format(bcdir))
        if (bctype.lower()=='disp'):
            # type of boundary conditions (BC)
            self.ubcbot[j] = True   # True: displacement BC on bottom node
        elif (bctype.lower()=='force'):
            self.ubcbot[j] = False   # False: force BC on bottom node
            if np.abs(val) > 1.e-6:
                raise ValueError('Finite force values at bottom boundary not supported.')
        else:
            raise ValueError('bcbot: Unknown BC: {}'.format(bctype))
        
    def bctop(self, val, bctype, bcdir='y'):
        '''Define boundary conditions on top nodes, either force or displacement type. 
        If non-zero, the boundary loads will be incremented step-wise until the given 
        boundary conditions are fulfilled.
        
        Parameters
        ----------
        val     : float
            Displacement or force in bcdir direction 
        bctype : str
            Type of boundary condition ('disp' or 'force')
        bcdir : str or int
            Direction of boundary load (optional, default: 'y')
        '''
        if self.dim!=2:
            warnings.warn('BC on top nodes will be ignored for 2D model')
        if bcdir.lower()=='x' or bcdir==0:
            self.bct[0] = val
            j = 0
        elif bcdir.lower()=='y' or bcdir==1:
            self.bct[1] = val
            j = 1
        else:
            raise ValueError('bcleft: Unknown value for direction {}: '\
                             .format(bcdir))
        if (bctype.lower()=='disp'):
            # type of boundary conditions (BC)
            self.ubctop[j] = True   # True: displacement BC on rhs node
        elif (bctype.lower()=='force'):
            self.ubctop[j] = False   # False: force BC on rhs node
        else:
            raise TypeError('bctop: Unknown BC: {}'.format(bctype))
            
    def bcnode(self, node, val, bctype, bcdir):
        '''Define boundary conditions on a set of nodes defined in ``node``, 
        either force or displacement type in x or y-direction are accepted. 
        If non.zero, the boundary loads will be incremented step-wise until the given 
        boundary conditions are fulfilled.
        
        Since nodes must be given, this subroutine can only be called after
        meshing.
        
        Parameters
        ----------
        node   : int or list of int
            Node or set of nodes to which BC shall be applied
        val    : float
            Displacement or force in bcdir direction 
        bctype : str
            Type of boundary condition ('disp' or 'force')
        bcdir : str or int
            Direction of boundary load ('x' or 'y'; 0 or 1)
        '''
        if self.dim!=2:
            warnings.warn('BC on chosen nodes will be ignored for 2D model')
        if type(node)==list:
            self.noset = node
        else:
            self.noset = [node]
        if bcdir.lower()=='x' or bcdir==0:
            self.bcn[0] = val
            j = 0
        elif bcdir.lower()=='y' or bcdir==1:
            self.bcn[1] = val
            j = 1
        else:
            raise ValueError('bcleft: Unknown value for direction {}'\
                             .format(bcdir))
        if (bctype.lower()=='disp'):
            # type of boundary conditions (BC)
            self.ubcn[j] = True   # True: displacement BC on node
        elif (bctype.lower()=='force'):
            self.ubcn[j] = False   # False: force BC on node
        else:
            raise TypeError('bcnode: Unknown BC: {}'.format(bctype))

    def mesh(self, elmts=None, nodes=None, NX=10, NY=1, SF=1):
        '''
        Import mesh or
        generate structured mesh with quadrilateral elements (2d models). 
        First, nodal positions ``Model.npos`` are defined such that nodes lie
        at corners (linear shape function) and edges (quadratic shape function)
        of elements. 
        Then, elements are initialized as object of class ``Model.Element``,
        which requires the list of nodes associated with the element, the 
        dimensions of the element, and the material of the section in which 
        the element is situated to be passed.
        
        Parameters
        ----------
        elmts : (NX, NY) array
            Represents number of material as defined by list Model.mat
        nodes : (2,) array
            Defines positions of nodes on regular grid.
        NX : int
            Number of elements in x-direction (optional, default: 10)
        NY : int
            Number of elements in y-direction (optional, default: 1)
        SF : int
            Degree of shape functions: 1=linear, 2=quadratic
            (optional, default: 1)
        '''
        self.shapefact = SF
        DIM = self.dim
        if elmts is not None:
            el = np.array(elmts, dtype=int)
            sh = el.shape
            if len(sh) != DIM:
                raise ValueError('Cannot use a {}-shaped mesh with a {}-dimemsional model'\
                                 .format(sh, DIM))
            NX = sh[0]
            NY = sh[1] if DIM > 1 else 1
                
        if (NX < self.Nsec):
            raise TypeError('Error: Number of elements is smaller than number of sections')
        if (NY>1 and DIM==1):
            NY = 1
            warnings.warn('Warning: NY=1 for 1-d model')
        if self.u is not None:
            warnings.warn('Warning: Solution of previous steps is deleted')
            self.u = None
            self.f = None
        self.NnodeX = self.shapefact*NX + 1  # number of nodes along x axis
        self.NnodeY = (DIM-1)*self.shapefact*NY + 1  # number of nodes along y axis
        self.Nnode = self.NnodeX*self.NnodeY   # total number of nodes
        self.Ndof  = self.Nnode*DIM  # degrees of freedom
        if nodes is None:
            self.npos = np.zeros(self.Ndof)  # position array of nodes
        else:
            self.npos = np.ravel(nodes, order='C')
            if len(self.npos) != self.Nnode:
                raise ValueError('Inconsistent definition of nodes: '+
                      '{} nodes for {}-dim model with shape function={}'\
                      .format(len(self.npos)/DIM, DIM, SF))
        self.Nel = NX*NY
        self.element = [None] * self.Nel  # empty list for elements
        self.noleft  = []  # list of nodes on left boundary
        self.noright = []  # list of nodes on right boundary
        self.nobot   = []  # list of nodes on bottom boundary
        self.notop   = []  # list of nodes on top boundary
        self.noinner = []  # list of inner nodes
        
        if elmts is None:
            #Calculate number of elements per section -- only laminate structure
            hh = self.LS / self.lenx  # proportion of segment length to total length of model
            nes = [int(x) for x in np.round(hh*NX)]  # nes gives number of elements per segement in proportion 
            if (np.sum(nes) != NX):  # add or remove elements of largest section if necessary
                im = np.argmax(self.LS)
                nes[im] = nes[im] - np.sum(nes) + NX

            # Define nodal positions and element shapes -- only for laminate structure
            jstart = 0
            nrow = self.NnodeY
            dy = self.leny / NY
            for i in range(self.Nsec):
                # define nodal positions first
                ncol = nes[i]*self.shapefact + 1
                dx = self.LS[i] / nes[i]
                nr = np.max([1, nrow-1])
                elstart = np.sum(nes[0:i],dtype=int)*nr
                n1 = (int(elstart/NY)*nrow + int(np.mod(elstart,NY)))*\
                    self.shapefact
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
                # initialize elements
                for j in range(nes[i]*nr):
                    ih = elstart + j   # index of current element
                    n1 = (int(ih/NY)*nrow + ih%NY)*self.shapefact
                    n2 = n1 + self.shapefact
                    n3 = n1 + nrow*self.shapefact
                    n4 = n3 + self.shapefact
                    if (self.shapefact*DIM==1):
                        self.element[ih] = self.Element(self, [n1, n2], dx, dy,
                                                        self.mat[i])  # 1-d, lin shape fct
                    elif (self.shapefact*DIM==4):
                        nh = n1 + nrow + 1
                        hh = [n1, n1+1, n2, nh, nh+1, n3, n3+1, n4]  # 2-d, quad shape fct
                        self.element[ih] = self.Element(self, hh, dx, dy, self.mat[i])
                    elif (DIM==2):
                        hh = [n1, n2, n3, n4]  # 2-d, lin shape fct
                        self.element[ih] = self.Element(self, hh, dx, dy, self.mat[i])
                    else:
                        hh = [n1, n1+1, n2]  # 1-d, lin shape fct
                        self.element[ih] = self.Element(self, hh, dx, dy, self.mat[i])
                jstart = 1
        else:
            # create nodes on regular mesh if no nodes are given
            if nodes is None:
                dx = self.lenx/NX
                dy = self.leny/NY
                for j in range(self.NnodeX):
                    for k in range(self.NnodeY):
                        inode = j*self.NnodeY + k
                        self.npos[inode*DIM] = j*dx # x-position of node
                        if (DIM==2):
                            self.npos[inode*DIM+1] = k*dy # y-position of node
                        nin = True
                        if (j==0): 
                            self.noleft.append(inode)
                            nin = False
                        if (k==0):
                            self.nobot.append(inode)
                            nin = False
                        if (k==self.NnodeY-1):
                            self.notop.append(inode)
                            nin = False
                        if (j==self.NnodeX-1):
                            self.noright.append(inode)
                            nin = False
                        if nin:
                            self.noinner.append(inode)
            else:
                # save nodes on boundaries
                tol = 0.001*self.lenx/NX
                for inode, pos in enumerate(self.npos):
                    nin = True
                    if pos < tol: 
                        if DIM==1 or inode%2==0:
                            self.noleft.append(inode)
                        if DIM==2 and inode%2==1:
                            self.nobot.append(inode)
                        nin = False
                    if pos>self.lenx-tol and (DIM==1 or inode%2==0):
                        self.noright.append(inode)
                        nin = False
                    if pos>self.leny-tol and DIM==2 and inode%2==1:
                        self.notop.append(inode)
                        nin = False
                    if nin:
                        self.noinner.append(inode)
            # initialize elements
            for j in range(NX):
                for k in range(NY):
                    i = el[j, k] - 1
                    ih = j*NY + k
                    n1 = (int(ih/NY)*self.NnodeY + ih % NY)*self.shapefact
                    n2 = n1 + self.shapefact
                    n3 = n1 + self.NnodeY*self.shapefact
                    n4 = n3 + self.shapefact
                    if (self.shapefact*DIM==1):
                        self.element[ih] = self.Element(self, [n1, n2], dx, dy,
                                                        self.mat[i])  # 1-d, lin shape fct
                    elif (self.shapefact*DIM==4):
                        nh = n1 + self.NnodeY + 1
                        hh = [n1, n1+1, n2, nh, nh+1, n3, n3+1, n4]  # 2-d, quad shape fct
                        self.element[ih] = self.Element(self, hh, dx, dy, self.mat[i])
                    elif (DIM==2):
                        hh = [n1, n2, n3, n4]  # 2-d, lin shape fct
                        self.element[ih] = self.Element(self, hh, dx, dy, self.mat[i])
                    else:
                        hh = [n1, n1+1, n2]  # 1-d, quadratic shape fct
                        self.element[ih] = self.Element(self, hh, dx, dy, self.mat[i])

    def setupK(self):
        '''Calculate and assemble system stiffness matrix based on element stiffness matrices.
        
        Returns
        -------
        K  : 2d-array
            System stiffness matrix
        '''
        DIM = self.dim
        K = np.zeros((self.Ndof, self.Ndof))  # initialize system stiffness matrix
        for el in self.element:
            #assemble element stiffness matrix into system stiffness matrix
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
        '''Solve linear system of equations K.u = f with respect to u, to obtain distortions
        of the system under the applied boundary conditions for mechanical equilibrium, i.e.,
        when the total force on internal nodes is zero. In the first step, the stiffness 
        matrix K, the nodal displacements u and the nodal forces f are modified to conform
        with the boundary conditions (``calc_BC``), then an elastic predictor step is calculated,
        which is already the final solution for linear problems. For non-linear problems, i.e. for 
        plastic materials, the load step must be controlled and subdivided to fulfill 
        the side conditions of plastic materials, i.e. the equivalent stress must remain 
        smaller than the yield strength, or the flow stress during plastic yielding. This is 
        ensured in a self.consistency loop for non-linear models. The system of 
        equations is solved by invoking the subroutine numpy.linalg.solve. The method yields
        the final solution 
        for nodal displacements u and nodal forces f as attributes of 
        class ``Model``; global stress, global total strain and global plastic strain are 
        evaluated and stored as attributes, too. Element solutions for stresses and strains 
        are stored as attributes of class ``Element``, see documentation of this class.
        
        Parameters
        ----------
        min_step : int
            Minimum number of load steps (optional)
        verb     : Boolean
            Be verbose in text output (optional, default: False)
            
        Yields
        ------
        Model.u     : (Model.Ndof,) array
            Nodal displacements
        Model.f     : (Model.Ndof,) array
            Nodal forces
        Model.sgl   : (N,6) array
            Global stress as as Voigt tensor for each incremental load step (homogenized element solution)
        Model.egl   : (N,6) array
            Global total strain as as Voigt tensor for each incremental load step (homogenized element solution)
        Model.epgl  : (N,6) array
            Global plastic strain as as Voigt tensor for each incremental load step (homogenized element solution)
        Element.sig : (6,) array
            Element solution for stress tensor
        Element.eps : (6,) array
            Element solution for total strain tensor
        Element.epl : (6,) array
            Element solution for plastic strain tensor
        '''
        #test if meshing has been performed
        if self.Nnode is None:
            raise AttributeError('Attributes for mesh not set, but required by solver.')
        
        #calculate reduced stiffness matrix according to BC
        def Kred(K, ind):
            Kred = np.zeros((len(ind), len(ind)))
            for i in range(len(ind)):
                for j in range(len(ind)):
                    Kred[i,j] = K[ind[i], ind[j]]
            return Kred
            
        #calculate scaling factor for load steps
        def calc_scf():
            sc_list = []
            for el in self.element:
                # test if yield criterion is exceeded
                sref = Stress(el.dsig()).seq(el.Mat) # max. element stress increment
                if el.Mat.sy!=None and sref>0.1:  # not necessary for elastic material or small steps
                    yf0 = el.Mat.calc_yf(el.sig, epl=el.epl)  # yield fct. at start of load step
                    #if element starts in elastic regime load step can only touch yield surface
                    if  yf0 < -0.15:
                        if el.Mat.ML_yf:
                            # for categorial ML yield function, calculate yf0
                            # as exact distance to yield surface
                            yf0 = el.Mat.ML_full_yf(el.sig, el.epl, ld=sld, verb=verb)
                        hh = np.minimum(1., -yf0/sref)
                        sc_list.append(hh)
                    else:
                        # make sure load step does not exceed yield surface too much
                        hh = np.minimum(1., np.sqrt(1.5)*el.Mat.get_sflow(eps_eq(el.epl))/sref)
                    sc_list.append(hh)
            # select scaling appropriate scaling such that no element crosses yield surface
            if len(sc_list)==0: sc_list=[1.]
            hh = np.std(sc_list)
            if hh < 0.1:
                scf = np.amin(sc_list)  # if almost all elements have same yield fct, take minimum
            else:
                hm = np.mean(sc_list)
                scf = np.maximum(1.e-3, hm-hh) # otherwise take average - standard deviation
            if scf<1.e-3:
                if verb:
                    warnings.warn('Warning: Small load increment in calc_scf: '+str(scf))
                scf = 1.e-3
            return scf

        #define BC: modify stiffness matrix for displacement BC, calculate consistent force BC
        def calc_BC(K, bcl0, bcb0, dbcr, dbct, dbcn):
            '''BC on lhs and bottom nodes nodes is always static.  
            Displacement type BC are applied by adding known boundary forces to solution vector
            to reduce rank of system of equations by 1 (one row eliminated)
            
            Paramaters
            ----------
            K : (Ndof, Ndof)-array
                stiffness matrix
            bcl0, bcb0 : dim-arrays
                static BC on lhs and bottom nodes
            dbcr, dbct : dim-arrays
                increment of BC on rhs and top nodes
            dbcn : dim-array
                increment of BC on selected node set
                
            Returns
            -------
            du : (Ndof)-array
                Increment of nodal displacements at bpundary
            df : (Ndof)-array
                Increment of nodal forces at boundary 
            ind : list
                List of all nodal DOF for which solution must be calculated, 
                i.e. for which no displacement-type BC are given.
            '''
            du = np.zeros(self.Ndof)
            df = np.zeros(self.Ndof)
            ind = list(range(self.Ndof))  # list of all nodal DOF, will be shortened according to BC
            for k in range(self.dim):
                if self.ubcleft[k]:
                    for j in self.noleft:
                        i = j*self.dim + k   # postion of x or y-values of node #j in u/f vector
                        ind.remove(i)
                        du[i] = bcl0[k]
                        df[ind] -= K[ind,i]*bcl0[k]
            
            if self.dim==2:
                # BC on bottom nodes is always static
                # apply BC by adding known boundary forces to solution vector
                # to reduce rank of system of equations by 1 (one row eliminated)
                for k in range(self.dim):
                    if self.ubcbot[k]:
                        for j in self.nobot:
                            i = j*self.dim + k   # postion of x or y-values of node #j in u/f vector
                            if i in ind:
                                ind.remove(i)
                                du[i] = bcb0[k]
                            else:
                                if du[i] != bcb0[k]:
                                    warnings.warn('Inconsistent BC at left ({}) and bottom node {} ({}).'\
                                    .format(du[i], j, bcb0[k]))
                            df[ind] -= K[ind,i]*bcb0[k]

            # rhs node BC can be force or displacement
            # apply BC and solve corresponding system of equations
            for k in range(self.dim):
                if self.ubcright[k]:
                    # displacement BC
                    # add known boundary force to solution vector to
                    # eliminate row Ndof from system of equations
                    for j in self.noright:
                        i = j*self.dim + k
                        if i in ind:
                            ind.remove(i)
                            du[i] = dbcr[k]
                        else:
                            if du[i] != dbcr[k]:
                                warnings.warn('Inconsistent BC at right node {} ({}) and bottom ({}).'\
                                .format(j,du[i], dbcr[k]))
                        hh = list(range(self.Ndof))
                        hh.remove(i)
                        df[hh] -= K[i, hh]*dbcr[k]
                else:
                    # force bc on rhs nodes
                    for j in self.noright:
                        i = j*self.dim + k
                        hh=1./(self.NnodeY-1) # calculate share of force on nodes
                        hy = self.npos[j*self.dim+1] #y-position of node
                        if hy<1.e-3 or hy>self.leny-1.e-3:
                            hh*=0.5  # reduce force on corner nodes
                        df[i] += dbcr[k]*hh
                
            # BC on top nodes can be force or displacement
            # apply BC and solve corresponding system of equations
            if self.dim==2:
                for k in range(self.dim):
                    if self.ubctop[k]:
                        # displacement BC
                        # add known boundary force to solution vector to
                        # eliminate row Ndof from system of equations
                        for j in self.notop:
                            i = j*self.dim + k
                            if i in ind:
                                ind.remove(i)
                                du[i] = dbct[k]
                            else:
                                if du[i] != dbct[k]:
                                    warnings.warn('Inconsistent BC at top ({}) and left/right node {} ({}).'\
                                    .format(du[i], j, dbcr[k]))
                            df[ind] -= K[ind,i]*dbct[k]
                    else:
                        # force bc on top nodes
                        for j in self.notop:
                            i = j*self.dim + k
                            hh=1./(self.NnodeX-1) # share of force for each node
                            hx = self.npos[j*self.dim] #x-position of node
                            if hx<1.e-3 or hx>self.lenx-1.e-3:
                                hh*=0.5  # reduce force on corner nodes
                            df[i] += dbct[k]*hh
                            
            # BC on selected node set can be force or displacement
            # apply BC and solve corresponding system of equations
            if self.dim==2 and self.noset is not None:
                if dbcn is None:
                    raise ValueError('No BC for selected node set given.')
                for k in range(self.dim):
                    if self.ubcn[k]:
                        # displacement BC
                        # add known boundary force to solution vector to
                        # eliminate row Ndof from system of equations
                        for j in self.noset:
                            i = j*self.dim + k
                            if i in ind:
                                ind.remove(i)
                                du[i] = dbcn[k]
                            else:
                                if du[i] != dbcn[k]:
                                    warnings.warn('Inconsistent BC at node set ({}) and left/right node {} ({}).'\
                                    .format(du[i], j, dbcn[k]))
                            df[ind] -= K[ind,i]*dbcn[k]
                    else:
                        # force bc on node set
                        for j in self.noset:
                            i = j*self.dim + k
                            df[i] += dbcn[k]
            return du, df, ind
                
        jin = []
        for j in self.noinner:
            jin.append(j*self.dim)
            jin.append(j*self.dim+1)
                
        if self.u is None:
            # declare and initialize solution and result vectors, and boundary conditions
            self.u = np.zeros(self.Ndof)
            self.f = np.zeros(self.Ndof)
            self.sgl = np.zeros((1,6))
            self.egl = np.zeros((1,6))
            self.epgl = np.zeros((1,6)) 
            #initialize element quantities
            for el in self.element:
                el.elstiff = el.CV
                el.calc_Kel()
                el.eps = np.zeros(6)
                el.sig = np.zeros(6)
                el.epl = np.zeros(6)
            bcr0 = np.zeros(self.dim)
            bct0 = np.zeros(self.dim)
            self.bct_mem = np.zeros(self.dim)
            self.bcr_mem = np.zeros(self.dim)
            if self.noset is not None:
                bcn0 = np.zeros(self.dim)
                self.bcn_mem = np.zeros(self.dim)
        else:
            bcr0 = self.bcr_mem
            bct0 = self.bct_mem
            if self.noset is not None:
                bcn0 = self.bcn_mem
        bcl0 = self.bcl
        bcb0 = self.bcb
        K = self.setupK()  # assemble system stiffness matrix from element stiffness matrices
        # construct Voigt type tensor in loading direction for search of yield
        # point, used for scaling of load step for ML flow rules
        sld = np.zeros(6)
        if np.abs(self.bcr[0]) > 1.e-6:
            sld[0] = np.sign(self.bcr[0])
        if self.dim > 1:
            if np.abs(self.bct[1]) > 1.e-6:
                sld[1] = np.sign(self.bct[1])
            if np.abs(self.bcr[1]) > 1.e-6:
                sld[5] = np.sign(self.bcr[1])
        if np.abs(self.bct[0]) > 1.e-6:
            sld[5] = np.sign(self.bct[0])
        if np.linalg.norm(sld) < 1.e-3:
            warnings.warn('solve: inconsistent BC sld={}, bct={}, bcr={}'
                          .format(sld, self.bct, self.bcr))
            sld[0] = 1.
                
        #define loop for external load steps (BC subdivision)
        #during each load step mechanical equilibrium is calculated for sub-step
        #the tangent stiffness matrix of the last load step is used as initial guess
        #current tangent stiffness matrix compatible with BC is determined iteratively
        il = 0
        nit = 0
        niter = []
        co_nconv = []
        bc_inc = True
        nconv = 0
        while bc_inc:
            #define global increments for boundary conditions
            max_dbct = self.bct - bct0
            max_dbcr = self.bcr - bcr0
            if min_step is not None:
                sc = np.maximum(1, min_step - il)
                max_dbct /= sc
                max_dbcr /= sc
            #calculate du and df fulfilling mech. equil. for max. load step consistent with stiffness matrix K
            dbcr = max_dbcr
            dbct = max_dbct
            if self.noset is not None:
                max_dbcn = self.bcn - bcn0
                if min_step is not None:
                    max_dbcn /= np.maximum(1, min_step - il)
                dbcn = max_dbcn
            else:
                dbcn = None
            
            # linear model can be solved directly in one step, elastic predictor for nonlinear models
            self.du, df, ind = calc_BC(K, bcl0, bcb0, dbcr, dbct, dbcn) # consider BC for system of equ. 
            self.du[ind] = np.linalg.solve(Kred(K, ind), df[ind]) # Solve reduced system of equations
            
            if self.nonlin:
                #calculate scaling factor for predictor step in case of non-linear model
                # calculate global predictor step to hit the yield surface
                scale_bc = (calc_scf() if il < 10 else 1.)
                dbcr = max_dbcr*scale_bc  # improves the accuracy of the initial yield point
                dbct = max_dbct*scale_bc  # for finite element simulation
                nit = 0
                change = True
                conv = False
                if verb:
                    print('***Load step #',il)
                    print('scaling factor',scale_bc)
                while (change or not conv) and nit<=15:
                    # repeat solution step until stiffness matrix remains unchanged
                    # a proper Newton-Raphson algorithm should be implemented here
                    if il < 6 and nit>1:
                        # start reducing load increments to reach convergence
                        hs = 0.5
                        for k in range(self.dim):
                            if max_dbcr[k] >= 0:
                                hh = np.minimum(self.bcr[k]-bcr0[k], dbcr[k]*hs)
                                dbcr[k] = np.maximum(0.05*max_dbcr[k], hh)
                            else:
                                hh = np.maximum(self.bcr[k]-bcr0[k], dbcr[k]*hs)
                                dbcr[k] = np.minimum(0.05*max_dbcr[k], hh)
                            if max_dbct[k] >= 0:
                                hh = np.minimum(self.bct[k]-bct0[k], dbct[k]*hs)
                                dbct[k] = np.maximum(0.05*max_dbct[k], hh)
                            else:
                                hh = np.maximum(self.bct[k]-bct0[k], dbct[k]*hs)
                                dbct[k] = np.minimum(0.05*max_dbct[k], hh)
                            if self.noset is not None:
                                if max_dbcn[k] >= 0:
                                    hh = np.minimum(self.bcn[k]-bcn0[k], dbcn[k]*hs)
                                    dbcn[k] = np.maximum(0.05*max_dbcn[k], hh)
                                else:
                                    hh = np.maximum(self.bcn[k]-bcn0[k], dbcn[k]*hs)
                                    dbcn[k] = np.minimum(0.05*max_dbcn[k], hh)
                        
                    # solve system with current K matrix
                    K = self.setupK()  # assemble updated tangent stiffness matrix
                    self.du, df, ind = calc_BC(K, bcl0, bcb0, dbcr, dbct, dbcn)
                    self.du[ind] = np.linalg.solve(Kred(K, ind), df[ind]) # solve du with current stiffness matrix
                    
                    # evaluate material response in each element
                    f = []
                    change = False # flag if stiffness matrix is changing
                    for el in self.element:
                        if (el.Mat.sy!=None):  # not necessary for elastic material
                            fyld, el.res_sig, el.res_depl, gr_stiff = \
                                el.Mat.response(el.sig, el.epl, el.deps(), el.CV) 
                            el.res_deps = el.deps()
                            f.append(fyld/el.Mat.get_sflow(eps_eq(el.epl)))    # store result of yield function
                            hh = np.linalg.norm(el.elstiff-gr_stiff) # Frobenius norm of differences b/w stiffness matrices before and after load step
                            if hh > 1.e-3:
                                # if difference is too large, update element stiffness matrix
                                if nit < 15:
                                    el.elstiff = gr_stiff
                                else:
                                    el.elstiff = 0.5*(gr_stiff + el.elstiff)
                                el.calc_Kel()  # update element stiffness matrix
                                change = True
                            el.stat_nlin['max_steps'] = np.maximum(el.Mat.msg['nsteps'], el.stat_nlin['max_steps'])
                            el.stat_nlin['max_dstiff'] = np.maximum(hh, el.stat_nlin['max_dstiff'])
                        else:
                            f.append(0.)
                    f = np.array(f)
                    conv = np.all(f <= yf_tolerance * 1.0001)
                    if verb:
                        if not conv:
                            print('\n  ###  Warning: No convergence of plasticity algorithm in trial step #',nit)
                            print('  ###  yield function=',f)#,'residual forces on inner nodes=',fres[jin])
                            print('  ###  Convergence stats (ptol=', yf_tolerance, '):')
                            for j,el in enumerate(self.element):
                                print('EL #',j,'iteration steps:', \
                                  el.stat_nlin['max_steps'], 'max. Dstiff:',el.stat_nlin['max_dstiff']) 
                            print('\n')
                        print('+++Inner trial step #',nit)
                        #fres = K @ (self.u+self.du)
                        print('load increment right:', dbcr)
                        print('load increment top:',dbct)
                        if self.noset is not None:
                            print('load increment set:',dbcn)
                    if not conv:
                        nconv += 1
                    nit += 1
                # end while change
            # end if nonlin    
            # update internal variables with results of load step
            self.u += self.du
            self.f += K @ self.du
            for el in self.element:
                if el.res_sig is None:
                    el.epl += el.depl()
                    el.sig += el.dsig()
                else:
                    el.epl += el.res_depl
                    el.sig = el.res_sig    
                el.eps = el.eps_t()

            # update load step
            il += 1
            niter.append(nit-1)
            co_nconv.append(nconv)
            bcr0 += dbcr
            hl0 = np.abs(bcr0[0]-self.bcr[0])>1.e-6 and np.abs(self.bcr[0])>1.e-9
            if self.dim>1:
                hl1 = np.abs(bcr0[1]-self.bcr[1])>1.e-6 and np.abs(self.bcr[1])>1.e-9
                bct0 += dbct
                hr0 = np.abs(bct0[0]-self.bct[0])>1.e-6 and np.abs(self.bct[0])>1.e-9
                hr1 = np.abs(bct0[1]-self.bct[1])>1.e-6 and np.abs(self.bct[1])>1.e-9
                if self.noset is not None:
                    bcn0 += dbcn
                    hr0 = hr0 or (np.abs(bcn0[0]-self.bcn[0])>1.e-6 and np.abs(self.bcn[0])>1.e-9)
                    hr1 = hr1 or (np.abs(bcn0[1]-self.bcn[1])>1.e-6 and np.abs(self.bcn[1])>1.e-9)
            else:
                hl1 = False
                hr0 = False
                hr1 = False
            bc_inc = hr0 or hr1 or hl0 or hl1
            #store time dependent quantities
            self.calc_global()  # calculate global values for solution
            self.sgl  = np.append(self.sgl, [self.glob['sig']], axis=0)
            self.egl  = np.append(self.egl, [self.glob['eps']], axis=0)
            self.epgl = np.append(self.epgl,[self.glob['epl']], axis=0)
            if verb:
                print('Iteration step #',nit)
                print('Load increment ', il, 'total',self.ubctop,'top ',bct0,'/',self.bct,'; last step ',dbct)
                print('Load increment ', il, 'total',self.ubcright,'rhs',bcr0,'/',self.bcr,'; last step ',dbcr)
                if self.noset is not None:
                    print('Load increment ', il, 'total',self.ubcn,'set',bcn0,'/',self.bcn,'; last step ',dbcn)
                print('BC strain (11,22,12): ', np.around([self.glob['ebc1'],self.glob['ebc2'],self.glob['ebc12']],decimals=5))
                print('BC stress (11,22,12): ', np.around([self.glob['sbc1'],self.glob['sbc2'],self.glob['sbc12']],decimals=3))
                print('Global strain: ', np.around(self.glob['eps'],decimals=5))
                print('Global stress: ', np.around(self.glob['sig'],decimals=3))
                print('Global plastic strain: ', np.around(self.glob['epl'],decimals=6))
                seq = sig_eq_j2(self.glob['sig'])
                if seq>1.e-3:
                    hh =  np.abs(self.glob['sbc1'] - self.glob['sig'][0])
                    hh += np.abs(self.glob['sbc2'] - self.glob['sig'][1])
                    hh += np.abs(self.glob['sbc12'] - self.glob['sig'][5])
                    hh /= seq
                    if hh > 1.e-3:
                        warnings.warn('***Inconstistent stiffness matrix!\
                        Rel. error is stress={}'.format(hh))
                        print(self.glob)
                hh = [el.sig-el.CV@(el.eps-el.epl) for el in self.element]
                if np.abs(np.amax(hh)) > 1.:
                    warnings.warn('\n ***TEST failed: {}\n\n'.format(hh))
                print('----------------------------')
        self.bct_mem = bct0
        self.bcr_mem = bcr0
        self.nsteps = il
        self.niter = niter
        self.co_nconv = co_nconv

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
        
        Yields
        ------
        Model.glob : dictionary
            Values for stress ('sig'), total strain ('eps') and plastic strain ('epl')
            as homogenized element solutions (Voigt tensors); values for (11)- and 
            (22)-components of stress ('sbc1','scb2') and total strain ('ebc1', 'ebc2')
            as homogenized values of boundary nodes.
        '''
        #calculate global values from BC
        uxl, uyl, fxl, fyl = self.bcval(self.noleft)
        uxr, uyr, fxr, fyr = self.bcval(self.noright)
        self.glob['ebc1'] = (uxr-uxl)/self.lenx
        self.glob['sbc1'] = 0.5*(fxr-fxl)/(self.leny*self.thick)
        self.glob['ebc21'] = (uyr-uyl)/self.lenx
        self.glob['sbc21'] = 0.5*(fyr-fyl)/(self.leny*self.thick)
        if (self.dim==2):
            uxb, uyb, fxb, fyb = self.bcval(self.nobot)
            uxt, uyt, fxt, fyt = self.bcval(self.notop)
            self.glob['ebc2'] = (uyt-uyb)/self.leny
            self.glob['sbc2'] = 0.5*(fyt-fyb)/(self.lenx*self.thick)
            self.glob['ebc12'] = (uxt-uxb)/self.leny
            self.glob['sbc12'] = 0.5*(fxt-fxb)/(self.lenx*self.thick)
        #calculate global values from element solutions
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

    def plot(self, fsel, mag=10, colormap='viridis', cdepth=20, showmesh=True, shownodes=True,
            vmin=None, vmax=None, annot=True, file=None):
        '''Produce graphical output: draw elements in deformed shape with color 
        according to field variable 'fsel'; uses matplotlib
        
        Parameters
        ----------
        fsel   : str
            Field selector for library field, see Keyword Arguments for possible values
        mag    : float
            Magnification factor for displacements (optional, default: 10)
        vmin   : float
            Start value for range of plotted values
        vmax   : float
            End value for range of plotted values
        cdepth : int
            Number of colors in colormap (optional, default: 20)
        showmesh : Boolean
            Set/unset plotting of lines for element edges (optional, default: True)
        shownodes: Boolean
            Set/unset plotting of nodes (optional, default: True)
        colormap : str
            Name of colormap to be used (optional, default: viridis)
        annot : Boolean
            Show annotations for x and y-axis (optional, default: True)
        file  : str
            If a filename is provided, plot is exported as PDF (optional, default: None)
            
        Keyword Arguments
        -----------------
        strain1  : 
            total strain in horizontal direction
        strain2  :
            total strain in vertical direction
        strain12  :
            total shear strain, xy-component
        stress1  : 
            horizontal stress component
        stress2  : 
            vertical stress component
        stress12  : 
            xy-shear stress component
        plastic1 :
            plastic strain in horizontal direction
        plastic2 :
            plastic strain in vertical direction
        plastic12 :
            plastic strain, xy-component
        seq      :
            equivalent stress (Hill-formulation for anisotropic plasticity)
        seqJ2    :
            equivalent J2 stress
        peeq     : 
            equivalent plastic strain
        etot     : 
            equivalent total strain
        ux       : 
            horizontal displacement
        uy       : 
            vertical displacement
        mat     :
            materials and sections of model
        '''
        fig, ax = plt.subplots(1)
        cmap = plt.cm.get_cmap(colormap, cdepth)

        def strain1():
            hh = [el.eps[0]*100 for el in self.element]
            text_cb = r'$\epsilon^\mathrm{tot}_{11}$ (%)'
            return hh, text_cb

        def strain2():
            hh = [el.eps[1]*100 for el in self.element]
            text_cb = r'$\epsilon^\mathrm{tot}_{22}$ (%)'
            return hh, text_cb

        def strain12():
            hh = [el.eps[5]*100 for el in self.element]
            text_cb = r'$\epsilon^\mathrm{tot}_{12}$ (%)'
            return hh, text_cb

        def stress1():
            hh = [el.sig[0] for el in self.element]
            text_cb = r'$\sigma_{11}$ (MPa)'
            return hh, text_cb

        def stress2():
            hh = [el.sig[1] for el in self.element]
            text_cb = r'$\sigma_{22}$ (MPa)'
            return hh, text_cb

        def stress12():
            hh = [el.sig[5] for el in self.element]
            text_cb = r'$\sigma_{12}$ (MPa)'
            return hh, text_cb

        def plastic1():
            hh = [el.epl[0]*100 for el in self.element]
            text_cb = r'$\epsilon^\mathrm{pl}_{11}$ (%)'
            return hh, text_cb

        def plastic2():
            hh = [el.epl[1]*100 for el in self.element]
            text_cb = r'$\epsilon^\mathrm{pl}_{22}$ (%)'
            return hh, text_cb

        def plastic12():
            hh = [el.epl[5]*100 for el in self.element]
            text_cb = r'$\epsilon^\mathrm{pl}_{12}$ (%)'
            return hh, text_cb

        def stress_eq():
            hh = [Stress(el.sig).seq(el.Mat) for el in self.element]
            text_cb = r'$\sigma_{eq}$ (MPa)'
            return hh, text_cb

        def stress_eqJ2():
            hh = [Stress(el.sig).seq_j2() for el in self.element]
            text_cb = r'$\sigma^\mathrm{J2}_{eq}$ (MPa)'
            return hh, text_cb

        def strain_peeq():
            hh = [eps_eq(el.epl)*100 for el in self.element]
            text_cb = r'$\epsilon^\mathrm{pl}_{eq}$ (%)'
            return hh, text_cb

        def strain_etot():
            hh = [eps_eq(el.eps)*100 for el in self.element]
            text_cb = r'$\epsilon^\mathrm{tot}_{eq}$ (%)'
            return hh, text_cb

        def disp_x():
            hh = [el.eps[0]*self.lenx for el in self.element]
            text_cb = r'$u_x$ (mm)'
            return hh, text_cb

        def disp_y():
            hh = [el.eps[1]*self.leny for el in self.element]
            text_cb = r'$u_y$ (mm)'
            return hh, text_cb

        def disp_mat():
            hh = [el.Mat.num for el in self.element]
            text_cb = 'Material number'
            return hh, text_cb

        field={
            'strain1'  : strain1(),
            'strain2'  : strain2(),
            'strain12' : strain12(),
            'stress1'  : stress1(),
            'stress2'  : stress2(),
            'stress12' : stress12(),
            'plastic1' : plastic1(),
            'plastic2' : plastic2(),
            'plastic12': plastic12(),
            'seq'      : stress_eq(),
            'seqJ2'    : stress_eqJ2(),
            'peeq'     : strain_peeq(),
            'etot'     : strain_etot(),
            'ux'       : disp_x(),
            'uy'       : disp_y(),
            'mat'      : disp_mat()
        }
        
        #define color value by mapping field value of element to interval [0,1]
        val, text_cb = field[fsel]
        auto_scale = (vmin is None) and (vmax is None)
        if vmin is None:
            vmin = np.amin(val)
        if vmax is None:
            vmax = np.amax(val)
        delta = np.abs(vmax - vmin)
        if auto_scale and (delta < 0.1 or delta/vmax < 0.04):
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
        col = np.round(np.subtract(val,vmin)/delta, decimals=5)
        
        #create element plots
        for el in self.element:
            # draw filled polygon for each element
            if (self.dim==1):
                ih = np.amin(el.nodes)  # left node of current element
                jh = np.amax(el.nodes)  # right node of current element
                ih1 = ih*self.dim       # position of ux in u vector
                jh1 = jh*self.dim       # position of ux in u vector
                hx1 = self.npos[ih]     # x position of left node 
                hx2 = self.npos[jh]     # x position of right node
                if mag>0. and self.u is not None:
                    hx1 += mag*self.u[ih1] # add nodal displacement
                    hx2 += mag*self.u[jh1] 
                hh = self.thick*0.5
                hx = [hx1, hx2, hx2, hx1]
                hy = [-hh, -hh, hh, hh]
            else:
                hx = [0, 0, 0, 0]
                hy = [0, 0, 0, 0]
                k = [0, 3, 1, 2]
                for p, ih in enumerate(el.nodes):
                    j = ih*self.dim
                    hx[k[p]] = self.npos[j]
                    hy[k[p]] = self.npos[j+1]
                    if mag>0. and self.u is not None:
                        hx[k[p]] += mag*self.u[j]
                        hy[k[p]] += mag*self.u[j+1]
                        
            ax.fill(hx, hy, color=cmap(col[self.element.index(el)]))
            if (showmesh):
                hx.append(hx[0])
                hy.append(hy[0])
                ax.plot(hx, hy, 'k', lw=1)  # plot edges of elements

        #plot nodes
        if (shownodes):
            hh = self.npos
            if mag>0. and self.u is not None:
                hh += mag*self.u
            if (self.dim==1):
                hx = hh
                hy = np.zeros(self.Ndof)
            else:
                hx = hh[0:self.Ndof:2]
                hy = hh[1:self.Ndof:2]
            ax.scatter(hx, hy, s=50, c='red', marker='o', zorder=3)
 
        #add colorbar
        axl = fig.add_axes([1.01, 0.15, 0.04, 0.7])  #[left, bottom, width, height]
        # for use in juypter note book: left = 1.01, for python: left = 0.86
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        cb1 = colorbar.ColorbarBase(axl, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label(text_cb)
        #add axis annotations
        if annot:
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
        ax.set_aspect('equal', 'box')  # enforce equal scale on both axes
        #fig.tight_layout()
        #save plot to file if filename is provided
        if file is not None:
            fig.savefig(file+'.pdf', format='pdf', dpi=300)
        plt.show()
