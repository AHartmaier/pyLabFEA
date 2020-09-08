#Module pylabfea.mod
'''Module pylabefea.model introduces global functions for mechanical quantities and class ``Model`` that 
contains that attributes and methods needed in FEA. Materials are defined in 
module pylabfea.material

uses NumPy, SciPy, MatPlotLib

Version: 2.1 (2020-04-01)
Author: Alexander Hartmaier, ICAMS/Ruhr-University Bochum, April 2020
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)'''
from pylabfea.basic import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
   
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
        List of objects of class ``Element``, dimension Nel (defined in ``mesh``)
    u    : (Ndof,) array
        List of nodal displacements (defined in ``solve``)
    f    : (Ndof,) array
        List of nodal forces (defined in ``solve``)
    sgl  : (N,6) array
        Time evolution of global stress tensor with incremental load steps (defined in ``solve``)
    egl  : (N,6) array
        Time evolution of global total strain tensor with incremental load steps (defined in ``solve``)
    epgl : (N,6) array
        Time evolution of global plastic strain tensor with incremental load steps (defined in ``solve``)
    glob : python dictionary
        Global values homogenized from BC or element solutions, contains the elements: 
        'ebc1', 'ebc2', 'sbc1', 'sbc2' : global strain and stress from BC (type: float)
        'eps', 'epl', 'sig',  : global strain, plastic strain, and stress tensors homogenized 
        from element solutions (type: Voigt tensor)     (defined in ``calc_global``)
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
        #print('Model initialized')
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
        '''

        def __init__(self, model, nodes, lx, ly, sect, mat):
            self.Model = model
            self.nodes = nodes
            self.Lelx = lx
            self.Lely = ly
            self.Sect = sect
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
            if (self.Mat.sy==None):
                depl = np.zeros(6)
            else:
                depl = self.Mat.epl_dot(self.sig, self.epl, self.CV, self.deps())
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
        '''Assigns an object of class ``Material`` to each section.
        
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
        '''Generate structured mesh with quadrilateral elements (2d models). First,
        nodal positions ``Model.npos`` are defined such that nodes lie at 
        corners (linear shape function) and edges (quadratic shape function) of elements. 
        Then, elements are initialized as object of class ``Model.Element``, which requires the list of 
        nodes associated with the element, the dimensions of the element, and the material of
        the section in which the element is situated to be passed.
        
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
        '''Calculate and assemble system stiffness matrix based on element stiffness matrices.
        
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
            Nodel forces
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
        'calculate reduced stiffness matrix according to BC'
        def Kred(ind):
            Kred = np.zeros((len(ind), len(ind)))
            for i in range(len(ind)):
                for j in range(len(ind)):
                    Kred[i,j] = K[ind[i], ind[j]]
            return Kred
        
        'evaluate yield function in each element/for each Gauss point and change element stiffness matrix'
        'to tangent stiffness if in plastic regime'
        def calc_el_yldfct():
            f = []
            # test if yield criterion is exceeded
            for el in self.element:
                if (el.Mat.sy!=None):  # not necessary for elastic material
                    peeq = eps_eq(el.epl+el.depl()) # calculate equiv. plastic strain
                    sig = el.sig+el.dsig()
                    if el.Mat.ML_yf and el.Mat.msparam is not None:
                        hh = el.Mat.ML_full_yf(sig)
                    else:
                        hh = el.Mat.calc_yf(sig, peeq=peeq)
                    if hh > ptol:
                        # for debugging
                        '''h0 = el.Mat.calc_yf(el.sig,peeq=eps_eq(el.epl)) # calculate yield function at last converged step
                        if h0 < -ptol:
                            'origin of step in elastic region, force step to end at yield surface'
                            print('Warning in calc_el_yldfct: Increment crosses yield surface')
                            print('sig, peeq, hh, h0, ptol, el.sig, el.epl', sig, peeq, hh, h0, ptol, el.sig, el.epl)
                            hh = 0.
                        else:
                            'step fully in plastic regime'
                        '''
                        el.elstiff = el.Mat.C_tan(el.sig, el.CV)  # for plasticity calculate tangent stiffness
                        el.calc_Kel()                             # and update element stiffness matrix
                    f.append(hh)
                else:
                    f.append(0.)
            return np.array(f)
        
        'calculate scaling factor for load steps'
        def calc_scf():
            sc_list = [1.]
            for el in self.element:
                # test if yield criterion is exceeded
                if (el.Mat.sy!=None):  # not necessary for elastic material
                    peeq = eps_eq(el.epl) # calculate equiv. plastic strain
                    yf0 = el.Mat.calc_yf(el.sig, peeq=peeq)  # yield fct. at start of load step
                    'if element starts in elastic regime load step can only touch yield surface'
                    if  yf0 < -0.15:
                        sref = Stress(el.sig+el.dsig()).seq(el.Mat) # element stress at max load step
                        if el.Mat.ML_yf:
                            'for categorial ML yield function, calculate yf0 as distance to yield surface'
                            'construct normal stress vector in loading sirection for search of yield point'
                            hs = np.zeros(3)
                            if np.abs(max_dbcr)>1.e-6:
                                hs[0] = el.Mat.sy*np.sign(max_dbcr)
                            if np.abs(max_dbct)>1.e-6:
                                hs[1] = el.Mat.sy*np.sign(max_dbct)
                            yf0 = el.Mat.ML_full_yf(el.sig, ld=hs, verb=verb)  # distance of initial stress state to yield locus
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
                    hh=1./(self.NnodeY-1)
                    hy = self.npos[i+1] #y-position of node
                    if hy<1.e-3 or hy>self.leny-1.e-3:
                        hh*=0.5  # reduce force on corner nodes
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
                        hh=1./(self.NnodeX-1)
                        hx = self.npos[i-1] #x-position of node
                        if hx<1.e-3 or hx>self.lenx-1.e-3:
                            hh*=0.5  # reduce force on corner nodes
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
            'initialize element quantities'
            for el in self.element:
                el.elstiff = el.CV
                el.calc_Kel()
                el.eps = np.zeros(6)
                el.sig = np.zeros(6)
                el.epl = np.zeros(6)
            bcr0 = 0.
            bct0 = 0.
            self.bct_mem = 0.
            self.bcr_mem = 0.
        else:
            bcr0 = self.bcr_mem
            bct0 = self.bct_mem
        ul = self.uleft
        ub = self.ubot
        K = self.setupK()  # assemble system stiffness matrix from element stiffness matrices
        
        'define loop for external load steps (BC subdivision)'
        'during each load step mechanical equilibrium is calculated for sub-step'
        'the tangent stiffness matrix of the last load step is used as initial guess'
        'current tangent stiffness matrix compatible with BC is determined iteratively'
        il = 0
        bc_inc = True
        while bc_inc:
            'define global increments for boundary conditions'
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
            'calculate du and df fulfilling mech. equil. for max. load step consistent with stiffness matrix K'
            dbcr = max_dbcr
            dbct = max_dbct
            self.du, df, ind = calc_BC(K, ul, ub, dbcr, dbct) # consider BC for system of equ. 
            self.du[ind] = np.linalg.solve(Kred(ind), df[ind]) # Solve reduced system of equations
            'calculate scaling factor for predictor step in case of non-linear model'
            if self.nonlin:
                scale_bc = calc_scf() # calculate global predictor step to hit the yield surface
                if verb:
                    print('##scaling factor',scale_bc)
                dbcr = max_dbcr*scale_bc  # dbcr >= self.bcr - bcr0
                dbct = max_dbct*scale_bc  # dbct <= self.bct - bct0
                'calculate du and df fulfilling for scaled load step consistent with stiffness matrix K'
                self.du, df, ind = calc_BC(K, ul, ub, dbcr, dbct) # consider BC for system of equ.  
                self.du[ind] = np.linalg.solve(Kred(ind), df[ind]) # Solve reduced system of equations
                f = calc_el_yldfct() # evaluate elemental yld.fct. and  modify elem. stiffness matrix if required
                conv = np.all(f<=ptol)
                i = 0
                if verb:
                    print('**Load step #',il)
                    fres = K@(self.u+self.du)
                    print('yield function=',f,'residual forces on inner nodes=',fres[jin])
                while not conv:
                    'if not converged with predictor step and initial stiffness matrix,'
                    'use tangent stiffness assigned to elements in "calc_el_yldfct"'
                    'same step size is used for initial predictor step'
                    K = self.setupK()  # assemble tangent stiffness matrix
                    self.du, df, ind = calc_BC(K, ul, ub, dbcr, dbct)
                    self.du[ind] = np.linalg.solve(Kred(ind), df[ind]) # solve du with current stiffness matrix
                    f = calc_el_yldfct()
                    conv = np.all(f<=ptol)
                    if verb:
                        print('+++Inner load step #',i)
                        fres = K@(self.u+self.du)
                        print('load increment right:', dbcr)
                        print('load increment top:',dbct)
                        print('yield function=',f,'residual forces on inner nodes=',fres[jin])
                        el = self.element[0]
                        print('sig, disg, epl, depl, deps:',el.sig, el.dsig(), el.epl, el.depl(), el.deps())
                    if i>7 and not conv:
                        print('\n conv,i,f,ptol,dbcr,dbct',conv,i,f,ptol,dbcr,dbct)
                        sys.exit('Error: No convergence achieved in plasticity routine')
                    if not conv:
                        dbcr *= 0.25
                        dbct *= 0.25
                    i += 1
            'update internal variables with results of load step'
            self.u += self.du
            self.f += K@self.du
            for el in self.element:
                el.eps = el.eps_t()
                el.epl += el.depl()
                el.sig += el.dsig()
                if el.Mat.msparam is not None:
                    peeq = eps_eq(el.epl)
                    el.Mat.set_workhard(peeq)
            'update load step'
            il += 1
            bcr0 += dbcr
            hl = np.abs(bcr0-self.bcr)>1.e-6 and np.abs(self.bcr)>1.e-9
            if self.dim>1:
                bct0 += dbct
                hr = np.abs(bct0-self.bct)>1.e-6 and np.abs(self.bct)>1.e-9
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
        
        Yields
        ------
        Model.glob : dictionary
            Values for stress ('sig'), total strain ('eps') and plastic strain ('epl')
            as homogenized element solutions (Voigt tensors); values for (11)- and 
            (22)-components of stress ('sbc1','scb2') and total strain ('ebc1', 'ebc2')
            as homogenized values of boundary nodes.
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
        stress1  : 
            horizontal stress component
        stress2  : 
            vertical stress component
        plastic1 :
            plastic strain in horizontal direction
        plastic2 :
            plastic strain in vertical direction
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
        '''
        fig, ax = plt.subplots(1)
        cmap = mpl.cm.get_cmap(colormap, cdepth)
        def strain1():
            hh = [el.eps[0]*100 for el in self.element]
            text_cb = '$\epsilon^\mathrm{tot}_{11}$ (%)'
            return hh, text_cb
        def strain2():
            hh = [el.eps[1]*100 for el in self.element]
            text_cb = '$\epsilon^\mathrm{tot}_{22}$ (%)'
            return hh, text_cb
        def stress1():
            hh = [el.sig[0] for el in self.element]
            text_cb = '$\sigma_{11}$ (MPa)'
            return hh, text_cb
        def stress2():
            hh = [el.sig[1] for el in self.element]
            text_cb = '$\sigma_{22}$ (MPa)'
            return hh, text_cb
        def plastic1():
            hh = [el.epl[0]*100 for el in self.element]
            text_cb = '$\epsilon^\mathrm{pl}_{11}$ (%)'
            return hh, text_cb
        def plastic2():
            hh = [el.epl[1]*100 for el in self.element]
            text_cb = '$\epsilon^\mathrm{pl}_{22}$ (%)'
            return hh, text_cb
        def stress_eq():
            hh = [Stress(el.sig).seq(el.Mat) for el in self.element]
            text_cb = '$\sigma_{eq}$ (MPa)'
            return hh, text_cb
        def stress_eqJ2():
            hh = [Stress(el.sig).sJ2() for el in self.element]
            text_cb = '$\sigma^\mathrm{J2}_{eq}$ (MPa)'
            return hh, text_cb
        def strain_peeq():
            hh = [eps_eq(el.epl)*100 for el in self.element]
            text_cb = '$\epsilon^\mathrm{pl}_{eq}$ (%)'
            return hh, text_cb
        def strain_etot():
            hh = [eps_eq(el.eps)*100 for el in self.element]
            text_cb = '$\epsilon^\mathrm{tot}_{eq}$ (%)'
            return hh, text_cb
        def disp_x():
            hh = [el.eps[0]*self.lenx for el in self.element]
            text_cb = '$u_x$ (mm)'
            return hh, text_cb
        def disp_y():
            hh = [el.eps[1]*self.leny for el in self.element]
            text_cb = '$u_y$ (mm)'
            return hh, text_cb
        field={
            'strain1' : strain1(),
            'strain2' : strain2(),
            'stress1' : stress1(),
            'stress2' : stress2(),
            'plastic1': plastic1(),
            'plastic2': plastic2(),
            'seq'     : stress_eq(),
            'seqJ2'   : stress_eqJ2(),
            'peeq'    : strain_peeq(),
            'etot'    : strain_etot(),
            'ux'      : disp_x(),
            'uy'      : disp_y()
        }
        
        'define color value by mapping field value of element to interval [0,1]'
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
        
        'create element plots'
        for el in self.element:
            # draw filled polygon for each element
            if (self.dim==1):
                ih = np.amin(el.nodes)       # left node of current element
                jh = np.amax(el.nodes)    # right node of current element
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
        cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label(text_cb)
        'add axis annotations'
        if annot:
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
        'save plot to file if filename is provided'
        if file is not None:
            fig.savefig(file+'.pdf', format='pdf', dpi=300)
        plt.show()
