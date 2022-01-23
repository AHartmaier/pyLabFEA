import pytest
import os
import pylabfea as FE
import numpy as np

def test_material():
    #check if the elastic values are assigned correctly
    assert np.abs(mat1.C11 - 160493.8271604938) < 1E-5
    assert np.abs(mat1.C12 - 86419.75308641973) < 1E-5
    assert np.abs(mat1.C44 - 37037.03703703704) < 1E-5 

def test_model():
    #check if linear FEA model produces correct result
    assert np.abs(voigt_stiff - mod_stiff) < 1E-5
    assert np.abs(fem2.glob['sig'][1] - fem2.glob['sbc2']) < 1E-5
    assert np.abs(fem2.glob['eps'][1] - fem2.glob['ebc2']) < 1E-5
    assert np.abs(fem2.glob['epl'][1] - 0.04966042764325635) < 1E-5

def test_plasticity():
    #check if nonlinear stress-strain data is correct
    assert np.abs(mat2.propJ2['stx']['ys'] - 146.38501094227996) < 1E-5
    assert np.abs(mat2.propJ2['sty']['seq'][-1] - 168.5141123395444) < 1E-5
    assert np.abs(mat2.propJ2['sty']['peeq'][-1] - 0.04969421741530513) < 1E-5
    assert np.abs(mat2.propJ2['et2']['ys'] - 136.93063937629154) < 1E-5
    assert np.abs(mat2.propJ2['ect']['peeq'][-1] - 0.04570405456408677) < 1E-5
    assert np.abs(mat2.propJ2['ect']['seq'][-1] - 168.3199594723871) < 1E-5
    
def test_workhard():
    # check if work hardening is implemented correctly
    assert np.abs(mat3.propJ2['stx']['seq'][-1] - 347.68211920529546) < 1E-5
    assert np.abs(mat3.propJ2['sty']['peeq'][-1] - 0.09883666666666659) < 1E-5
    assert np.abs(mat3.sigeps['et2']['sig'][-1][0] - 307.1334214002634) < 1E-5
    assert np.abs(mat3.sigeps['ect']['sig'][-1][0] + 192.15895530336059) < 1E-5
    
def test_hill_6p():
    # check if sdim=6 and hill_6p works fine
    assert np.abs(mat4.propJ2['stx']['peeq'][-1] - 0.05039661) < 1E-5
    assert np.abs(mat4.propJ2['sty']['seq'][-1] - 114.28035811) < 1E-5
    assert np.abs(mat4.sigeps['et2']['sig'][-1][1] - 102.534840) < 1E-5
    assert np.abs(mat4.sigeps['ect']['sig'][-1][0] + 54.6031702) < 1E-5
    
def test_bcnode():
    # setup material definition for regular mesh
    NX=18
    NY=18
    NXi1 = int(NX/3)
    NXi2 = 2*NXi1
    NYi1 = int(NY/3)
    NYi2 = 2*NYi1
    el = np.ones((NX, NY))
    el[NXi1:NXi2, NYi1:NYi2] = 2

    # define materials
    mat1 = FE.Material(num=1)            # call class to generate material object
    mat1.elasticity(E=100.e3, nu=0.27)   # define elastic properties
    mat2 = FE.Material(num=2)            # define second material
    mat2.elasticity(E=3.e3, nu=0.3)      # material is purely elastic

    # setup model for elongation in y-direction
    fe = FE.Model(dim=2, planestress=False)   # initialize finite element model
    fe.geom(sect=2, LX=4., LY=4.) # define geometry with two sections
    fe.assign([mat1, mat2])       # assign materials to sections

    #boundary conditions: uniaxial stress in longitudinal direction                  
    fe.bcbot(0.)                    # fix bottom boundary
    fe.bcright(0., 'force')           # boundary condition on lateral edges of model
    fe.bcleft(0., 'force')
    fe.bctop(0.01*fe.leny, 'disp')  # strain applied to top nodes

    # meshing and plotting of model
    fe.mesh(elmts=el, NX=NX, NY=NY)  # create regular mesh with sections as defined in el
    # fix lateral displacements of corner node to prevent rigig body motion
    hh = [no in fe.nobot for no in fe.noleft]
    noc = np.nonzero(hh)[0]  # find corner node
    fe.bcnode(noc, 0., 'disp', 'x')  # fix laterial displacement
    fe.solve()
    assert np.abs(fe.u[684] + 9.730777232237817e-3) < 1E-5
    assert np.abs(fe.element[0].sig[5] - 2.2990816342732256) < 1E-5
    assert np.abs(fe.element[5*NY+7].sig[0] - 45.68020736256676) < 1E-5
    assert np.abs(fe.element[6*NY+7].sig[1] - 69.16252458086865) < 1E-5
    assert noc == [0]
    
#define model for elasticity tests
fem_v = FE.Model(dim=2, planestress=True)   # call class to generate container for finite element model
fem_v.geom([2, 1, 2, 1, 2], LY=4.) # define sections in absolute lengths
mat1 = FE.Material()               # call class to generate container for material
mat1.elasticity(E=100.e3, nu=0.35)   # define materials by their elastic propertiess
mat2 = FE.Material()               # define second material
mat2.elasticity(E=300.e3, nu=0.3)
fem_v.assign([mat1, mat2, mat1, mat2, mat1])  # assign the proper material to each section
fmat1 = 6./8.   # calculate volume fraction of each material
fmat2 = 2./8.
#boundary conditions: uniaxial stress in longitudinal direction
fem_v.bcleft(0.)                     # fix left and bottom boundary
fem_v.bcbot(0.)
fem_v.bcright(0., 'force')           # free boundary condition on right edge of model
fem_v.bctop(0.1*fem_v.leny, 'disp')  # strain applied to top nodes
#solution and evalutation of effective properties
fem_v.mesh(NX=16, NY=4) # create mesh
fem_v.solve()           # solve system of equations
fem_v.calc_global()     # calculate global stress and strain
mod_stiff = fem_v.glob['sig'][1]/fem_v.glob['eps'][1]     # effective stiffness of numerical model
voigt_stiff = fmat1*mat1.E + fmat2*mat2.E # Voigt stiffness: weighted average of Young's moduli wrt volume fractions

fem2 = FE.Model(dim=2, planestress=False)   # call class to generate container for finite element model
fem2.geom([2, 2], LY=4.) # define sections in absolute lengths
mat2.plasticity(sy=150.,khard=500.,sdim=3)   # define material with isotropic plasticity
fem2.assign([mat1, mat2])  # assign the proper material to each section
#boundary conditions: uniaxial stress in longitudinal direction
fem2.bcleft(0.)                     # fix left and bottom boundary
fem2.bcbot(0.)
fem2.bcright(0., 'force')           # free boundary condition on right edge of model
fem2.bctop(0.1*fem2.leny, 'disp')  # strain applied to top nodes
#solution and evalutation of effective properties
fem2.mesh(NX=4, NY=4) # create mesh
fem2.solve()           # solve system of equations
fem2.calc_global()     # calculate global stress and strain

# add plastic properties for further tests
mat2.plasticity(sy=150., hill=[0.7,1.,1.4], khard=100., sdim=3)   # define material with ideal isotropic plasticity
mat2.calc_properties(eps=0.05) # calculate the stress-strain curves up to a total strain of 5%

# test work hardening implementation
mat3 = FE.Material()               # define  material
mat3.elasticity(E=300.e3, nu=0.3)
mat3.plasticity(sy=150., khard=2000.)   # define material with ideal isotropic plasticity
mat3.calc_properties(eps=0.1, sigeps=True) #, sigeps=True, min_step=2, verb=True) # calculate the stress-strain curves up to a total strain of 5%

#test plasticity for full stress tensor
mat4 = FE.Material()               # define  material
mat4.elasticity(E=200.e3, nu=0.3)
mat4.plasticity(sy=100., hill=[0.7,1.,1.4,1.,1.2,0.8], khard=100., sdim=6)
mat4.calc_properties(eps=0.05, sigeps=True)
