#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyLabFEA example of square-shaped elastic inclusion in elastic-plastic matrix
under uniaxial strain

Inclusion can be stiffer or more compliant than matrix to observe differences
in stress and strain fields.

Lateral sides can be fixed or free.

Author: Alexander Hartmaier
ICAMS / Ruhr University Bochum, Germany

December 2024

Published as part of pyLabFEA package under GNU GPL v3 license
"""
from pylabfea import Model, Material
import numpy as np

# boundary conditions
sides = 'force'  # free sides, change to 'disp' for fixed lateral sides
eps_tot = 0.01  # total strain in y-direction

# elastic material parameters
E1 = 100.e3  # Young's modulus of matrix
E2 = 3.e3  # Young's modulus of inclusion, change to 300.e3 for stiff inclusion

# setup material definition for regular mesh
NX = NY = 18
NXi1 = int(NX / 3)
NXi2 = 2 * NXi1
NYi1 = int(NY / 3)
NYi2 = 2 * NYi1
el = np.ones((NX, NY))
el[NXi1:NXi2, NYi1:NYi2] = 2

# define materials
mat1 = Material(num=1)  # call class to generate material object
mat1.elasticity(E=E1, nu=0.27)  # define elastic properties
mat1.plasticity(sy=150., khard=500., sdim=6)  # isotropic plasticity
mat2 = Material(num=2)  # define second material
mat2.elasticity(E=E2, nu=0.3)  # material is purely elastic

# setup model for elongation in y-direction
fe = Model(dim=2, planestress=False)  # initialize finite element model
fe.geom(sect=2, LX=4., LY=4.)  # define geometry with two sections
fe.assign([mat1, mat2])  # assign materials to sections

# boundary conditions: uniaxial stress in longitudinal direction
fe.bcbot(0.)  # fix bottom boundary
fe.bcright(0., sides)  # boundary condition on lateral edges of model
fe.bcleft(0., sides)
fe.bctop(eps_tot * fe.leny, 'disp')  # strain applied to top nodes

# meshing and plotting of model
fe.mesh(elmts=el, NX=NX, NY=NY)  # create regular mesh with sections as defined in el
if sides == 'force':
    # fix lateral displacements of corner node to prevent rigid body motion
    hh = [no in fe.nobot for no in fe.noleft]
    noc = np.nonzero(hh)[0]  # find corner node
    fe.bcnode(noc, 0., 'disp', 'x')  # fix lateral displacement
fe.plot('mat', mag=1, shownodes=False)

# find solution and plot stress and strain fields
fe.solve()  # calculate mechanical equilibrium under boundary conditions
fe.plot('stress1', mag=4, shownodes=False)
fe.plot('stress2', mag=4, shownodes=False)
fe.plot('seq', mag=4, shownodes=False)
fe.plot('peeq', mag=4, shownodes=False)
