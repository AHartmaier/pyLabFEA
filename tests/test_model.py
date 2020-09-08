import pytest
import os
import pylabfea as FE
import numpy as np

def test_material():
    mat1 = FE.Material()
    mat1.elasticity(E=200.e3, nu=0.3)

    #check if the values are assigned correctly
    assert np.abs(mat1.C11 - 269230.76923076925) < 1E-5
    assert np.abs(mat1.C12 - 115384.61538461539) < 1E-5
    assert np.abs(mat1.C44 - 76923.07692307692) < 1E-5     