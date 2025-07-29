# -*- coding: utf-8 -*-

"""Top-level package for pyLabFEA"""

from pylabfea.basic import Strain, Stress, a_vec, b_vec, yf_tolerance, \
    eps_eq, sig_polar_ang, yf_tolerance, sig_princ2cyl, \
    sig_eq_j2, sig_cyl2princ, sig_cyl2voigt, sig_princ, \
    pickle2mat, sig_dev, sig_spherical_to_cartesian, \
    seq_J2, sprinc, sp_cart, svoigt, s_cyl, sdev  # legacy

from pylabfea.model import Model
from pylabfea.material import Material
from pylabfea.data import Data, find_transition_index, get_elastic_coefficients
from pylabfea.training import load_cases, training_score, create_test_sig
from importlib.metadata import version

__author__ = """Alexander Hartmaier, Ronak Shoghi, Jan Schmidt"""
__email__ = 'alexander.hartmaier@rub.de'
__version__ = version('pylabfea')
