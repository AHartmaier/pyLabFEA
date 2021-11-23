# -*- coding: utf-8 -*-

"""Top-level package for pyLabFEA"""
           
from pylabfea.basic import Strain, Stress, a_vec, b_vec, \
                           eps_eq, polar_ang, ptol, s_cyl, \
                           seq_J2, sp_cart, svoigt, sprinc, \
                           pckl2mat, sdev
from pylabfea.model import Model
from pylabfea.material import Material
from pylabfea.data import Data
from pkg_resources import get_distribution

__author__ = """Alexander Hartmaier"""
__email__ = 'alexander.hartmaier@rub.de'
__version__ = get_distribution('pylabfea').version

