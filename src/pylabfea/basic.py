# Module pylabfea.basic
"""Module pylabfea.basic introduces basic methods and attributes like calculation of
equivalent stresses and strain, conversions of
cylindrical to principal stresses and vice versa, and the classes ``Stress`` and ``Strain``
for efficient operations with these quantities.

uses NumPy, pickle

Version: 4.0 (2021-11-27)
Author: Alexander Hartmaier, ICAMS/Ruhr University Bochum, Germany
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)"""

import numpy as np
import pickle

# ===================================
# define global methods and variables
# ===================================
a_vec = np.array([1., -0.5, -0.5]) / np.sqrt(1.5)
"""First unit vector spanning deviatoric stress plane (real axis)"""

b_vec = np.array([0., 0.5, -0.5]) * np.sqrt(2)
"""Second unit vector spanning deviatoric stress plane (imaginary axis)"""

yf_tolerance = 5.e-3
"""Tolerance: Plastic yielding if yield function > yf_tolerance"""


def sig_eq_j2(sig: np.ndarray):
    """Calculate sj2 equivalent stress from any stress tensor

    Parameters
    ----------
    sig : (3,), (6,) (N,3) or (N,6) array
         (3,), (N,3): Principal stress or list of principal stresses;
         (6,), (N,6): Voigt stress

    Returns
    -------
    seq : float or (N,) array
        sj2 equivalent stresses
    """
    nsc: int = len(sig)  # number of stress components
    sh = np.shape(sig)
    if sh == (3,):
        sp = np.array([sig])
    elif sh == (6,):
        sp = np.array([sig_princ(sig)[0]])
    elif sh == (nsc, 6):
        sp = sig_princ(sig)[0]
    elif sh == (nsc, 3):
        sp = np.array(sig)
    else:
        raise TypeError('Error: Unknown format of stress in sig_eq_j2: nsc={nsc}, sh={sh}')
    d12 = sp[:, 0] - sp[:, 1]
    d23 = sp[:, 1] - sp[:, 2]
    d31 = sp[:, 2] - sp[:, 0]
    sj2 = 0.5 * (np.square(d12) + np.square(d23) + np.square(d31))
    seq = np.sqrt(sj2)  # sj2 eqiv. stress
    if sh == (3,) or sh == (6,):
        seq = seq[0]
    return seq


def sig_polar_ang(sig: np.ndarray):
    """Transform stresses into polar angle on deviatoric plane spanned by a_vec and b_vec

    Parameters
    ----------
    sig : (3,), (6,) (N,3) or (N,6) array
         (3,), (N,3): Principal stresses;
         (6,), (N,6): Voigt stress

    Returns
    -------
    theta : float or (N,) array
        polar angles in deviatoric plane as positive angle between sig and a_vec in range [-pi,+p]
    """
    sh = np.shape(sig)
    nsc = len(sig)
    if sh == (3,):
        sp = np.array([sig])
    elif sh == (6,):
        sp = [sig_princ(sig)[0]]
    elif sh == (nsc, 6):
        sp = sig_princ(sig)[0]
    elif sh == (nsc, 3):
        sp = np.array(sig)
    else:
        raise TypeError(f'Error: Unknown format of stress in polar_angle: : nsc={nsc}, sh={sh}')
    hyd = np.sum(sp, axis=1) / 3.  # hydrostatic component
    dev = sp - hyd[:, None]  # deviatoric princ. stress
    vn = np.linalg.norm(dev, axis=1)  # norms of princ. stress vectors
    ind = np.nonzero(vn < 1.e-4)[0]
    vn[ind] = 1.
    dsa = np.dot(dev / vn[:, None], a_vec)
    dsb = np.dot(dev / vn[:, None], b_vec)
    theta = np.angle(dsa + 1j * dsb)
    if sh == (3,) or sh == (6,):
        theta = theta[0]
    return theta


def sig_princ(sig: np.ndarray):
    """Convert Voigt stress tensors into principal stresses and eigenvectors.

    Parameters
    ----------
    sig : (6,), (nsc,6), (3,3), or (nsc,3,3) array
        Voigt stress tensor (dim=6) or Cartesian stress tensor (dim=3x3)

    Returns
    -------
    spa : (3,) or (nsc,3) array
        Principal stresses
    eva : (3,3) or (nsc,3,3) array
        Eigenvectors/rotation matrices of stress tensor
    """
    nsc = len(sig)  # number of stress components
    sh = np.shape(sig)
    if sh == (3, 3):
        nsc = 1  # sig is Cartesian single stress tensor
        st = np.array([sig])
    elif sh == (nsc, 3, 3):
        st = np.array(sig)  # sig is array of Cart. stress tensors
    elif sh == (6,):
        nsc = 1  # sig is single Voigt stress
        st = np.zeros((1, 3, 3))
        st[0, 0, 0] = sig[0]
        st[0, 1, 1] = sig[1]
        st[0, 2, 2] = sig[2]
        st[0, 2, 1] = st[0, 1, 2] = sig[3]
        st[0, 2, 0] = st[0, 0, 2] = sig[4]
        st[0, 1, 0] = st[0, 0, 1] = sig[5]
    elif sh == (nsc, 6):
        st = np.zeros((nsc, 3, 3))
        for i in range(nsc):
            st[i, 0, 0] = sig[i, 0]
            st[i, 1, 1] = sig[i, 1]
            st[i, 2, 2] = sig[i, 2]
            st[i, 2, 1] = st[i, 1, 2] = sig[i, 3]
            st[i, 2, 0] = st[i, 0, 2] = sig[i, 4]
            st[i, 1, 0] = st[i, 0, 1] = sig[i, 5]
    else:
        raise TypeError('Error: Unknown format of stress in sig_princ: nsc={nsc}, sh={sh}')

    # calculate principal stresses and eigen vectors
    spa = np.zeros((nsc, 3))
    eva = np.zeros((nsc, 3, 3))
    for n in range(nsc):
        sp, ev = np.linalg.eig(st[n])  # solve eigenvalue problem
        # arrange principal stress components according to major force axes
        iev = np.argmax(np.abs(ev), axis=1)
        j = np.zeros(3, dtype=int)
        i0 = [i for i, x in enumerate(iev) if x == 0]  # positions of indices 0
        i1 = [i for i, x in enumerate(iev) if x == 1]
        i2 = [i for i, x in enumerate(iev) if x == 2]
        k0 = len(i0)
        for i in range(k0):
            j[i] = i0[i]
        for i in range(len(i1)):
            j[k0 + i] = i1[i]
        k0 += len(i1)
        for i in range(len(i2)):
            j[k0 + i] = i2[i]
        ev = np.array([ev[j[0], :], ev[j[1], :], ev[j[2], :]])
        sp = np.array([sp[j[0]], sp[j[1]], sp[j[2]]])
        # ensure positive determinant
        if np.linalg.det(ev) < 0:
            ev *= -1
        spa[n, :] = sp
        eva[n, :, :] = ev
    if sh == (3, 3) or sh == (6,):
        spa = spa[0]
        eva = eva[0, :, :]
    return spa, eva


def sig_cyl2princ(s_cyl) -> np.ndarray:
    """Convert cylindrical stress into 3D Cartesian principle stress

    Parameters
    ----------
    s_cyl : (2,), (3,), (N,2) or (N,3) array
         Cylindrical stress in form (seq, theta, (optional: p))

    Returns
    -------
    s_princ : (3,) or (N,3) array
        principle deviatoric stresses
    """
    sh = np.shape(s_cyl)
    if sh == (2,) or sh == (3,):
        s_cyl = np.array([s_cyl])
    seq = s_cyl[:, 0]
    theta = s_cyl[:, 1]
    s_princ = (np.tensordot(np.cos(theta), a_vec, axes=0) +
               np.tensordot(np.sin(theta), b_vec, axes=0)) * \
              np.sqrt(2. / 3.) * np.array([seq, seq, seq]).T
    if sh[0] == 3:
        p = s_cyl[:, 2]
        s_princ += np.array([p, p, p]).T / 3.
    if sh == (2,) or sh == (3,):
        s_princ = s_princ[0]
    return s_princ


def sig_cyl2voigt(sig_cyl: np.ndarray, eigen_vector: np.ndarray) -> np.ndarray:
    """Convert cylindrical stress and eigenvectors into Voigt stress tensor

    Parameters
    ----------
    sig_cyl : (3,) or (N,3) array
        Cylindrical stress in form (seq, theta, p)
    eigen_vector : (3,3) or (N,3,3) array
        Eigenvectors of stress tensor

    Returns
    -------
    sig_voigt : (6,) or (N,6) array
        Voigt stress tensor
    """
    sp: np.ndarray = sig_cyl2princ(sig_cyl)
    if np.linalg.det(eigen_vector) < 0:
        eigen_vector *= -1  # enforce right-handed system of eigenvectors
    st = np.diag(sp)  # diagonal matrix of princ. stresses
    hh = eigen_vector @ st @ eigen_vector.T  # rotate back into original stress frame
    sig_voigt = np.array([hh[0, 0], hh[1, 1], hh[2, 2], hh[1, 2], hh[0, 2], hh[0, 1]])

    return sig_voigt


def sig_princ2cyl(sig: np.ndarray, mat=None) -> np.ndarray:
    """Convert principal stress into cylindrical stress vector

    Parameters
    ----------
    sig : (3,), (6,), (N,3) or (N,6) array
         stress to be converted, if (3,) or (N,3) principal stress is assumed
    mat : object of class ``Material``
        Material for Hill-type principal stress (optional)

    Returns
    -------
    sc : (3,) or (N,3) array
        stress in cylindrical coordinates (seq, theta, p)
    """
    sh = np.shape(sig)
    nsc: int = len(sig)
    if sh == (3,):
        nsc = 1  # sig is single principle stress vector
        sp = np.array([sig])
        sig = np.array([[sig[0], sig[1], sig[2], 0, 0, 0]])
    elif sh == (nsc, 3):
        sp = np.array(sig)
        sig = np.append(sig, np.zeros((nsc, 3)), axis=1)
    elif sh == (6,):
        nsc = 1
        sp = np.array([Stress(sig).princ])
        sig = np.array([sig])
    elif sh == (nsc, 6):
        sp = sig_princ(sig)[0]
    else:
        raise TypeError(f'Error in s_cyl: Format not supported (N={nsc}, sh={sh})')
    sc = np.zeros((nsc, 3))
    if mat is None:
        sc[:, 0] = sig_eq_j2(sp)
    else:
        sc[:, 0] = mat.calc_seq(sig)
    sc[:, 1] = sig_polar_ang(sp)
    sc[:, 2] = np.sum(sp, axis=1) / 3.
    if sh == (3,) or sh == (6,):
        sc = sc[0]
    return sc


def sig_dev(sig: np.ndarray) -> np.ndarray:
    """Calculate deviatoric stress component from given stress tensor

    Parameters
    ----------
    sig : (3,), (6,) (N,3) or (N,6) array

    Returns
    -------
    sd : float or (N,) array
        deviatoric stresses
    """
    sh = np.shape(sig)
    hyd = np.zeros(sh)
    if sh == (3,) or sh == (6,):
        p = np.sum(sig[0:3]) / 3.
        hyd[0:3] = p[None]
    else:
        p = np.sum(sig[:, 0:3], axis=1) / 3.
        hyd[:, 0:3] = p[:, None]
    sd = sig - hyd
    return sd


def eps_eq(eps: np.ndarray):
    """Calculate equivalent strain

    Parameters
    ----------
    eps : (3,), (6,), (nsc,3) or (nsc,6) array
         (3,) or (nsc,3): Principal strains;
         (6,) or (nsc,6): Voigt strains

    Returns
    -------
    eeq : float or (nsc,) array
        equivalent strains
    """
    sh = np.shape(eps)
    if sh == (6,) or sh == (3,):
        eps = np.array([eps])
        nsc = 1
    else:
        nsc: int = len(eps)
    # ev = np.sum(eps[:,0:3],axis=1)
    # ed = eps[:,0:3] - np.array([ev,ev,ev]).T
    if sh == (6,) or sh == (nsc, 6):
        eeq = np.sqrt(
            2. * (np.sum(eps[:, 0:3] * eps[:, 0:3], axis=1) + 0.5 * np.sum(eps[:, 3:6] * eps[:, 3:6], axis=1)) / 3.)
    elif sh == (3,) or sh == (nsc, 3):
        eeq = np.sqrt(2. * np.sum(eps[:, 0:3] * eps[:, 0:3], axis=1) / 3.)
    else:
        raise ValueError(f'Error in eps_eq: Format not supported: nsc={nsc},sh={sh}')

    if sh == (6,) or sh == (3,):
        eeq = eeq[0]
    return eeq


# =========================
# define class for stresses
# =========================
class Stress(object):
    """Stores and converts Voigt stress tensors into different formats,
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
    """

    def __init__(self, sv):
        self.v = self.voigt = np.array(sv)
        # calculate (3x3)-tensorial representation
        self.t = self.tens = np.zeros((3, 3))
        self.tens[0, 0] = sv[0]
        self.tens[1, 1] = sv[1]
        self.tens[2, 2] = sv[2]
        self.tens[2, 1] = self.tens[1, 2] = sv[3]
        self.tens[2, 0] = self.tens[0, 2] = sv[4]
        self.tens[1, 0] = self.tens[0, 1] = sv[5]
        # calculate principal stresses and eigen vectors
        self.princ, self.evec = sig_princ(self.tens)
        self.p = self.princ
        self.h = self.hydrostatic = np.sum(self.p) / 3.
        self.d = self.dev = sv - np.array([self.h, self.h, self.h, 0., 0., 0.])

    def seq(self, mat=None):
        """calculate Hill-type equivalent stress, invokes corresponding method of class ``Material``

        Parameters
        ----------
        mat: object of class ``Material``
            contains Hill parameters and method needed for Hill-type equivalent stress
            (optional, default=None)

        Returns
        -------
        seq : float
            equivalent stress; material specific (J2, Hill, Tresca, Barlat) if mat is provided,
            J2 equivalent stress otherwise
        """
        if mat is None:
            seq = sig_eq_j2(self.p)  # give princ. stress as parameter to avoid re-calculation
        else:
            seq = mat.calc_seq(self.v)
        return seq

    def theta(self):
        """Calculate polar angle in deviatoric plane

        Returns
        -------
        ang : float
            polar angle of stress in deviatoric plane
        """
        ang = sig_polar_ang(self.p)
        return ang

    def seq_j2(self):
        """Calculate J2 principal stress

        Returns
        -------
        sj2 : float
            equivalent stress
        """
        sj2 = sig_eq_j2(self.p)
        return sj2

    def cyl(self):
        """Calculate cylindrical stress tensor

        Returns
        -------
        cyl : (3,) array
            stress in cylindrical form: (J2 eqiv. stress, polar angle, hydrostatic)
        """
        cyl = np.array([sig_eq_j2(self.p), sig_polar_ang(self.p), self.h])
        return cyl

    def lode_ang(self, arg):
        """Calculate Lode angle:
        Transforms principal stress space into hydrostatic stress, eqiv. stress, and Lode angle;
        definition of positive cosine for Lode angle is applied

        Parameters
        ----------
        arg : either float or object of class ``Material``
            if type is float: interpreted as equivalent stress
            if type is ``Material``: method of that class is used
            to calculate equivalent stress

        Returns
        -------
        la : float
            Lode angle
        """
        if type(arg) is float:
            seq = arg  # float-type parameters are interpreted as equiv. stress
        else:
            seq = self.seq(arg)  # otherwise parameter is Material
        j3 = np.linalg.det(self.tens - self.h * np.diag(np.ones(3)))
        hh = 0.5 * j3 * (3. / seq) ** 3
        la = np.arccos(hh) / 3.
        return la


# =======================
# define class for strain
# =======================
class Strain(object):
    """Stores and converts Voigt strain tensors into different formats,
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
    """

    def __init__(self, sv):
        self.v = self.voigt = np.array(sv)
        # calculate (3x3)-tensorial representation
        self.t = self.tens = np.zeros((3, 3))
        self.tens[0, 0] = sv[0]
        self.tens[1, 1] = sv[1]
        self.tens[2, 2] = sv[2]
        self.tens[2, 1] = self.tens[1, 2] = sv[3]
        self.tens[2, 0] = self.tens[0, 2] = sv[4]
        self.tens[1, 0] = self.tens[0, 1] = sv[5]
        # calculate principal stresses and eigen vectors
        self.princ, self.evec = np.linalg.eig(self.tens)
        self.p = self.princ

    def eeq(self):
        """Calculate equivalent strain

        Returns
        -------
        equiv : float
            Equivalent strain
        """
        equiv = eps_eq(self.v)
        return equiv

    def inv(self):
        """Calculate inverse of strain tensor ignoring zeros.

        Returns
        -------
        inv : (6,) array
        """
        inv = np.zeros(6)
        for i in range(6):
            if np.abs(self.voigt[i]) > 1.e-9:
                inv[i] = 1. / self.voigt[i]
        return inv


# =================================================
# define subroutines for reading objects from files
# =================================================
def pickle2mat(name, path='./'):
    """Read pickled material file.


    Parameters
    ----------
    name : string
        File name of pickled material to be read.
    path : string
        Path under which pickle-files are stored (optional, default: './')

    Returns
    -------
    pcl : Material object
        unpickled material

    """
    if name is None:
        raise ValueError('Name for pickled material must be given.')
    if path[-1] != '/':
        path += '/'
    with open(path + name, 'rb') as inp:
        pcl = pickle.load(inp)
    return pcl


# =================================================
# Alias functions to ensure backwards compatibility
# with legacy versions of pyLabFEA
# THESE FUNCTIONS SHOULD NOT BE USED ANY MORE!
# =================================================
def seq_J2(sig):
    return sig_eq_j2(sig)


def sprinc(sig):
    return sig_princ(sig)


def sp_cart(scyl):
    return sig_cyl2princ(scyl)


def svoigt(scyl, evec):
    return sig_cyl2voigt(scyl, evec)


def s_cyl(sig, mat=None):
    return sig_princ2cyl(sig, mat)


def sdev(sig):
    return sig_dev(sig)


def polar_ang(sig):
    return sig_polar_ang(sig)     
