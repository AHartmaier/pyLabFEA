# Module pylabfea.data
"""Module pylabfea.data introduces the class ``Data`` for handling of data resulting
from virtual or physical mechanical tests in the pyLabFEA package. This class provides the
methods necessary for analyzing data. During this processing, all information is gathered
that is required to define a material, i.e., all parameters for elasticity, plasticity,
and microstructures are provided from the data. Materials are defined in
module pylabfea.material based on the analyzed data of this module.

uses NumPy, SciPy, MatPlotLib

Version: 4.0 (2021-11-27)
Last Update: (24-04-2023)
Authors: Ronak Shoghi, Alexander Hartmaier, ICAMS/Ruhr University Bochum, Germany
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)"""

import json
import random
import warnings
import pylabfea as FE
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize


def find_transition_index(stress):
    """Calculates the index at which a significant transition in the total stress-strain relationship occurs.
    The function applies a Savitzky-Golay filter to smooth the stress data and to calculate the first
    and second derivatives. It identifies the transition index by finding the point where the second
    derivative starts to change from zero, i.e. the value in the elastic regime. This approach relies on
    the overall behavior of the stress-strain relationship
    rather than focusing on a specific plastic strain threshold (e.g., 0.002%).

    Parameters:
    ----------
    stress: 1-d numpy array
        Array of equivalent stress values along one load path.

    Returns:
    ----------
    idx: int
        The index within the stress array where a significant transition from linear behavior occurs.
    """
    nst = len(stress)
    wl1 = max(5, int(nst / 10))
    wl2 = max(2, int(nst / 50))
    sig_d1 = savgol_filter(stress, window_length=wl1, polyorder=1, deriv=1)
    sig_d2 = savgol_filter(sig_d1, window_length=wl2, polyorder=1, deriv=1)

    i0 = int(nst / 10)
    tol = np.mean(sig_d2[i0:i0 + wl2]) * 1.2
    idx = -1
    iend = int((nst - i0) / wl2) - 1
    for i in range(1, iend):
        mav = np.mean(sig_d2[i0 + i * wl2:i0 + (i + 1) * wl2])
        if np.abs(mav) > tol:
            idx = i0 + i * wl2
            break
    if idx < 0:
        print('Warning: Transition not determined properly')
        idx = i0
    return idx


def get_elastic_coefficients(eps, sig, method='least_square', initial_guess=None):
    """A function to compute the elastic coefficients (stiffness matrix) for a material
    based on stress-strain data. This function supports two methods for determining
    the stiffness matrix: a direct least squares approach and an optimization approach.
    The least squares method is used when 'least_square' is specified, which processes
    any desired stress-strain pairs( minimum must be 4).  The optimization approach,
    used when 'decomposition' method is specified, iteratively adjusts the
    stiffness matrix to minimize an objective function that measures the fit to the
    observed data, subject to physical plausibility constraints.

    Parameters
    ----------
    eps: (N, 6)-array
        Array with Voigt strain tensors at the end of the linear-elastic regime of different load cases.
    sig: (N, 6)-array
    Array of corresponding Voigt stress tensors at the end of the linear elastic regime of different load cases.
    method: str
        Method to be used for calculating the stiffness matrix. Options
        include 'least_square' and 'decomposition'. The 'least_square' method
        calculates the stiffness matrix using a least squares fit to the stress-strain data.
        The 'decomposition' method uses an optimization approach with the
        specified method for interpreting the stiffness matrix from the optimization
        variables. (Optional: default is 'least_square').
    initial_guess: np.ndarray or None
        Initial guess for the stiffness matrix coefficients
        used in the optimization approach. If None, a random initial guess is generated.
        This parameter is ignored when using the 'least_square' method. Default is None.

    Returns:
    C: (6, 6)-array
        The optimized symmetric 6x6 stiffness matrix (elastic coefficients) as a numpy array.
        If the optimization or calculation fails to converge to a solution within the
        maximum number of attempts, the function may return the last attempted solution
        or raise an error, depending on implementation details.
    """

    def map_flat_to_matrix(C_flat):
        """Maps a flat array of coefficients into a symmetric matrix C. This function takes the
        elements from the input array and places them into both the upper and the lower
        triangular portions of the matrix, ensuring symmetry. The operation is particularly
        useful for reconstructing symmetric matrices, such as stiffness or elasticity matrices,
        from a set of parameters.

        Parameters
        ----------
        C_flat: np.ndarray
            A flat array containing the unique coefficients of a symmetric
            matrix. The length of C_flat should be 21, corresponding to the number of unique
            elements in a 6x6 symmetric matrix (6 diagonal + 15 off-diagonal elements).

        Returns:
        C: np.ndarray
            A 6x6 symmetric matrix constructed from the input coefficients.
         """

        C = np.zeros((6, 6))
        indices = np.triu_indices(6)
        C[indices] = C_flat
        C[(indices[1], indices[0])] = C_flat
        return C

        # for Cholesky decomporion: https://en.wikipedia.org/wiki/Cholesky_decomposition

    def map_flat_to_L_and_C(C_flat):
        """Maps a flat array of coefficients into a lower triangular matrix L, and then
        computes a symmetric positive definite matrix C by multiplying L with its transpose.
        This approach is commonly used in the decomposition of a stiffness or elasticity matrix,
        enabling the reconstruction of these matrices from a reduced set of parameters in
        optimization problems.

        Parameters:
        - C_flat (np.ndarray): A flat array of coefficients. The length of this array should
          be such that it can fill the lower triangular part of a 6x6 matrix, it should
          have 21 elements (6+5+4+3+2+1).

        Returns:
        - tuple of np.ndarray: A tuple containing two numpy arrays:
            - L (np.ndarray): The lower triangular matrix formed by placing the elements of `C_flat`
              into the lower triangular indices of a 6x6 matrix.
            - C (np.ndarray): The symmetric positive definite matrix computed as the dot product of L and its transpose.
            """
        L = np.zeros((6, 6))
        indices = np.tril_indices(6)
        L[indices] = C_flat
        C = np.dot(L, L.T)
        return L, C

    def is_positive_definite(C):
        """Checks whether a given matrix is positive definite. A matrix is positive definite
        if all its eigenvalues are greater than zero. This characteristic is essential for
        ensuring the stability and uniqueness of solutions in many mathematical contexts,
        including optimization problems and systems of linear equations.

        Parameters
        ----------
        C: np.ndarray
            The matrix to be checked. This should be a square matrix.

        Returns
        -------
        _: bool:
            True if the matrix is positive definite, meaning all its eigenvalues are
            greater than zero; False otherwise.
        """
        return np.all(np.linalg.eigvals(C) > 0)

    def calculate_stress_from_strain(strain, C):
        """Calculates the predicted stress for a given strain using the material's stiffness
        matrix. This function multiplies the stiffness matrix C by the strain vector to
        predict the resulting stress, according to Hooke's Law for linear elastic materials.

        Parameters
        ----------
        strain: list or np.ndarray
            The strain vector, representing the deformation in the material. This should be a
            1D array or list of length 6, corresponding to the three normal and three shear
            components of strain.
        C: np.ndarray
            The stiffness matrix of the material. This should be a 6x6 symmetric
            matrix representing the material's resistance to deformation. The matrix correlates
            the strain to the stress in the material.

        Returns
        -------
        _: np.ndarray
            The predicted stress vector, calculated as the dot product of the stiffness
            matrix C and the strain vector. This vector has the same dimension as the strain
            input, containing the three normal and three shear components of stress.
        """
        strain_vector = np.array(strain)
        stress_predicted = np.dot(C, strain_vector)
        return stress_predicted

    def objective_function(data_pairs, x_flat, penalty_weight=1e9, lambda_reg=1e-3):
        """Calculates the objective function value for an optimization problem that aims
        to find the stiffness matrix coefficients using the decomposition method
        to construct the stiffness matrix from a flat array of coefficients. It includes
        penalties for non-positive definiteness of the matrix and a regularization term
        to prevent overfitting.

        Parameters
        ----------
        x_flat: np.ndarray
            A flat array of stiffness matrix coefficients.
        penalty_weight: float
            The weight of the penalty for non-positive definiteness. (Optional, default is 1e9).
        lambda_reg: float
            The regularization parameter to prevent overfitting by penalizing
            large coefficients. (Optional, default is 1e-3).

        Returns
        -------
        _: float
            The value of the objective function, which includes the sum of squared residuals
            between observed and predicted stresses, a penalty for non-positive definiteness, and
            a regularization term.
        """
        _, C = map_flat_to_L_and_C(x_flat)
        penalty = 0
        if not is_positive_definite(C):
            penalty = penalty_weight * np.sum(np.min(np.linalg.eigvals(C), 0) ** 2)
        sum_squared_residuals = 0
        for strain, observed_stress in data_pairs:
            predicted_stress = calculate_stress_from_strain(strain, C)
            residuals = observed_stress - predicted_stress
            sum_squared_residuals += np.sum(residuals ** 2)
        regularization_term = lambda_reg * np.sum(x_flat ** 2)
        return sum_squared_residuals + penalty + regularization_term

    def least_square(data_pairs, random_pairs_number=100):
        """Calculates the least squares solution for a set of equations derived from
        a specified number of random experiment pairs. Each pair consists of strains
        and stresses. In the case of general symmetry, the coefficients of the stiffness
        matrix are reduced to 21 from 36, and at least 4 stress-strain pairs (24 equations) are required
        to accurately determine these coefficients. This function solves for the coefficients
        that minimize the difference between the observed stresses and those predicted by
        the strains through a linear model, under the assumption of general symmetry.

        Parameters
        ----------
        - random_pairs_number (int): The number of random pairs of strains and stresses
          to use in the calculation. Default is 100, Minimum must be 4.

        Returns
        -------
        - C (np.ndarray): The stiffness matrix
        """
        if random_pairs_number > len(data_pairs):
            random_pairs_number = len(data_pairs)
            print("Warning: number of random pairs larger than data set. Using {random_pairs_number} pairs.}")
        random_pairs = random.sample(data_pairs, random_pairs_number)
        C_column_locator = {"C11": 1, "C12": 2, "C13": 3, "C14": 4, "C15": 5, "C16": 6,
                            "C21": 2, "C22": 7, "C23": 8, "C24": 9, "C25": 10, "C26": 11,
                            "C31": 3, "C32": 8, "C33": 12, "C34": 13, "C35": 14, "C36": 15,
                            "C41": 4, "C42": 9, "C43": 13, "C44": 16, "C45": 17, "C46": 18,
                            "C51": 5, "C52": 10, "C53": 14, "C54": 17, "C55": 19, "C56": 20,
                            "C61": 6, "C62": 11, "C63": 15, "C64": 18, "C65": 20, "C66": 21}

        b = np.zeros(len(random_pairs) * 6)
        A = np.zeros((len(random_pairs) * 6, 21))
        row_counter = 0
        for experiment in random_pairs:
            Pair_Counter = 1
            strains = experiment[0]
            stresses = experiment[1]

            for index in range(len(stresses)):
                stress = stresses[index]
                C_locations = [f"C{Pair_Counter}1",
                               f"C{Pair_Counter}2",
                               f"C{Pair_Counter}3",
                               f"C{Pair_Counter}4",
                               f"C{Pair_Counter}5",
                               f"C{Pair_Counter}6"]
                C_locations_translated = [C_column_locator[C_locations[0]],
                                          C_column_locator[C_locations[1]],
                                          C_column_locator[C_locations[2]],
                                          C_column_locator[C_locations[3]],
                                          C_column_locator[C_locations[4]],
                                          C_column_locator[C_locations[5]]]

                A[row_counter][C_locations_translated[0] - 1] = strains[0]
                A[row_counter][C_locations_translated[1] - 1] = strains[1]
                A[row_counter][C_locations_translated[2] - 1] = strains[2]
                A[row_counter][C_locations_translated[3] - 1] = strains[3]
                A[row_counter][C_locations_translated[4] - 1] = strains[4]
                A[row_counter][C_locations_translated[5] - 1] = strains[5]
                b[row_counter] = stress

                row_counter += 1
                Pair_Counter += 1

        # b.reshape(-1, 1)
        C_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        C = map_flat_to_matrix(C_flat)
        return C

    # start elastic coefficient identification
    data_pairs = list(zip(eps, sig))
    max_attempts = 50
    attempts = 0
    success = False
    npairs = len(data_pairs)
    while attempts < max_attempts and not success:
        if method == 'least_square':
            optimized_C = least_square(data_pairs, random_pairs_number=npairs)
            success = True
        elif method == 'decomposition':
            if initial_guess is None:
                initial_guess = np.random.rand(
                    21)  # Adjust the number according to the actual problem dimension
            result = minimize(objective_function, initial_guess, args=(data_pairs, method,), method='L-BFGS-B')
            if result.success:
                success = True
                _, optimized_C = map_flat_to_L_and_C(result.x)
            else:
                attempts += 1
                print("Optimization attempt {} failed".format(attempts))
        else:
            raise ValueError("Invalid method selected. Choose 'least_square' or 'decomposition'.")

    if not success:
        print("Optimization of material stiffness matrix failed after {} attempts".format(max_attempts))

    return np.array(optimized_C)


class Data(object):
    """Define class for handling data from virtual mechanical tests in micromechanical
    simulations and data from physical mechanical tests on materials with various
    microstructures. Data is used to train machine learning flow rules in pyLabFEA.

    Parameters
    ----------
    source: str or data array
        Either filename of datafile or array of initial yield stresses
    path_data: str
        Trunc of pathname for data files (optional, default: './')
    name: str
        Name of Dataset (optional, default: 'Dataset')
    mat_name : str
        Name of material (optional, default: 'Simulanium)
    sdim: int
        Dimensionality of stresses; if sdim = 3 only principal stresses are considered (optional, default: 6)
    epl_crit : float
        Critical plastic strain at which yield strength is defined(optional, default: 2.e-3)
    epl_start : float
        Start value of equiv. plastic strain at which data will be sampled(optional, default: 1.e-3)
    epl_max : float
        Maximum equiv. plastic strain up to which data is considered (optional, default: 0.05)
    plot : Boolean
        Plot data (optional, default: False)
    wh_data : Boolean
        Data for flow stresses and plastic strain tensors in work hardening regime exists (optional, default: True)

    Attributes
    ----------
    name: str
        Name of dataset
    sy_av: float
        Average initial yield stress
    E_av: float
        Average Young's modulus
    nu_av: float
        Average Poisson ratio
    mat_param: dictionary
        | Contains available data for ML yield function and for microstructural parameters
        | Items
        |    epc : critical value of equiv. plastic strain that defines the onset of plastic yielding
        |    ep_start : start value of equiv. plastic strain at which data is acquired
        |    ep_max : maximum value of equiv. plastic strain up to which data is considered
        |    delta_ep : minimum separation of plastic strains used for training, if 0 no separation is enforced
        |    lc_indices : array with start index for each load case
        |    peeq_max : maximum of equiv. plastic strain that occurred in data set, corrected for onset of plasticity at epc
        |    sdim : dimension of stress vector (must be 3 or 6)
        |    Name : material name
        |    Dataset : Name of dataset
        |    wh_data : indicate if strain hardening data exists
        |    Ntext : number of textures
        |    tx_name : name of texture
        |    texture : texture parameters
        |    flow_stress : array of flow stresses correlated to plastic strains
        |    plastic_strain : array of plastic strains corrected for onset of plasticity at epc
        |    E_av : avereage Young's modulus derived from data
        |    nu_av : average Poisson's ratio derived from data
        |    sy_av : average yield strength, i.e. flow stress at epc, obtained from data
        |    Nlc : number of load cases in data
        |    sy_list : ???
        |    sig_ideal : interpolated stress tensors at onset of plastic yielding (epc) for each load case

    """

    def __init__(self, source, path_data='./',
                 name='Dataset', mat_name="Simulanium",
                 sdim=6,
                 epl_crit=None,
                 epl_start=None, epl_max=None,
                 depl=0.,
                 plot=False,
                 wh_data=True,
                 texture_name='Random'):
        self.lc_data = None
        if sdim != 3 and sdim != 6:
            raise ValueError('Value of sdim must be either 3 or 6')
        self.mat_data = dict()
        self.mat_data['epc'] = epl_crit
        self.mat_data['ep_start'] = epl_start
        self.mat_data['ep_max'] = epl_max
        self.mat_data['delta_ep'] = depl
        self.mat_data['sdim'] = sdim
        self.mat_data['Name'] = mat_name
        self.mat_data['Dataset'] = name
        self.mat_data['wh_data'] = wh_data
        self.mat_data['Ntext'] = 1
        self.mat_data['tx_name'] = texture_name
        self.mat_data['texture'] = np.zeros(1)

        if isinstance(source, str):
            self.lc_data = self.read_data(path_data + source)
            self.parse_data(epl_crit, epl_start, epl_max, depl)  # add data to mat_data
        else:
            raw_data = np.array(source)
            self.convert_data(raw_data)  # add data to mat_data
        if plot:
            self.plot_training_data()

    def key_parser(self, key):
        parameters = key.split('_')
        Keys_Parsed = {"Stress_Type": parameters[0], "Load_Type": parameters[1], "Hash_Load": parameters[2],
                       "Hash_Orientation": parameters[3], "Texture_Type": parameters[4]}
        return Keys_Parsed

    def read_data(self, Data_File):
        data = json.load(open(Data_File))
        Final_Data = dict()

        for key, val in data.items():
            res = val['Results']
            Sigma = [res["S11"], res["S22"], res["S33"], res["S23"], res["S13"], res["S12"]]  # Order !!!
            E_Total = [res["E11"], res["E22"], res["E33"], res["E23"], res["E13"], res["E12"]]
            E_Plastic = [res["Ep11"], res["Ep22"], res["Ep33"], res["Ep23"], res["Ep13"], res["Ep12"]]
            Len_Sigma = len(Sigma[0])
            seq_full = np.zeros(Len_Sigma)
            peeq_full = np.zeros(Len_Sigma)
            peeq_plastic = np.zeros(Len_Sigma)
            Original_Stresses = np.zeros((Len_Sigma, 6))
            Original_Plastic_Strains = np.zeros((Len_Sigma, 6))
            Original_Total_Strains = np.zeros((Len_Sigma, 6))
            Plastic_Strains_Shifted = np.zeros((Len_Sigma, 6))
            epl = []
            if self.mat_data['epc'] is None:
                epc = 0.0
            else:
                epc = self.mat_data['epc']
            for i in range(Len_Sigma):
                Stress_6D = np.array([Sigma[0][i], Sigma[1][i], Sigma[2][i], Sigma[3][i], Sigma[4][i], Sigma[5][i]])
                Original_Stresses[i, :] = Stress_6D
                seq_full[i] = FE.sig_eq_j2(Stress_6D)

                E_Plastic_6D = np.array([E_Plastic[0][i], E_Plastic[1][i], E_Plastic[2][i],
                                         E_Plastic[3][i], E_Plastic[4][i], E_Plastic[5][i]])

                peeq_plastic[i] = FE.eps_eq(E_Plastic_6D)
                Original_Plastic_Strains[i, :] = E_Plastic_6D

                E_Total_6D = np.array([E_Total[0][i], E_Total[1][i], E_Total[2][i],
                                       E_Total[3][i], E_Total[4][i], E_Total[5][i]])

                peeq_full[i] = FE.eps_eq(E_Total_6D)
                Original_Total_Strains[i] = E_Total_6D
                # For having also the elastic data and shift the 0 plastic strain
                # to 0.02% to match the micromechanical data.
                # Can be used in Stress-Strain Reconstruction.
                scale = np.maximum(peeq_plastic[i], 1.e-10)
                Plastic_Strains_Shifted[i] = E_Plastic_6D * (1. - epc / scale)

            Final_Data[key] = {"Stress": Original_Stresses,
                               "Eq_Stress": seq_full,
                               "Strain_Plastic": Original_Plastic_Strains,
                               "Eq_Strain_Plastic": peeq_plastic,  # always shifted ???
                               "Shifted_Strain_Plastic": Plastic_Strains_Shifted,  # required ???
                               "Strain_Total": Original_Total_Strains,
                               "Eq_Strain_Total": peeq_full,
                               }
        return Final_Data

    def parse_data(self, epl_crit, epl_start, epl_max, depl):
        """
        Read data and store in attribute 'mat_data'
        Estimate elastic properties and initial yield strength from data for each load case and form averages.

        Parameters
        ----------
        epl_crit : float
            Critical value for onset of yielding
        epl_start : float
            Start value of equiv. plastic strain at which data acquisition of flow stresses will start
        epl_max : float
            Maximum equiv. strain up to which data is considered
        depl : float
            Minimum separation of plastic strains used for training, if 0, no minimum is enforced
        """
        # initializations
        Nlc = len(self.lc_data.keys())
        sy_av = 0.
        peeq_max = 0.
        ct = 0
        ep_c = 0.0
        ep_s = 0.0
        ep_m = 0.0
        sy_list = []
        sig = []
        epl = []
        sig_ideal = []
        lc_ind_list = np.zeros(Nlc + 1, dtype=int)
        elstrain = []
        elstress = []
        it_list = []
        for key, val in self.lc_data.items():
            # estimate yield point for load case
            # (1) find transition index w/o definition of critical plastic strain
            it = find_transition_index(val["Eq_Stress"])
            elstrain.append(val['Strain_Total'][it])  # total strain tensor at transition
            elstress.append(val['Stress'][it])  # stress tensor at transition
            peeq = val['Eq_Strain_Plastic']
            if epl_crit is None:
                epc_lc = peeq[it]
            else:
                epc_lc = epl_crit
            if epl_start is None:
                eps_lc = peeq[it]
            else:
                eps_lc = epl_start
            if epl_max is None:
                epm_lc = max(peeq)
            else:
                epm_lc = epl_max

            # (2) estimate yield point for ideal plasticity consideration
            i_ideal = np.nonzero(peeq < epc_lc)[0]
            if len(i_ideal) < 2:
                print(
                    f'Skipping data set {key} (No {ct}): No elastic range before yield onset.')
                Nlc -= 1
                continue

            # (3) identify elastic and plastic regions based on critical values for plastic strain
            iel = np.nonzero(peeq < eps_lc)[0]
            ipl = np.nonzero(np.logical_and(peeq >= eps_lc, peeq <= epm_lc))[0]
            if len(iel) < 2:
                print(f'Skipping data set {key} (No {ct}): No elastic range: IEL: {iel}, vals: {len(peeq)}')
                Nlc -= 1
                continue
            if len(ipl) < 2:
                print(
                    f'Skipping data set {key} (No {ct}): No plastic range: IPL: {ipl}, vals: {len(peeq)}; '
                    f'{epl_start}, {epl_max}')
                Nlc -= 1
                continue
            # store different indices and critical values for statistical analysis
            it_list.append([it, int(i_ideal[-1]), int(iel[-1]), int(ipl[0])])
            ep_c += epc_lc
            ep_s += eps_lc
            ep_m += epm_lc

            # get initial yield stress
            sig_ideal.append(val['Stress'][i_ideal[-1]])  # stress tensor at crit. plastic strain
            sy_av += val['Eq_Stress'][i_ideal[-1]]  # calculate avereaged equiv. stress at epc over all load cases
            s0 = val['Eq_Stress'][iel[-1]]
            s1 = val['Eq_Stress'][ipl[0]]
            e0 = peeq[iel[-1]]
            e1 = peeq[ipl[0]]
            s_start = s0 + (eps_lc - e0) * (s1 - s0) / (e1 - e0)
            sy_list.append(s_start)  # interpolated equiv. stress at plastic strain eps_lc
            if peeq[ipl[-1]] > peeq_max:
                peeq_max = peeq[ipl[-1]]

            # enforce minimum distance depl between strain values along load cases
            # shift values to have 0 equivalent plastic strain at start of yielding.
            # values might be negative (atomistic / noisy data), cutoff at 0 enforced
            # difference to 'Shifted_Strain_Plastic' ???
            eps = -depl
            nv = 0
            for i in ipl:
                hh = peeq[i]
                if hh >= eps + depl:
                    sig.append(val['Stress'][i])
                    sc_epl = max(0., 1. - epc_lc / hh)
                    epl.append(val['Strain_Plastic'][i] * sc_epl)
                    eps = hh
                    nv += 1
            lc_ind_list[ct] = nv + lc_ind_list[ct - 1]  # store end index for values of this load case

            # get texture name
            ''' Warning: This should be read only once from metadata !!!'''
            Key_Translated = self.key_parser(key)
            self.mat_data['ms_type'] = Key_Translated["Texture_Type"]  # unimodal texture type
            ct += 1

        # initialize mat_data dictionary used to create material objects
        C = get_elastic_coefficients(elstrain, elstress, method='least_square')
        sy_av /= Nlc
        self.mat_data['flow_stress'] = np.array(sig)  # list of stress tensors for all load cases
        self.mat_data['plastic_strain'] = np.array(epl)  # list of plastic strain tensors for all load cases
        self.mat_data['lc_indices'] = lc_ind_list  # list of starting indices of data sets for different load cases
        self.mat_data['epc'] = ep_c / Nlc  # critical value of plastic strain for yield onset definition
        self.mat_data['ep_start'] = ep_s / Nlc  # start value of plastic strain for training data
        self.mat_data['ep_max'] = ep_m / Nlc  # nominal maximum plastic strain to be considered
        self.mat_data['peeq_max'] = peeq_max - ep_c / Nlc  # maximum plastic strain occuring in training data
        self.mat_data['elast_const'] = C  # anisotropic elastic coefficients obtained from data
        self.mat_data['sy_av'] = sy_av  # equiv. stress at epc averaged over all load cases
        self.mat_data['Nlc'] = Nlc  # number of load cases
        self.mat_data['sy_list'] = sy_list  # equiv. stress at ep_start
        self.mat_data['sig_ideal'] = np.array(sig_ideal)  # stress tensor at epc
        self.mat_data['elstress'] = elstress  # list of all stress tensors at the end of linear regime
        self.mat_data['elstrain'] = elstrain  # list of all total strain tensors at end of linear regime
        self.mat_data['transition_ind'] = it_list  # list of indices identified at elastic/plastic transition
                                                   # by various methods
        print(f'\n###   Data set: {self.mat_data["Name"]}  ###')
        print(f'Type of microstructure: {Key_Translated["Texture_Type"]}')
        print(f'Estimated elastic constants (in GPa): C={C * 1.E-3}')  # get units from metadata ???
        print(f'Estimated yield strength: {sy_av:5.2f} MPa at PEEQ = {ep_s:5.3f}')

    def convert_data(self, sig):
        """
        Convert data provided only for stress tensor at yield point into mat_param dictionary

        Parameters
        ----------
        sig : ndarray
            Stress tensors at yield onset
        """
        print("inside where it should not be")
        Nlc = len(sig)
        sdim = len(sig[0, :])
        if sdim != self.mat_data['sdim']:
            warnings.warn(
                'Warning: dimension of stress in data does not agree with parameter sdim. Use value from data.')
        self.mat_data['sig_ideal'] = sig
        self.mat_data['wh_data'] = False
        lc_ind_list = np.linspace(0, Nlc)
        self.mat_data['lc_indices'] = np.append(lc_ind_list, 0.)
        self.mat_data['E_av'] = 1.  # WARNING: value cannot be derived from data
        self.mat_data['nu_av'] = 0.3  # WARNING: value cannot be derived from data
        self.mat_data['sy_av'] = np.mean(FE.sig_eq_j2(sig))
        self.mat_data['peeq_max'] = self.mat_data['epc']
        self.mat_data['Nlc'] = Nlc
        print(f'\n###   Data set: {self.mat_data["Name"]}  ###')
        print(f'Converted data for {Nlc} stress tensors at yield onset into material data.')
        print('WARNING: Elastic parameters cannot be derived from data. Please set them manually.')

    def add2mat_data(self, data_dict, key):
        # create shifted plastic strain ???
        epl_crit = self.mat_data['epc']
        epl_start = self.mat_data['ep_start']
        epl_max = self.mat_data['ep_max']
        depl = self.mat_data['delta_ep']
        self.lc_data[key] = data_dict
        self.parse_data(epl_crit, epl_start, epl_max, depl)

    def plot_training_data(self, emax=1):
        # Function for plotting stress strain curves from the database using total strain and plastic strain values.
        for data, xlabel, ylabel in [(self.lc_data, "Total Strain", "Stress"),
                                     (self.lc_data, "Plastic Strain", "Stress")]:
            self.plot_data(data, xlabel, ylabel, emax=emax)

    def plot_data(self, data, xlabel, ylabel, emax=None):
        for key, val in data.items():
            plt.scatter(val["Strain_Total"], val["Stress"], s=1)
            plt.tick_params(axis='both', which='major', labelsize=12)
            if emax is not None:
                plt.xlim(0, emax)
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
        plt.show()

    def plot_set(self, db, mat_param):
        fontsize = 18
        cmap = plt.cm.get_cmap('viridis', 10)
        plt.figure(figsize=(18, 7))
        plt.subplots_adjust(wspace=0.2)
        plt.subplot(1, 2, 1)
        for i, key in enumerate(db.keys()):
            col = FE.polar_ang(np.array(db[key]['Load'])) / np.pi
            plt.plot(FE.eps_eq(np.array(db[key]['Main_Strains'])) * 100,
                     FE.seq_J2(np.array(db[key]['Main_Stresses'])), color=cmap(col))
        plt.xlabel(r'$\epsilon_{eq}^\mathrm{tot}$ (%)', fontsize=fontsize)
        plt.ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=fontsize)
        plt.title('Equiv. total strain vs. equiv. J2 stress', fontsize=fontsize)
        plt.tick_params(axis="x", labelsize=fontsize - 4)
        plt.tick_params(axis="y", labelsize=fontsize - 4)

        syc_0 = []
        syc_1 = []
        for i, key in enumerate(db.keys()):
            plt.subplot(1, 2, 2)  # , projection='polar')
            syc = FE.s_cyl(np.array(db[key]["Parsed_Data"]["S_yld"]))
            plt.plot(np.array(db[key]["Parsed_Data"]["S_Cyl"])[db[key]["Parsed_Data"]["Plastic_data"], 1],
                     np.array(db[key]["Parsed_Data"]["S_Cyl"])[db[key]["Parsed_Data"]["Plastic_data"], 0], 'or')
            plt.plot(np.array(db[key]["Parsed_Data"]["S_Cyl"])[db[key]["Parsed_Data"]["Elastic_data"], 1],
                     np.array(db[key]["Parsed_Data"]["S_Cyl"])[db[key]["Parsed_Data"]["Elastic_data"], 0], 'ob')
            syc_0.append(syc[0][0])
            syc_1.append(syc[0][1])

        syc_1, syc_0 = zip(*sorted(zip(syc_1, syc_0)))
        plt.plot(syc_1, syc_0, '-k')
        plt.plot([-np.pi, np.pi], [mat_param['sy_av'], mat_param['sy_av']], '--k')
        plt.legend(['raw data above yield point', 'raw data below yield point',
                    'interpolated yield strength', 'average yield strength'], loc=(1.04, 0.8),
                   fontsize=fontsize - 2)
        plt.title('Raw data ', fontsize=fontsize)
        plt.xlabel(r'$\theta$ (rad)', fontsize=fontsize)
        plt.ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=fontsize)
        plt.tick_params(axis="x", labelsize=fontsize - 4)
        plt.tick_params(axis="y", labelsize=fontsize - 4)
        plt.show()

    # Check
    def plot_yield_locus(self, db, mat_data, active, scatter=False, data=None,
                         data_label=None, arrow=False, file=None, title=None,
                         fontsize=18):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                               figsize=(15, 8))
        cmap = plt.cm.get_cmap('viridis', 10)
        # ms_max = np.amax(self.mat_param[active])
        Ndat = len(mat_data[active])
        v0 = mat_data[active][0]
        scale = mat_data[active][-1] - v0
        if np.any(np.abs(scale) < 1.e-3):
            scale = 1.
        sc = []
        ppe = []
        scy = []
        for i in range(Ndat):
            if active == 'flow_stress':
                sc.append(FE.s_cyl(mat_data['flow_stress'][i]))
                ppe.append(FE.eps_eq(mat_data['plastic_strain'][i]))
                for j in range(len(ppe)):
                    if ppe[j] < 0.003:
                        scy.append(sc[j])
        label = 'Flow Stress'
        color = 'b'
        scy = np.array(scy)
        scy = np.array(scy)
        scy = np.array(scy)
        ax.scatter(scy[:, 1], scy[:, 0], marker=".")
        plt.show()
