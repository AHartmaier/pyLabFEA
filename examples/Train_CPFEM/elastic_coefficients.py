import pylabfea as FE
import numpy as np
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt



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


def objective_function(x_flat, penalty_weight=1e9, lambda_reg=1e-3):
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


def least_square(random_pairs_number=100):
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
    random_pairs = random.sample(data_pairs, random_pairs_number)
    C_coloumn_locator = {"C11": 1, "C12": 2, "C13": 3, "C14": 4, "C15": 5, "C16": 6,
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
            C_locations_translated = [C_coloumn_locator[C_locations[0]],
                                      C_coloumn_locator[C_locations[1]],
                                      C_coloumn_locator[C_locations[2]],
                                      C_coloumn_locator[C_locations[3]],
                                      C_coloumn_locator[C_locations[4]],
                                      C_coloumn_locator[C_locations[5]]]

            A[row_counter][C_locations_translated[0] - 1] = strains[0]
            A[row_counter][C_locations_translated[1] - 1] = strains[1]
            A[row_counter][C_locations_translated[2] - 1] = strains[2]
            A[row_counter][C_locations_translated[3] - 1] = strains[3]
            A[row_counter][C_locations_translated[4] - 1] = strains[4]
            A[row_counter][C_locations_translated[5] - 1] = strains[5]
            b[row_counter] = stress

            row_counter += 1
            Pair_Counter += 1

    #b.reshape(-1, 1)
    C_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    C = map_flat_to_matrix(C_flat)
    return C


def get_elastic_coefficients(method='direct', initial_guess=None):
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
    method: str
        Method to be used for calculating the stiffness matrix. Options
        include 'least_square' and 'decomposition'. The 'least_square' method
        calculates the stiffness matrix using a least squares fit to the stress-strain data.
        The 'decomposition' method uses an optimization approach with the
        specified method for interpreting the stiffness matrix from the optimization
        variables. (Optional: default is 'decomposition').
    initial_guess: np.ndarray or None
        Initial guess for the stiffness matrix coefficients
        used in the optimization approach. If None, a random initial guess is generated.
        This parameter is ignored when using the 'least_square' method. Default is None.

    Returns:
    _: np.ndarray
        The optimized symmetric 6x6 stiffness matrix (elastic coefficients) as a numpy array.
        If the optimization or calculation fails to converge to a solution within the
        maximum number of attempts, the function may return the last attempted solution
        or raise an error, depending on implementation details.
    """
    max_attempts = 50
    attempts = 0
    success = False
    while attempts < max_attempts and not success:
        if method == 'least_square':
            optimized_C = least_square(random_pairs_number=296)  # All available stress-strain pair is 296
            success = True
            print(optimized_C)
        elif method == 'decomposition':
            if initial_guess is None:
                initial_guess = np.random.rand(21)  # Adjust the number according to the actual problem dimension
            result = minimize(objective_function, initial_guess, args = (method,), method = 'L-BFGS-B')
            if result.success:
                success=True
                _, optimized_C = map_flat_to_L_and_C(result.x)
            else:
                attempts += 1
                print("Optimization attempt {} failed".format(attempts))
        else:
            raise ValueError("Invalid method selected. Choose 'least_square' or 'decomposition'.")

    if success:
        print("Optimization succeeded after {} attempts".format(attempts))
        print("Optimized C matrix:")
        print(optimized_C)
    else:
        print("Optimization failed after {} attempts".format(max_attempts))

    return np.array(optimized_C)

# main code block
db = FE.Data("Data_Random_Texture.json",  wh_data=True)
stress = db.mat_data['elstress']
strain = db.mat_data['elstrains']
assert len(stress) == len(strain), "Stress and strain data must have the same length"
data_pairs = list(zip(strain, stress))

C = get_elastic_coefficients(method='least_square')

# Visualize the results. Using the calculated C matrix to predict stresses from strains for 10% of the data points.
predicted_stresses = np.array([calculate_stress_from_strain(s, C) for s in strain])
stresses = np.array(stress)
predicted_stresses = np.array(predicted_stresses)
num_data_points = stresses.shape[0]
subset_size = int(num_data_points * 0.1)
np.random.seed(42)
selected_indices = np.random.choice(num_data_points, subset_size, replace=False)
selected_actual_stresses = stresses[selected_indices]
selected_predicted_stresses = predicted_stresses[selected_indices]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
component_names = ['Stress_11', 'Stress_22', 'Stress_33', 'Stress_12', 'Stress_13', 'Stress_23']
for i, ax in enumerate(axes):
    ax.scatter(selected_actual_stresses[:, i], selected_predicted_stresses[:, i], alpha=0.5, label='Data Points')
    max_stress = max(selected_actual_stresses[:, i].max(), selected_predicted_stresses[:, i].max())
    min_stress = min(selected_actual_stresses[:, i].min(), selected_predicted_stresses[:, i].min())
    ax.plot([min_stress, max_stress], [min_stress, max_stress], 'k--', label='Perfect Fit')
    ax.set_xlabel(f'Actual {component_names[i]} Stress')
    ax.set_ylabel(f'Predicted {component_names[i]} Stress')
    ax.set_title(f'Component: {component_names[i]}')
    ax.grid(True)
    ax.legend()
plt.tight_layout()
plt.show()
