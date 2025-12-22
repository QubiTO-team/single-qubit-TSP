import numpy as np
from qiskit.quantum_info import Statevector

def get_bloch_coordinates_from_statevector(statevector):
    """
    Extract Bloch sphere coordinates (x, y, z) from a Qiskit Statevector.
    
    Parameters:
    - statevector: Qiskit Statevector object (single qubit)
    
    Returns:
    - tuple: (x, y, z) Bloch coordinates
    """
    if not isinstance(statevector, Statevector):
        raise TypeError("Input must be a Qiskit Statevector")
    
    # Get the state amplitudes
    amplitudes = statevector.data
    
    if len(amplitudes) != 2:
        raise ValueError("Statevector must be for a single qubit (2 amplitudes)")
    
    alpha, beta = amplitudes[0], amplitudes[1]
    
    # Compute Bloch coordinates
    x = 2 * np.real(alpha * np.conj(beta))
    y = 2 * np.imag(alpha * np.conj(beta))
    z = np.abs(alpha)**2 - np.abs(beta)**2
    
    return [x, y, z]

    
def rescale_distances(dist_matrix, new_min=0, new_max=np.pi/2):
    """
    Rescale the distance matrix to fit within a specified range.

    Parameters:
    - new_min: Minimum value of the new scale
    - new_max: Maximum value of the new scale

    Returns:
    - rescaled_dist_matrix: Numpy array with rescaled distances
    """
    old_max = np.max(dist_matrix)

    rescaled_dist_matrix = dist_matrix / old_max
    rescaled_dist_matrix = rescaled_dist_matrix * (new_max - new_min) + new_min

    return rescaled_dist_matrix