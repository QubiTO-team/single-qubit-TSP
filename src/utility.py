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