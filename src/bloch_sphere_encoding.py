from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RZGate
from qiskit.visualization import plot_bloch_multivector
import numpy as np
from kaleidoscope import bloch_sphere

class BlochSphereEncoder:
    def __init__(self, instance):
        self.n_cities = instance.n_cities
        self.dist_matrix = instance.dist_matrix
        self.city_states = self.encode_cities()

    def encode_cities(self):
        """
        Encode cities into quantum states represented on the Bloch sphere.

        Returns:
        - states: List of Statevector objects representing the encoded quantum states
        """
        states = []
        angle = (2*np.pi) / self.n_cities
        rz = RZGate(angle)

        for i in range(self.n_cities):
            operator = rz.power(i)
            state = Statevector.from_label('+').evolve(operator)
            states.append(state)

        return states
    
    def get_city_state(self, city_index):
        """
        Get the quantum state of a specific city.

        Parameters:
        - city_index: Index of the city

        Returns:
        - Statevector object representing the city's quantum state
        """
        return self.city_states[city_index]
    
    @staticmethod
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
    
    def plot_city_on_bloch_sphere(self, city_index):
        """
        Plot the quantum state of a specific city on the Bloch sphere.

        Parameters:
        - city_index: Index of the city

        Returns:
        - matplotlib Figure object
        """
        state = self.get_city_state(city_index)
        coord = self.get_bloch_coordinates_from_statevector(state)
        return bloch_sphere(points=[coord])

    def plot_all_cities_on_bloch_sphere(self):
        """
        Plot the quantum states of all cities on the Bloch sphere.

        Returns:
        - matplotlib Figure object with all city states
        """
        coords = [self.get_bloch_coordinates_from_statevector(state) for state in self.city_states]
        return bloch_sphere(points=coords)
