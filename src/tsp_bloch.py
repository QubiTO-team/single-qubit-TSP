import numpy as np
from qiskit.quantum_info import Statevector
from kaleidoscope import bloch_sphere
from utility import get_bloch_coordinates_from_statevector

class TSPBlochInstance:
    def __init__(self, num_cities, city_states, dist_matrix=None):
        self.num_cities = num_cities
        self.city_states = city_states
        self.dist_matrix = dist_matrix
    
    def get_city_state(self, city_index):
        """
        Get the quantum state of a specific city.

        Parameters:
        - city_index: Index of the city

        Returns:
        - Statevector object representing the city's quantum state
        """
        return self.city_states[city_index]
    
    def plot_city_on_bloch_sphere(self, city_index):
        """
        Plot the quantum state of a specific city on the Bloch sphere.

        Parameters:
        - city_index: Index of the city

        Returns:
        - matplotlib Figure object
        """
        state = self.get_city_state(city_index)
        coord = get_bloch_coordinates_from_statevector(state)
        return bloch_sphere(points=[coord])

    def plot_all_cities_on_bloch_sphere(self):
        """
        Plot the quantum states of all cities on the Bloch sphere.

        Returns:
        - matplotlib Figure object with all city states
        """
        coords = [get_bloch_coordinates_from_statevector(state) for state in self.city_states]
        return bloch_sphere(points=coords)
