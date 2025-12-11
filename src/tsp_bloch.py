import numpy as np
from qiskit.quantum_info import Statevector
from kaleidoscope import bloch_sphere
from utility import get_bloch_coordinates_from_statevector

class TSPBlochInstance:
    def __init__(self, num_cities, P, dist_matrix, graph):
        self.num_cities = num_cities
        self.P = P
        self.dist_matrix = dist_matrix
        self.graph = graph
    
    def get_city_state(self, city_index):
        """
        Get the quantum state of a specific city.

        Parameters:
        - city_index: Index of the city

        Returns:
        - Statevector object representing the city's quantum state
        """
        return self.P[city_index][city_index]
    
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
        coords = [get_bloch_coordinates_from_statevector(self.P[i][i]) for i in range(self.num_cities)]
        return bloch_sphere(points=coords)
    
    def plot_all_states_on_bloch_sphere(self):
        """
        Plot all quantum states (including intermediates) on the Bloch sphere.

        Returns:
        - matplotlib Figure object with all states
        """
        coords = []
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                coord = get_bloch_coordinates_from_statevector(self.P[i][j])
                coords.append(coord)
        return bloch_sphere(points=coords)
