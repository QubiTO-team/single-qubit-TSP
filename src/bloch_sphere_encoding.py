from qiskit.quantum_info import Statevector
import numpy as np
from tsp_bloch import TSPBlochInstance
from utility import rescale_distances

class BlochSphereEncoder:
    def __init__(self, instance):
        self.n_cities = instance.n_cities

        self.dist_matrix = instance.dist_matrix

        self.graph = instance.graph

        self.city_states = self.encode_cities()

        self.P = self.calculate_P_matrix()

        self.allowed_routes = instance.allowed_routes

    def get_encoded_instance(self):
        """
        Get the TSP instance with cities encoded as quantum states.

        Returns:
        - TSPBlochInstance object with encoded city states
        """
        return TSPBlochInstance(self.n_cities, self.P, self.allowed_routes)

    def encode_cities(self):
        """
        Encode cities into quantum states represented on the Bloch sphere.

        Returns:
        - states: List of Statevector objects representing the encoded quantum states
        """
        states = []
        self.city_angles = []
        angle = 0
        for i in range(self.n_cities):
            state = Statevector([np.sqrt(2)/2, np.exp(1j*angle)*np.sqrt(2)/2])
            states.append(state)
            self.city_angles.append(angle)
            angle += (2 * np.pi / self.n_cities)

        return states
    
    def calculate_intermediate_states(self, city_index):
        """
        Calculate intermediate quantum states between a city and its neighbors.

        Parameters:
        - city_index: Index of the city

        Returns:
        - intermediate_states: List of Statevector objects representing intermediate states
        """
        intermediate_states = []
        current_state = self.city_states[city_index]

        for i in range(self.n_cities):
            if i != city_index:
                xi = np.pi/2 - self.dist_matrix[city_index][i]
                phi = self.city_angles[city_index]
                intermediate_state = Statevector([np.cos(xi/2), np.exp(1j*phi)*np.sin(xi/2)])
                intermediate_states.append(intermediate_state)
            else:
                intermediate_states.append(current_state)

        return intermediate_states
    
    def calculate_P_matrix(self):
        """
        Calculate the Pij states.

        Returns:
        - P: Matrix of Statevectors
        """
        P = []
        for i in range(self.n_cities):
            P_row = self.calculate_intermediate_states(i)
            P.append(P_row)
        return P
