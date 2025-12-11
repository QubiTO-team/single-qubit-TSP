from qiskit.quantum_info import Statevector
import numpy as np
from tsp_bloch import TSPBlochInstance

class BlochSphereEncoder:
    def __init__(self, instance):
        self.n_cities = instance.n_cities

        self.dist_matrix = instance.dist_matrix
        self.dist_matrix = self.rescale_distances()

        self.graph = instance.graph

        self.city_states = self.encode_cities()

    def get_encoded_instance(self):
        """
        Get the TSP instance with cities encoded as quantum states.

        Returns:
        - TSPBlochInstance object with encoded city states
        """
        return TSPBlochInstance(self.n_cities, self.city_states, self.dist_matrix)

    def encode_cities(self):
        """
        Encode cities into quantum states represented on the Bloch sphere.

        Returns:
        - states: List of Statevector objects representing the encoded quantum states
        """
        states = []
        angle = 0
        for i in range(self.n_cities):
            state = Statevector([np.sqrt(2)/2, np.exp(1j*angle)*np.sqrt(2)/2])
            states.append(state)
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
                neighbor_state = self.city_states[i]
                for alpha in np.linspace(0, 1, num=5):
                    intermediate_state = (1 - alpha) * current_state + alpha * neighbor_state
                    intermediate_state = intermediate_state / np.linalg.norm(intermediate_state.data)
                    intermediate_states.append(Statevector(intermediate_state.data))
            else:
                intermediate_states.append(current_state)

        return intermediate_states
    
    def rescale_distances(self, new_min=0, new_max=np.pi/2):
        """
        Rescale the distance matrix to fit within a specified range.

        Parameters:
        - new_min: Minimum value of the new scale
        - new_max: Maximum value of the new scale

        Returns:
        - rescaled_dist_matrix: Numpy array with rescaled distances
        """
        old_max = np.max(self.dist_matrix)

        rescaled_dist_matrix = self.dist_matrix / old_max
        rescaled_dist_matrix = rescaled_dist_matrix * (new_max - new_min) + new_min

        return rescaled_dist_matrix