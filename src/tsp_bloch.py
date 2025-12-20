import numpy as np
from qiskit.quantum_info import Operator
from kaleidoscope import bloch_sphere
from utility import get_bloch_coordinates_from_statevector

class TravelOperator:
    def __init__(self, from_city, to_city, P, dist_matrix):
        self.from_city = from_city
        self.to_city = to_city
        self.set_up_op(P)
        self.set_down_op(P)
        self.set_cost(dist_matrix)

    def set_up_op(self, P):
        """
        Set the 'up' quantum operator.

        Parameters:
        - P: Matrix of quantum states
        """
        v1 = P[self.from_city][self.from_city].data
        v2 = P[self.from_city][self.to_city].data
        mat = np.outer(v2, v1.conj())
        self.up = Operator(mat)

    
    def set_down_op(self, P):
        """
        Set the 'down' quantum operator.

        Parameters:
        - P: Matrix of quantum states
        """
        v1 = P[self.from_city][self.to_city].data
        v2 = P[self.to_city][self.to_city].data
        mat = np.outer(v2, v1.conj())
        self.down = Operator(mat)

    def set_cost(self, dist_matrix):
        """
        Set the cost of traveling from one city to another.

        Parameters:
        - dist_matrix: Matrix of distances between cities
        """
        self.cost = dist_matrix[self.from_city][self.to_city]

class TSPBlochInstance:
    def __init__(self, num_cities, P, dist_matrix, graph, allowed_routes):
        self.num_cities = num_cities
        self.P = P
        self.dist_matrix = dist_matrix
        self.graph = graph
        self.allowed_routes = allowed_routes
        self.set_travel_operators()
    
    def set_travel_operators(self):
        """
        Set travel operators for all pairs of cities.

        Returns:
        - travel_operators: List of TravelOperator objects
        """
        travel_operators = [[] for _ in range(self.num_cities)]
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    operator = TravelOperator(i, j, self.P, self.dist_matrix)
                    travel_operators[i].append(operator)
                else:
                    travel_operators[i].append(None)
        self.travel_operators = travel_operators

    def get_allowed_routes(self):
        """
        Get all allowed routes for the TSP instance.

        Returns:
        - allowed_routes: List of lists, each representing a valid route
        """
        return self.allowed_routes

    def get_city_state(self, city_index):
        """
        Get the quantum state of a specific city.

        Parameters:
        - city_index: Index of the city

        Returns:
        - Statevector object representing the city's quantum state
        """
        return self.P[city_index][city_index]
    
    def get_state(self,i,j):
        """
        Get a quantum state Pij from the P matrix.

        Parameters:
        - i: Index of the first city
        - j: Index of the second city

        Returns:
        - Statevector object representing the quantum state
        """
        return self.P[i][j]
    
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
        points_color = ["#FF7300C3"for _ in range(self.num_cities)]
        return bloch_sphere(points=coords, points_color=points_color)
    
    def plot_all_states_on_bloch_sphere(self):
        """
        Plot all quantum states (including intermediates) on the Bloch sphere.

        Returns:
        - matplotlib Figure object with all states
        """
        coords = []
        points_color = []
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                coord = get_bloch_coordinates_from_statevector(self.P[i][j])
                coords.append(coord)
                if i == j:
                    points_color.append("#FF7300C3")
                else:
                    points_color.append("#7BFF0097")
        return bloch_sphere(points=coords, points_color=points_color)