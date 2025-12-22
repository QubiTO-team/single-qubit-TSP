import numpy as np
from qiskit.quantum_info import Operator
from kaleidoscope import bloch_sphere
from utility import get_bloch_coordinates_from_statevector

class TravelOperator:
    def __init__(self, from_city, to_city, P):
        self.from_city = from_city
        self.to_city = to_city
        self.from_state = P[from_city][from_city]
        self.intermediate_state = P[from_city][to_city]
        self.to_state = P[to_city][to_city]
        self.set_up_op()
        self.set_down_op()

    def set_up_op(self):
        """
        Set the 'up' quantum operator.

        Parameters:
        - P: Matrix of quantum states
        """
        v1 = self.from_state.data
        v2 = self.intermediate_state.data
        mat = np.outer(v2, v1.conj())
        self.up = Operator(mat)

    
    def set_down_op(self):
        """
        Set the 'down' quantum operator.

        Parameters:
        - P: Matrix of quantum states
        """
        v1 = self.intermediate_state.data
        v2 = self.to_state.data
        mat = np.outer(v2, v1.conj())
        self.down = Operator(mat)

    def calculate_cost(self):
        """
        Set the cost of traveling from one city to another.

        Parameters:
        - P: Matrix of quantum states
        """
        cost = 2 * np.arccos(np.real(np.inner(self.intermediate_state.data.conj(), self.from_state.data))) # Perchè 2????
        return cost

class TSPBlochInstance:
    def __init__(self, num_cities, P, allowed_routes):
        self.num_cities = num_cities
        self.P = P
        self.allowed_routes = allowed_routes
        self.set_travel_operators()

    def solve_brute_force(self, backend=None, verbose=False):
        """
        Solve the TSP instance using brute-force search.

        Returns:
        - best_route: List representing the best route found
        - min_cost: Cost of the best route
        - mean_error: Mean error in final states across all routes
        """
        min_cost = float('inf')
        best_route = None
        mean_error = 0.0

        if backend is None:

            for r in range(len(self.allowed_routes)):
                route = self.allowed_routes[r]
                total_cost = 0
                initial_state = self.get_city_state(route[0])
                current_state = initial_state
                final_state = initial_state
                for i in range(len(route)-1):
                    from_city = route[i]
                    to_city = route[i+1]
                    travel_operator = self.travel_operators[from_city][to_city]
                    current_state = current_state.evolve(travel_operator.up).evolve(travel_operator.down)
                    total_cost += travel_operator.calculate_cost()
                if verbose:
                    print(f"Route: {route}, Cost: {total_cost}")
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_route = route
                mean_error = (mean_error * r + np.linalg.norm(final_state.data - current_state.data)) / (r + 1)

        return best_route, min_cost, mean_error
    
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
                    operator = TravelOperator(i, j, self.P)
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