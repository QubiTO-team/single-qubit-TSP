import numpy as np
from qiskit.quantum_info import Operator, DensityMatrix
from kaleidoscope import bloch_sphere
from utility import get_bloch_coordinates_from_statevector
from qiskit import QuantumCircuit, transpile
from qiskit_experiments.library import StateTomography
from qiskit_experiments.framework import ExperimentData
from qiskit.quantum_info import state_fidelity as fidelity

class TravelOperator:
    def __init__(self, from_city, to_city, P):
        self.from_city = from_city
        self.to_city = to_city
        self.from_state = P[from_city][from_city]
        self.intermediate_state = P[from_city][to_city]
        self.to_state = P[to_city][to_city]
        self.up = self.compute_operator(self.from_state, self.intermediate_state)
        self.down = self.compute_operator(self.intermediate_state, self.to_state)

    def compute_operator(self, s1, s2):
        """
        Set the 'up' quantum operator.
        """
        v1 = get_bloch_coordinates_from_statevector(s1)
        v2 = get_bloch_coordinates_from_statevector(s2)
        
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)
        try:
            axis = axis / axis_norm
        except:
            print("Warning: Zero division error in axis normalization. Using default axis.")
            axis = np.array([1,0,0])
        
        dot_product = np.dot(v1, v2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        delta = np.arccos(dot_product)

        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        nx, ny, nz = axis

        n_dot_sigma = nx * X + ny * Y + nz * Z

        mat = (np.cos(delta/2) * I - 1j * np.sin(delta/2) * n_dot_sigma)

        return Operator(mat)

    def calculate_cost(self):
        """
        Set the cost of traveling from one city to another.

        Parameters:
        - P: Matrix of quantum states
        """
        cost = 2 * np.arccos(np.real(np.inner(self.intermediate_state.data.conj(), self.from_state.data)))
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

        else:

            for r in range(len(self.allowed_routes)):
                qc = QuantumCircuit(1)
                route = self.allowed_routes[r]
                total_cost = 0
                initial_state = self.get_city_state(route[0])
                initial_rho = DensityMatrix(initial_state)
                qc.initialize(initial_state, 0)
                for i in range(len(route)-1):
                    from_city = route[i]
                    to_city = route[i+1]
                    travel_operator = self.travel_operators[from_city][to_city]
                    qc.append(travel_operator.up, [0])
                    qc.append(travel_operator.down, [0])
                    total_cost += travel_operator.calculate_cost()
                if verbose:
                    print(f"Route: {route}, Cost: {total_cost}")
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_route = route

                qstexp = StateTomography(qc)
                circs = qstexp.circuits()
                transpiled_circs = transpile(circs, backend=backend)
                job = backend.run(transpiled_circs, shots=1000)
                result = job.result()

                qstdata = ExperimentData(experiment=qstexp)
                manual_data = []
                for i, q_circ in enumerate(circs):
                    counts = result.get_counts(i)
                    data_dict = {
                        "counts": counts,
                        "metadata": q_circ.metadata.copy()
                    }
                    manual_data.append(data_dict)
                qstdata.add_data(manual_data)
                qstexp.analysis.run(qstdata).block_for_results()

                final_state = qstdata.analysis_results("state", dataframe=True).iloc[0].value
                
                mean_error = (mean_error * r + (1 - fidelity(final_state, initial_rho))) / (r + 1)

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