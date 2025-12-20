import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import permutations


class TSPInstance:
    def __init__(self, n_cities, coords=None, asymmetric=False, dist_matrix=None):
        """
        Initialize TSP instance with city coordinates.
        
        Args:
        - n_cities: number of cities
        - coords: numpy array of shape (n_cities, 2) with city coordinates
        - asymmetric: if True, generate asymmetric TSP instance
        - dist_matrix: custom distance matrix (optional)
        """
        self.n_cities = n_cities
        self.asymmetric = asymmetric
        
        if coords is not None:
            self.coords = coords
        else:
            self.coords = np.random.rand(n_cities, 2) * 100

        if dist_matrix is not None:
            self.dist_matrix = dist_matrix
        else:
            self.dist_matrix = self.compute_distance_matrix()

        self.graph = self.create_tsp_graph(self.coords, self.dist_matrix)

        self.calculate_allowed_routes()

    def calculate_allowed_routes(self):
        """
        Calculate all allowed routes (permutations) for the TSP instance.

        Returns:
        - allowed_routes: List of lists, each representing a valid route
        """

        cities = list(range(self.n_cities))
        all_routes = list(permutations(cities))
        allowed_routes = []
        for route in all_routes:
            if route[0] == 0:
                route += (0,)  # Return to starting city
                allowed_routes.append(route)
        self.allowed_routes = allowed_routes

    def create_tsp_graph(self, coords, dist_matrix):
        """
        Create a NetworkX graph from TSP instance.

        Parameters:
        - coords: numpy array of shape (n_cities, 2) with city coordinates
        - dist_matrix: numpy array of shape (n_cities, n_cities) with distances

        Returns:
        - G: NetworkX graph with nodes as cities and edges with distance weights
        """
        n_cities = len(coords)
        if self.asymmetric:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        # Add nodes with positions
        for i in range(n_cities):
            G.add_node(i, pos=coords[i])

        # Add edges with weights
        if self.asymmetric:
            # Add all directed edges
            for i in range(n_cities):
                for j in range(n_cities):
                    if i != j:
                        G.add_edge(i, j, weight=dist_matrix[i, j])
        else:
            # Add undirected edges
            for i in range(n_cities):
                for j in range(i + 1, n_cities):
                    G.add_edge(i, j, weight=dist_matrix[i, j])

        return G
    
    def compute_distance_matrix(self):
        """
        Compute the distance matrix between cities.
        For symmetric TSP: Euclidean distances.
        For asymmetric TSP: Euclidean distances with random asymmetric perturbations.
        
        Returns:
        - dist_matrix: numpy array of shape (n_cities, n_cities)
        """
        dist_matrix = np.zeros((self.n_cities, self.n_cities))
        if self.asymmetric:
            # Generate asymmetric distances based on coordinates with random perturbations
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    if i != j:
                        base_dist = np.linalg.norm(self.coords[i] - self.coords[j])
                        # Add asymmetry by applying random multipliers
                        dist_matrix[i, j] = base_dist * np.random.uniform(0.7, 1.4)
                    # else: 0 (no self-loop)
        else:
            # Symmetric Euclidean distances
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    dist_matrix[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])
        return dist_matrix
    
    def plot_tsp_instance(self):
        """
        Plot the TSP cities and the graph with enhanced styling.
        """
        # Use a modern style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')  # fallback if seaborn not available
        
        plt.figure(figsize=(10, 8))
        
        # Get positions from the graph
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Draw the graph edges with transparency
        if self.asymmetric:
            nx.draw_networkx_edges(self.graph, pos, edge_color='skyblue', width=2, alpha=0.7, 
                                   arrows=True, arrowsize=20, arrowstyle='->')
        else:
            nx.draw_networkx_edges(self.graph, pos, edge_color='skyblue', width=2, alpha=0.7)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightcoral', node_size=500, 
                               edgecolors='darkred', linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight='bold', 
                                font_color='black')
        
        title_text = 'Asymmetric TSP Instance: Cities and Directed Connections' if self.asymmetric else 'TSP Instance: Cities and Connections'
        plt.title(title_text, fontsize=16, fontweight='bold')
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')  # Ensure equal aspect ratio
        plt.tight_layout()
        plt.show()

