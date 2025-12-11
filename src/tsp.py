import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class TSPInstance:
    def __init__(self, n_cities, coords=None):
        """
        Initialize TSP instance with city coordinates.
        
        Args:
        - coords: numpy array of shape (n_cities, 2) with city coordinates
        """
        self.n_cities = n_cities
        if coords is not None:
            self.coords = coords
        else:
            self.coords = np.random.rand(n_cities, 2) * 100

        self.dist_matrix = self.compute_distance_matrix()

        self.graph = self.create_tsp_graph(self.coords, self.dist_matrix)

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
        G = nx.Graph()

        # Add nodes with positions
        for i in range(n_cities):
            G.add_node(i, pos=coords[i])

        # Add edges with weights
        for i in range(n_cities):
            for j in range(i + 1, n_cities):
                G.add_edge(i, j, weight=dist_matrix[i, j])

        return G
    
    def compute_distance_matrix(self):
        """
        Compute the Euclidean distance matrix between cities.
        
        Returns:
        - dist_matrix: numpy array of shape (n_cities, n_cities)
        """
        dist_matrix = np.zeros((self.n_cities, self.n_cities))
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
        nx.draw_networkx_edges(self.graph, pos, edge_color='skyblue', width=2, alpha=0.7)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightcoral', node_size=500, 
                               edgecolors='darkred', linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight='bold', 
                                font_color='black')
        
        plt.title('TSP Instance: Cities and Connections', fontsize=16, fontweight='bold')
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')  # Ensure equal aspect ratio
        plt.tight_layout()
        plt.show()

