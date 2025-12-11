import numpy as np
import matplotlib.pyplot as plt

def generate_tsp_instance(n_cities):
    """
    Generate a TSP instance with n_cities.
    
    Returns:
    - coords: numpy array of shape (n_cities, 2) with city coordinates
    - dist_matrix: numpy array of shape (n_cities, n_cities) with distances
    """
    # Generate random coordinates in [0, 100] x [0, 100]
    coords = np.random.rand(n_cities, 2) * 100
    
    # Compute Euclidean distance matrix
    dist_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    return coords, dist_matrix

def plot_tsp_instance(coords):
    """
    Plot the TSP cities.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=100)
    for i, (x, y) in enumerate(coords):
        plt.text(x + 1, y + 1, f'City {i}', fontsize=12)
    plt.title('TSP Cities')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()