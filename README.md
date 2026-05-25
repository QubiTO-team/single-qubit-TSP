# Solving the Travelling Salesman Problem Using a Single Qubit

## Overview

This repository contains a Python and Qiskit-based implementation of the quantum algorithm detailed in the paper *"Solving The Travelling Salesman Problem Using A Single Qubit"* by Goswami et al..

Traditional quantum approaches to the Travelling Salesman Problem (TSP) rely on Quadratic Unconstrained Binary Optimization (QUBO) or heavy gate-based circuits, which are highly resource-expensive and scale poorly on Noisy Intermediate-Scale Quantum (NISQ) devices. This project implements an alternative framework: modeling the TSP as a discrete quantum Brachistochrone problem encoded entirely onto a single qubit.

By encoding cities as distinct states on the Bloch sphere and distances as geometric paths, the algorithm leverages quantum superposition to traverse multiple routes simultaneously. Quantum optimal control techniques are then used to steer the system toward the optimal Hamiltonian cycle.

## Repository Structure

The codebase is modularized to separate the classical graph generation, the geometric quantum encoding, and the physical execution on simulated or real quantum backends.

* **`tsp.py`**: Contains the `TSPInstance` class. Handles the generation of symmetric and asymmetric classical TSP instances, distance matrix computation, and baseline classical approximations (via NetworkX).
* **`bloch_sphere_encoding.py`**: Contains the `BlochSphereEncoder` class. Manages the geometric mapping of the classical TSP onto the Bloch sphere. It places primary city states on the equator and scales relative distances onto geodesic curves connected to the pole .


* **`tsp_bloch.py`**: The core quantum logic module. Defines the `TravelOperator` to compute the required up ($U^u$) and down ($U^d$) rotational matrices for traversal . Contains the `TSPBlochInstance` class which manages circuit execution, state tomography, and brute-force path validation.


* **`utility.py`**: Helper functions for matrix rescaling and Cartesian-to-Bloch coordinate transformations.

## Dependencies

This project relies on standard quantum computing and data science libraries.

* `qiskit` (Circuit construction and Statevector simulation)
* `qiskit-experiments` (State tomography)
* `numpy` (Matrix operations and transformations)
* `networkx` (Classical graph operations)
* `matplotlib` & `kaleidoscope` (Visualizing graphs and Bloch sphere states)

## Running on Lagrange Quantum Computer

**Important:** In order to execute the code on the Lagrange quantum
computer managed by Fondazione LINKS and hosted at Politecnico di
Torino, you are required to have a valid token file. Please ensure your
token file is correctly set up in your environment before attempting to
submit jobs to the hardware.

## Current Capabilities

* **Classical Graph Generation:** Supports both uniform random symmetric instances and directionally weighted asymmetric instances.
* 
**Deterministic Encoding:** Successfully scales distance matrices and executes the inverse-stereographic-like projection required to map cities to `Statevector` objects without spatial overlap .


* 
**Brute-Force Quantum Validation:** Simulates individual paths sequentially using rotational operators to validate the time functional ($T$) and confirm the geometric encoding strictly correlates with classical path distance .



## Roadmap / Next Steps

The repository is currently undergoing active development to implement the final phase of the algorithm: **Quantum Parallelism and Optimal Control**.

1. 
**Superposition Initialization:** Construct the forward-pass evolution to apply weighted linear combinations of rotation operators, establishing the interference pattern of all possible paths in a single pass .


2. 
**State Tomography & Decoding:** Implement the post-processing module to invert the overlap matrix ($\mathcal{E}\mathbf{X} = \mathbf{K}$) and decode the final measured density matrix back into a classical route .


3. 
**SPSA Optimization:** Integrate the Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer to systematically adjust the rotational coefficients and minimize the classical distance ($D$) of the decoded route.



## References

* Goswami, K., Veereshi, G. A., Schmelcher, P., & Mukherjee, R. (2024). *Solving The Travelling Salesman Problem Using A Single Qubit*. arXiv:2407.17207v2 [quant-ph].