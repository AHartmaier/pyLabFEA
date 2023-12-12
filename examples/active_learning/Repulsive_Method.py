#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is designed to generate and distribute points evenly on the surface of a unit sphere in multi-dimensional
spaces (specifically 3D and 6D). It includes functions for generating random points on a sphere, calculating distances 
to the k-th nearest neighbors, and applying a repulsion algorithm to spread the points more uniformly across the sphere's
surface.

The script demonstrates the effect of the repulsion algorithm by plotting histograms of the k-nearest neighbor distances 
before and after applying the algorithm, providing a visual representation of the points' distribution on the sphere's 
surface.

"""
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
def generate_random_points_on_sphere(n_points, dimensions):
    """Generate n_points random points on the surface of a unit sphere in given dimensions."""
    if dimensions == 3:
        # For 3D, use spherical coordinates
        phi = np.random.uniform(0, np.pi * 2, n_points)
        cos_theta = np.random.uniform(-1, 1, n_points)
        theta = np.arccos(cos_theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack((x, y, z))
    elif dimensions == 6:
        # For 6D, use a general approach for higher dimensions
        gaussians = np.random.normal(size=(n_points, dimensions))
        norm = np.linalg.norm(gaussians, axis=1)
        return gaussians / norm[:, np.newaxis]
    else:
        raise ValueError("Only 3D and 6D spheres are supported.")

def calculate_knn_distances(points, k):
    """Calculate the distance to the kth nearest neighbor for each point."""
    distances = distance_matrix(points, points)
    np.fill_diagonal(distances, np.inf)  # Ignore distance to self
    sorted_distances = np.sort(distances, axis=1)
    return sorted_distances[:, k-1]  # kth nearest neighbor distance

def apply_repulsion(points, k, iterations, learning_rate):
    """Apply repulsion to maximize the distance to the kth nearest neighbor."""
    for _ in range(iterations):
        distances = distance_matrix(points, points)
        np.fill_diagonal(distances, np.inf)
        indices = np.argsort(distances, axis=1)[:, :k]  # Indices of k nearest neighbors
        # Initialize displacement vector
        displacement = np.zeros_like(points)
        # Calculate repulsion from each of the k nearest neighbors
        for i, point in enumerate(points):
            neighbors = points[indices[i]]
            diff = point - neighbors  # Vector from neighbors to point
            distances_to_neighbors = distances[i, indices[i]].reshape(-1, 1)
            repulsion = diff / distances_to_neighbors ** 2  # Repulsion proportional to inverse square of distance
            displacement[i] = repulsion.sum(axis=0)
        # Update points with displacement
        points += learning_rate * displacement
        # Normalize to keep points on the sphere surface
        norms = np.linalg.norm(points, axis=1).reshape(-1, 1)
        points /= norms
    return points

# Parameters
n_points = 100  # Number of points
dimensions = 6
k = 5
iterations = 60  # Number of iterations for the repulsion algorithm
learning_rate = 0.01  # Learning rate for the displacement
# Generate initial random points on the surface of a 3D sphere
initial_points_6d = generate_random_points_on_sphere(n_points, dimensions)
initial_knn_distances_3d = calculate_knn_distances(initial_points_6d, k)
# Apply the repulsion method
final_points_3d = apply_repulsion(np.copy(initial_points_6d), k, iterations, learning_rate)
final_knn_distances_3d = calculate_knn_distances(final_points_3d, k)
# Plotting histograms of k-nearest neighbor distances
plt.hist(initial_knn_distances_3d, bins=20, alpha=0.7, label='Initial (6D Sphere)')
plt.hist(final_knn_distances_3d, bins=20, alpha=0.7, label='Final (6D Sphere)')
plt.xlabel('5 Nearest Neighboring Distance')
plt.ylabel('Frequency')
plt.legend()
plt.show()

