import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

sigma_x, sigma_y = 0.25, 0.25
sigma_r = 0.3
true_vehicle_pos = np.array([0.5, 0.5])


def generate_landmarks(K):
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    landmarks = np.vstack((np.cos(angles), np.sin(angles))).T
    return landmarks


def generate_measurements(landmarks, vehicle_pos, sigma_r):
    measurements = np.linalg.norm(landmarks - vehicle_pos, axis=1)
    measurements += np.random.normal(0, sigma_r, measurements.shape)
    return np.maximum(measurements, 0)



def map_objective(candidate_pos, landmarks, measurements, sigma_r, sigma_x, sigma_y):
    prior_term = -0.5 * (((candidate_pos[0] / sigma_x) ** 2) + ((candidate_pos[1] / sigma_y) ** 2))
    range_estimates = np.linalg.norm(landmarks - candidate_pos, axis=1)
    likelihood_term = np.sum(norm.logpdf(measurements, range_estimates, sigma_r))

    return -(prior_term + likelihood_term)


def plot_map_contours(K_values, true_pos, sigma_r, sigma_x, sigma_y):
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)

    for K in K_values:
        landmarks = generate_landmarks(K)
        measurements = generate_measurements(landmarks, true_pos, sigma_r)

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = map_objective([X[i, j], Y[i, j]], landmarks, measurements, sigma_r, sigma_x, sigma_y)

        plt.figure()
        contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
        plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', marker='o')
        plt.scatter(true_pos[0], true_pos[1], c='blue', marker='+')
        plt.title(f'MAP Equilevel Contours with K={K} Landmarks')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(contour)
        plt.axis('equal')
        plt.show()


K_values = [1, 2, 3, 4]

plot_map_contours(K_values, true_vehicle_pos, sigma_r, sigma_x, sigma_y)
