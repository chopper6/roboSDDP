#TODO: eventually the scenario should be the path to follow
import numpy as np, math



def generate(scenario, use_noise):
    A,B,C,x0, mean, covar, noise_cov_mv, noise_cov_ms = 0,0,0,0,0,0,0,0
    if scenario == 'sit still':
        A,B,C = np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]])
        x0, mean, covar = np.array([0,0]),np.array([0,0]),np.array([[1,0],[0,1]])
        noise_cov_mv, noise_cov_ms = np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])

    elif scenario == 'drift':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([1, 1]), np.array([1, 1]), np.array([[1, 0], [0, 1]])
        noise_cov_mv, noise_cov_ms = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])


    elif scenario == 'sint':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([1, 1]), np.array([1, 1]), np.array([[1, 0], [0, 1]])
        noise_cov_mv, noise_cov_ms = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])

    else: assert(False)

    #if not use_noise:
    #    noise_cov_mv, noise_cov_ms = np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])

    return A, B, C, x0, mean, covar, noise_cov_mv, noise_cov_ms

