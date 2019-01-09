import numpy as np, os
from matplotlib import pyplot as plt

def run():
    iters = 10
    scenario = 'sit still'
    xs, xestims = np.array([]),np.array([])
    A,B,C, x, mean, covar = gen_scenario(scenario)
    for i in range(iters):
        u = update_u(scenario,mean) #where mean is x_estim
        x = move(A,x,B,u)
        z = measure(C,x)
        mean, covar = kalman_filter(mean, covar, u, z, A,B,C)
        np.append(xs,x)
        np.append(xestims,mean)
    plot_run(xs,xestims)
    return



def plot_run(xs, xestims):
    dirname = os.path.dirname(__file__)
    plot_dir = os.path.join(dirname,'/plots/')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for i in range(len(xs[0])):
        param = xs[:,i]
        param_estim = xestims[:,i]
        ticks = [j for j in range(len(param))]
        plt.plot(ticks,param, color='blue')
        plt.plot(ticks,param_estim, color='green')
        plt.savefig(plot_dir + '/param' + str(i))


def kalman_filter(mean, covar, u, z, A, B,C):
    #currently assume the cov for the noise terms are 1
    mean_a_priori = np.dot(A,mean) + np.dot(B,u)
    covar_a_priori = np.dot(np.dot(A,covar),A.T) + 1
    K_piece = np.dot(np.dot(C, covar_a_priori), C.T) + 1
    if len(K_piece) > 1: inv = np.linalg.inv(K_piece)
    else: inv = 1/K_piece
    K = np.dot(covar_a_priori,C) + inv
    mean_estim = mean_a_priori + np.dot(K,z - np.dot(C,mean_a_priori))
    I = np.eye(len(covar)) #TODO: check this
    covar_estim = np.dot( I - np.dot(K,C), covar)

    return mean_estim, covar_estim


def move(A,x,B,u):
    noise = np.array([np.random.normal(0,1) for i in range(len(x))])
    x = np.dot(A,x)+np.dot(B,u)+noise
    return x

def measure(C,x):
    noise = np.array([np.random.normal(0,1) for i in range(len(C))]) #TODO: poss len(C[0]) instead?
    z = np.dot(C,x)+noise
    return z

def update_u(scenario, x_estim):
    u=[0,0]
    if scenario == 'sit still':
        u = np.array([-x_estim],[0])
    else: assert(False)
    return u


def gen_scenario(scenario):
    A,B,C,x0, mean, covar = 0,0,0,0,0,0
    if scenario == 'sit still':
        A,B,C = np.array([1,0]),np.array([1,0]),np.array([1,0])
        x0, mean, covar = np.array([0,0]),np.array([0,0]),np.array([0,0])
    else: assert(False)
    return A, B, C, x0, mean, covar



if __name__ == "__main__":
    run()