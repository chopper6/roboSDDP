import numpy as np, os
from matplotlib import pyplot as plt

def run(scenario, control, iters):
    xs, xestims = [],[]
    A,B,C, x, mean, covar, noise_cov_mv, noise_cov_ms = gen_scenario(scenario)
    #print_shapes([A,B,C,x,mean,covar], 'A,B,C,x,mean,cov')
    for i in range(iters):
        u = update_u(scenario,mean,control) #where mean is x_estim #TODO check order
        x = move(A,x,B,u)
        z = measure(C,x)
        mean, covar = kalman_filter(mean, covar, u, z, A,B,C, noise_cov_mv, noise_cov_ms, control)
        xs.append(x)
        xestims.append(mean)
    plot_run(xs,xestims, control,scenario)
    return


def print_shapes(objs, titles):
    print("Shapes of " + str(titles))
    for obj in objs:
        print(np.shape(obj))

def plot_run(xs, xestims, control, scenario):
    xs, xestims = np.array(xs), np.array(xestims)

    if scenario == 'sit still': titles = ['x','y']
    elif scenario == 'drift': titles = ['position','velocity']
    else: assert(False)

    dirname = os.getcwd()
    plot_dir = dirname + '/output/plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for i in range(len(xs[0])):
        param = xs[:,i]
        param_estim = xestims[:,i]
        n = len(param)
        ymax = max(max(param),max(param_estim))
        ymin = min(min(param),min(param_estim))
        yabs = max(abs(ymax), abs(ymin))
        MSE = np.linalg.norm(param-param_estim)/n
        MSD = np.linalg.norm(param)/n
        ticks = [j for j in range(len(param))]
        plt.text(0,ymax,"MSE = " + str(MSE), color='red')
        plt.text(0,ymax-yabs/8,"Mean dist from origin = " + str(MSD), color='red')
        plt.plot(ticks,param, color='blue', label='actual')
        plt.plot(ticks,param_estim, color='green', label='estimated')
        plt.title('Estimated vs Actual ' + str(titles[i]))
        plt.legend(loc = 'lower left')
        plt.savefig(plot_dir + '/control_' + str(control) + '_param_' + titles[i])
        plt.clf()


def kalman_filter(mean, covar, u, z, A, B,C, noise_cov_mv, noise_cov_ms, control):
    #currently assume the cov for the noise terms are 1
    mean_a_priori = np.dot(A,mean) + np.dot(B,u)
    covar_a_priori = np.dot(np.dot(A,covar),A.T) + noise_cov_mv
    K_piece = np.dot(np.dot(C, covar_a_priori), C.T) + noise_cov_ms
    if len(K_piece) > 1: inv = np.linalg.inv(K_piece)
    else:
        assert(False) #changed to only matrices
        inv = 1/K_piece
    K = np.dot( np.dot(covar_a_priori,C.T) , inv)
    mean_estim = mean_a_priori + np.dot(K,z - np.dot(C,mean_a_priori))
    I = np.eye(len(covar)) #TODO: check this
    covar_estim = np.dot( I - np.dot(K,C), covar)

    if control == 'none': mean = [0,0] #ie origin, note assumes 2d
    elif control == 'both': mean=mean_estim
    elif control == 'mvmt': mean=mean_a_priori
    elif control == 'msmt': mean=z
    else: assert(False)

    return mean, covar_estim


def move(A,x,B,u):
    noise = np.array([np.random.normal(0,1) for i in range(len(x))])
    x = np.dot(A,x)+np.dot(B,u)+noise
    return x

def measure(C,x):
    noise = np.array([np.random.normal(0,1) for i in range(len(C))]) #TODO: poss len(C[0]) instead?
    z = np.dot(C,x)+noise
    return z

def update_u(scenario, x_estim, control):
    u=[0,0]
    if control == 'none': return u
    if scenario == 'sit still':
        u = -1*x_estim
    elif scenario == 'drift':
        gain = .1
        u = [0,-1*gain*(x_estim[1]+x_estim[0])] #control vel based on estim position
    else: assert(False)
    return u


def gen_scenario(scenario):
    A,B,C,x0, mean, covar, noise_cov_mv, noise_cov_ms = 0,0,0,0,0,0,0,0
    if scenario == 'sit still':
        A,B,C = np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]])
        x0, mean, covar = np.array([0,0]),np.array([0,0]),np.array([[1,0],[0,1]])
        noise_cov_mv, noise_cov_ms = np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])

    elif scenario == 'drift':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([0, 1]), np.array([0, 1]), np.array([[1, 0], [0, 1]])
        noise_cov_mv, noise_cov_ms = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])

    else: assert(False)
    return A, B, C, x0, mean, covar, noise_cov_mv, noise_cov_ms



if __name__ == "__main__":
    # scenarios: sit still, drift
    # controls: none, both, msmt, mvmt

    controls = ['none','both']#,'msmt','mvmt']
    iters = 40
    #control = 'mvmt'
    scenario = 'drift'
    for control in controls:
        run(scenario, control, iters)