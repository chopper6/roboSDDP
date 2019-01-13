import numpy as np, math
import plot, stoch_control, scenario

def run(scen_choice, control, gain_choice, use_noise, iters):
    verbose = False
    xs, xestims, errs, us = [],[],[], []
    A,B,C, x, mean, covar, noise_cov_mv, noise_cov_ms = scenario.generate(scen_choice, use_noise)
    #print_shapes([A,B,C,x,mean,covar], 'A,B,C,x,mean,cov')
    for i in range(iters):
        errs.append(calc_err(scen_choice, mean))
        u = stoch_control.update_u(scen_choice,mean,control,gain_choice, errs) #where mean is x_estim #TODO check order
        us.append(u)

        x = move(A,x,B,u, use_noise)
        if verbose: print("posn, vel = " + str(x[0]) + ', ' + str(x[1]))
        z = measure(C,x, use_noise)

        mean, covar = stoch_control.kalman_filter(mean, covar, u, z, A,B,C, noise_cov_mv, noise_cov_ms, control)
        xs.append(x)
        xestims.append(mean)
    plot.state_estim(xs,xestims, control,scen_choice,gain_choice, use_noise)
    plot.control_err(us, errs, control, scen_choice, gain_choice, use_noise)
    return


def move(A,x,B,u, use_noise):
    verbose = False
    if verbose:
        print("\nMOVE")
        print("x = " + str(x))
        print("u = " + str(u))
        print("mvmt no nosie: " + str(np.dot(A,x)) + " + " + str(np.dot(B,u)))
    if use_noise: noise = np.array([np.random.normal(0,1) for i in range(len(x))])
    else: noise = np.array([0 for i in range(len(x))])
    x = np.dot(A,x)+np.dot(B,u)+noise
    return x

def measure(C,x, use_noise):
    if use_noise: noise = np.array([np.random.normal(0,1) for i in range(len(x))])
    else: noise = np.array([0 for i in range(len(x))])
    z = np.dot(C,x)+noise
    return z


def calc_err(scenario,mean):
    #where mean is x_hat
    # MSE
    err = (mean[0] + mean[1]) #np.linalg.norm(mean)/len(mean)
    return err



if __name__ == "__main__":
    # scenarios: sit still, drift
    # controls: none, both, msmt, mvmt

    controls = ['both'] #, 'none','msmt','mvmt']
    gain_choice = ['PD','PI','PID'] #['P','PD','PI','PID']
    iters = 80
    use_noises = [False,True]
    #control = 'mvmt'
    scen_choice = 'drift'
    for control in controls:
        for gain in gain_choice:
            for use_noise in use_noises:
                run(scen_choice, control, gain, use_noise, iters)