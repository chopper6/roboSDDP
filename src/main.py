# TODO: in general, seem to be over-complicating with the velocity 1-step-behind to control position game

import numpy as np, math
import plot, stoch_control, scenario

def run(scen_choice, control, gain_choice, use_noise, iters, verbose=False):
    xs, xestims, estim_errs, actual_errs, us, Ks = [],[],[], [], [],[]
    A,B,C, x, mean, covar, noise_cov_mv, noise_cov_ms, targets = scenario.generate(scen_choice, use_noise, iters)
    u = [0 for i in range(len(x))]
    #print_shapes([A,B,C,x,mean,covar], 'A,B,C,x,mean,cov')
    for i in range(iters):
        estim_err, actual_err = calc_err(scen_choice, mean, x, i, targets)
        estim_errs.append(estim_err)
        actual_errs.append(actual_err)
        u = stoch_control.update_u(scen_choice,mean,control,gain_choice, estim_errs, u, x,i, targets) #where mean is x_estim #TODO check order

        us.append(u)
        if verbose:
            print("\nu = " + str(u))
            print("x_hat = " + str(mean))
            print("x = " + str(x))

        x = move(A,x,B,u, noise_cov_mv, use_noise)
        if verbose: print("posn, vel = " + str(x[0]) + ', ' + str(x[1]))
        z = measure(C,x, noise_cov_ms, use_noise)

        mean, covar, K = stoch_control.kalman_filter(mean, covar, u, z, x, A,B,C, noise_cov_mv, noise_cov_ms, control)
        Ks.append(K)
        xs.append(x)
        xestims.append(mean)
        #if abs(x[1]) > 1000: break

    plot.state_estim(xs,xestims,targets, control,scen_choice,gain_choice, use_noise)
    plot.key()
    plot.control_err(us, estim_errs, actual_errs, targets, control, scen_choice, gain_choice, use_noise)
    plot.Kalman_weight(Ks, control, scen_choice, gain_choice, use_noise)

    return


def move(A,x,B,u, noise_cov_mv, use_noise):
    noise_type = 'add'
    verbose = False
    if verbose:
        print("\nMOVE")
        print("x = " + str(x))
        print("u = " + str(u))
        print("mvmt no nosie: " + str(np.dot(A,x)) + " + " + str(np.dot(B,u)))
    if use_noise:
        R = np.array([np.random.normal(0,1) for i in range(len(x))])
        if noise_type=='mult':
            R = np.array([[np.random.normal(0,1) for i in range(len(x))] for j in range(len(x))])
        noise = np.dot(R,noise_cov_mv)

    else: noise = np.array([0 for i in range(len(x))])
    x = x + np.dot(B,u)
    if noise_type == 'add': x = np.dot(A,x) + noise
    elif noise_type == 'mult':
        x = np.dot(np.dot(A,x), noise)
    else: assert(False)
    #x = np.dot(A,x)+np.dot(B,u)+noise
    return x

def measure(C,x, noise_cov_ms, use_noise):
    if use_noise:
        R = np.array([np.random.normal(0,4) for i in range(len(x))])
        noise = np.dot(R, noise_cov_ms)
    else: noise = np.array([0 for i in range(len(x))])
    z = np.dot(C,x)+noise
    return z


def calc_err(scenario,mean, x, iter, targets):
    #where mean is x_hat, ie using estim err
    # MSE

    if False: #old
        if scenario == 'sit still 2' or scenario == 'sit still': err = (mean[0]) #np.linalg.norm(mean)/len(mean)
        elif scenario == 'sint': err = abs(math.sin(iter)-mean[1])
        elif scenario == 'drift' or scenario == 'rude drift': err = abs(iter - mean[0])
        else: assert(False)
        return err

    elif scenario == '2D drift':
        perceived_err = np.linalg.norm(targets[iter]-[mean[0],mean[2]])
        actual_err = np.linalg.norm(targets[iter]-[x[0],x[2]])
        #perceived_err = abs(targets[iter,0] - mean[0])+abs(targets[iter,1]-mean[2])
        #actual_err = abs(targets[iter,0] - x[0]) + abs(targets[iter,1] - x[2])

    else:
        perceived_err = np.linalg.norm(targets[iter]-mean[0])
        actual_err = np.linalg.norm(targets[iter]-x[0])

    return perceived_err, actual_err



if __name__ == "__main__":
    # scenarios: sit still, drift
    # controls: none, both, msmt, mvmt

    controls = ['ideal','both', 'none', 'msmt','mvmt']
    gain_choice = ['P'] #,'PD','PI','PID'] #,'PD2', 'PID2']
    iters = 40
    verbose=False
    use_noises = [True]
    #control = 'mvmt'
    scen_choice = '2D rd path with drift'
    for control in controls:
        for gain in gain_choice:
            for use_noise in use_noises:
                print("\n\nRUN: " + control + ', ' + gain + ', noise '  + str(use_noise) + "\n")
                run(scen_choice, control, gain, use_noise, iters, verbose=verbose)

    run(scen_choice, 'both', 'P', False, iters, verbose=verbose)