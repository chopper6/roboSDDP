
import numpy as np, math
import plot, stoch_control, scenario



def run(scen_choice, target_trajectory, run_name, control, gain_choice, use_noise, iters, verbose=False):
    xs, xestims, estim_errs, actual_errs, us, Ks = [],[],[], [], [],[]
    A,B,C, x, mean, covar, noise_cov_mv, noise_cov_ms, targets = scenario.generate(scen_choice, use_noise, iters)
    xs.append(x)
    xestims.append(mean)
    #estim_errs.append(0)
    #actual_errs.append(0)
    if target_trajectory is not None: targets = target_trajectory # TODO: override with Jeremie's path

    u = [0 for i in range(len(x))]
    #print_shapes([A,B,C,x,mean,covar], 'A,B,C,x,mean,cov')
    for i in range(1,iters+1):
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

    plot.state_estim(run_name, xs,xestims,targets, control,scen_choice,gain_choice, use_noise)
    plot.key(run_name)
    plot.control_err(run_name, us, estim_errs, actual_errs, targets, control, scen_choice, gain_choice, use_noise)
    plot.Kalman_weight(run_name, Ks, control, scen_choice, gain_choice, use_noise)

    return xestims, xs, targets


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
    if noise_type == 'add':
        x = np.dot(A,x) + noise
    elif noise_type == 'mult':
        x = np.dot(np.dot(A,x), noise)
    else: assert(False)
    #x = np.dot(A,x)+np.dot(B,u)+noise
    return x

def measure(C,x, noise_cov_ms, use_noise):
    if use_noise:
        R = np.array([np.random.normal(0,4) for i in range(len(x))]) #TODO: magnitude should really be in covar matrix
        noise = np.dot(R, noise_cov_ms)
    else: noise = np.array([0 for i in range(len(x))])
    z = np.dot(C,x)+noise
    return z



def calc_err(scenario,mean, x, iter, targets):
    #where mean is x_hat, ie using estim err
    # MSE
    scen_d = scenario.split(' ')

    if False: #old
        if scenario == 'sit still 2' or scenario == 'sit still': err = (mean[0]) #np.linalg.norm(mean)/len(mean)
        elif scenario == 'sint': err = abs(math.sin(iter)-mean[1])
        elif scenario == 'drift' or scenario == 'rude drift': err = abs(iter - mean[0])
        else: assert(False)
        return err

    elif scen_d[0] == '2D':
        if iter==0: perceived_err, actual_err = 0,0
        else:
            perceived_err = np.linalg.norm(targets[iter-1]-[mean[0],mean[2]])
            actual_err = np.linalg.norm(targets[iter-1]-[x[0],x[2]])
            #perceived_err = abs(targets[iter,0] - mean[0])+abs(targets[iter,1]-mean[2])
            #actual_err = abs(targets[iter,0] - x[0]) + abs(targets[iter,1] - x[2])
        #print('\ntargets, x_hat, x, estim err, actual err')
        #print(targets[iter],[mean[0],mean[2]], [x[0],x[2]], perceived_err, actual_err)

    else:
        if iter==0: perceived_err, actual_err = 0,0
        else:
            perceived_err = np.linalg.norm(targets[iter-1]-mean[0])
            actual_err = np.linalg.norm(targets[iter-1]-x[0])

    perceived_err, actual_err = round(perceived_err,6), round(actual_err,6)
    return perceived_err, actual_err



if __name__ == "__main__":
    # scenarios: sit still, drift
    # controls: none, both, msmt, mvmt

    controls = ['both'] #, 'ideal', 'none', 'msmt','mvmt']
    gain_choice = ['None','P','PD','PI','PID'] #,'PD2', 'PID2'] #TODO: gain_choice should be set to 'None' (String-type, capital)
    iters = 16
    verbose=False
    use_noises = [True]
    target_trajectory = None
    scen_choice = '2D3 drift' #'2D rd path with drift' #TODO: should be '2D3 '
    run_name = 'a_run' #to make separate directories, for example, during presentation

    for control in controls:
        for gain in gain_choice:
            for use_noise in use_noises:
                print("\n\nRUN: " + control + ', ' + gain + ', noise '  + str(use_noise) + "\n")
                xestims, xs, targets = run(scen_choice, target_trajectory, run_name, control, gain, use_noise, iters, verbose=verbose)

    # a run without noise, just to check
    xestims, xs, targets = run(scen_choice, target_trajectory, run_name, 'both', 'P', False, iters, verbose=verbose)


    # NOTES:
    # Line 127: xestims, xs, targets = run() is the main function you should use

    # for 2D, x = [x_position, x_velocity, y_position, y_velocity], same for xestims (what the robo believes)

    # if target_trajectory != None, it will be used instead of the scenario (scen_choice)
    # targets should be in form: [[x1,y1], [x2,y2] ... [xn,yn], [xn,yn]], may need to be an np.array()
    # and yes, repeat the last coordinates twice!!! It's sloppy I know...
    # first coordinates (x1,y1) should probably be the same as the init position of the object

    # scen_choice will still effect the dynamics (basically just ensure it is a 2D one)
    # scen_choice also effects the noise choice, which we can play with later
    # look at scenario.py for more

    # I have not figured out how to incorporate PID control, so "gain_choice" parameter is irrelevant

    # on plots:
    # A_KEY.png explains the basic format
    # possible control_choices = 'ideal','both', 'none', 'msmt','mvmt'
    # the "Kalman Weight" plots won't make much sense, possibly something to work on...

    # things to maybe try:
    # use measurement control for position and movement control for velocity
    # weirder noises, depending on how things go
