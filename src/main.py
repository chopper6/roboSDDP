
import numpy as np, math
import plot, stoch_control, scenario

#TODO: bug on y axis, only when stoch
#TODO: double check traj off-by-one (in scen + plots)

def run(scen_choice, target_trajectory, run_name, control, gain_choice, use_noise, iters, noise_type, cap_acc, mv_noise_mag, ms_noise_mag, verbose=False):
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

        x = move(A,x,B,u, noise_cov_mv, use_noise, noise_type, cap_acc, mv_noise_mag)
        if verbose: print("posn, vel = " + str(x[0]) + ', ' + str(x[1]))
        z = measure(C,x, noise_cov_ms, use_noise, noise_type, cap_acc, ms_noise_mag)

        mean, covar, K = stoch_control.kalman_filter(mean, covar, u, z, x, A,B,C, noise_cov_mv, noise_cov_ms, control, cap_acc)
        Ks.append(K)
        xs.append(x)
        xestims.append(mean)


        #if abs(x[1]) > 1000: break

    plot.state_estim(run_name, xs,xestims,targets, control,scen_choice,gain_choice, use_noise)
    plot.key(run_name)
    plot.control_err(run_name, us, estim_errs, actual_errs, targets, control, scen_choice, gain_choice, use_noise)
    plot.Kalman_weight(run_name, Ks, control, scen_choice, gain_choice, use_noise)

    return xestims, xs, targets


def move(A,x,B,u, noise_cov_mv, use_noise, noise_type, cap_acc, mv_noise_mag):

    verbose = False
    if verbose:
        print("\nMOVE")
        print("x = " + str(x))
        print("u = " + str(u))
        print("mvmt no nosie: " + str(np.dot(A,x)) + " + " + str(np.dot(B,u)))

    if use_noise:
        R = gen_noise_matrix(mv_noise_mag, noise_type, len(x))
        noise = np.dot(noise_cov_mv,R)
    else: noise = np.array([0 for i in range(len(x))])
    x = x + np.dot(B,u)
    if cap_acc is not None and cap_acc !=0: x = enforce_acc_cap(cap_acc, x)
    x = np.dot(A,x) + noise
    #print("\nmvmt noise = " + str(noise))
    if cap_acc is not None and cap_acc !=0: x = enforce_acc_cap(cap_acc, x)
    #x = np.dot(A,x)+np.dot(B,u)+noise
    return x

def measure(C,x, noise_cov_ms, use_noise, noise_type, cap_acc, ms_noise_mag):
    if use_noise:
        R = gen_noise_matrix(ms_noise_mag, noise_type, len(x))
        noise = np.dot(noise_cov_ms,R)
    else: noise = np.array([0 for i in range(len(x))])
    #print("measure noise = " + str(noise))
    z = np.dot(C,x)+noise

    z = enforce_acc_cap(cap_acc, z)

    return z

def gen_noise_matrix(noise_mag, noise_type, size):
    if noise_type == 'normal' or noise_type == 'gaussian':
        R = np.array([np.random.normal(0, noise_mag) for i in range(size)])
    elif noise_type == 'uniform':
        R = np.array([np.random.uniform(-noise_mag/2, noise_mag/2) for i in range(size)])
    elif noise_type == 'exp' or noise_type == 'exponential':
        R = np.array([np.random.exponential(noise_mag)-noise_mag for i in range(size)]) #centered ST mean=0
    else: assert(False)

    return R

def enforce_acc_cap(cap,x):
    #can use others besides x, ie z msmt and x_estim similarly constrained
    if cap is None or cap==0 or cap=='None': return x

    x_acc, y_acc = x[2], x[5]
    if x_acc > 0:
        x_acc = min(cap, x_acc)
    else:
        x_acc = max(-1 * cap, x_acc)

    if y_acc > 0:
        y_acc = min(cap, y_acc)
    else:
        y_acc = max(-1 * cap, y_acc)
    x[2], x[5] = x_acc, y_acc
    return x

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


    elif scen_d[0] == '2D3':
        if iter==0: perceived_err, actual_err = 0,0
        else:
            x_err = (np.linalg.norm(targets[iter-1,0]-[mean[0]]))
            y_err = (np.linalg.norm(targets[iter-1,1]-[mean[3]]))
            perceived_err = np.linalg.norm(targets[iter-1]-[mean[0],mean[3]])
            #perceived_err = x_err + y_err


            x_err = (np.linalg.norm(targets[iter-1,0]-[x[0]]))
            y_err = (np.linalg.norm(targets[iter-1,1]-[x[3]]))
            #actual_err = x_err + y_err
            actual_err = np.linalg.norm(targets[iter-1]-[x[0],x[3]])
            #print("err x, y = " + str(x_err) + ', ' + str(y_err))

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

    controls = ['both', 'ideal', 'none', 'msmt','mvmt'] #, 'both', 'none', 'msmt','mvmt']
    gain_choice = ['None'] #,'PD','PI','PID'] #,'PD2', 'PID2'] #TODO: gain_choice should be set to 'None' (String-type, capital)
    iters = 40
    verbose=False
    use_noises = [True]
    target_trajectory = None
    scen_choice = '2D3 drift ms corr' #TODO: should be '2D3 '
    run_name = 'a_run' #to make separate directories, for example, during presentation

    #TODO NEW PARAMS:
    noise_type = 'normal' #uniform and exp allowed
    cap_acc = 1 # set to 0 or None to ignore
    mv_noise_mag, ms_noise_mag = .02, 1 #rough suggestions

    for control in controls:
        for gain in gain_choice:
            for use_noise in use_noises:
                print("\n\nRUN: " + control + ', ' + gain + ', noise '  + str(use_noise) + "\n")
                xestims, xs, targets = run(scen_choice, target_trajectory, run_name, control, gain, use_noise, iters, noise_type, cap_acc, mv_noise_mag, ms_noise_mag, verbose=verbose)

    # a run without noise, just to check
    xestims, xs, targets = run(scen_choice, target_trajectory, run_name, 'both', 'None', False, iters, noise_type, cap_acc,mv_noise_mag, ms_noise_mag, verbose=verbose)


    # NOTES (poss obsolete by now...):
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
