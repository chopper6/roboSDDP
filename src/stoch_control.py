import numpy as np, math
import main


def kalman_filter(mean, covar, u, z, x, A, B,C, noise_cov_mv, noise_cov_ms, control, cap_acc):
    #currently assume the cov for the noise terms are 1
    mean_a_priori = mean + np.dot(B,u)
    mean_a_priori = np.dot(A,mean_a_priori)
    #print("mean_a_priori = " + str(mean_a_priori))

    covar_a_priori = np.dot(np.dot(A,covar),A.T) + noise_cov_mv
    K_piece = np.dot(np.dot(C, covar_a_priori), C.T) + noise_cov_ms
    if len(K_piece) > 1: inv = np.linalg.inv(K_piece)
    else:
        assert(False) #changed to only matrices
        inv = 1/K_piece
    K = np.dot( np.dot(covar_a_priori,C.T) , inv)
    #print("Kal filter K = " + str(K))
    mean_estim = mean_a_priori + np.dot(K,z - np.dot(C,mean_a_priori))
    I = np.eye(len(covar))
    covar_estim = np.dot( I - np.dot(K,C), covar)


    if cap_acc is not None and cap_acc !=0:
        mean_estim = main.enforce_acc_cap(cap_acc, mean_estim)
        mean_a_priori = main.enforce_acc_cap(cap_acc, mean_a_priori)

    if control == 'none': mean = [0 for i in range(len(mean))] #ie origin, note assumes 2d
    elif control == 'both': mean=mean_estim
    elif control == 'mvmt': mean=mean_a_priori
    elif control == 'msmt': mean=z
    elif control == 'ideal': mean=x
    else: assert(False)

    return mean, covar_estim, K


def update_u(scenario, x_estim, control, gain_choice,errs, u, x, iter, targets):
    new_u=[0 for i in range(len(u))]
    scen_d = scenario.split(' ')
    p_w, i_w, d_w = .5,.5,.5

    cap_acc = False #TODO cut it, have it elsewhere now (not change in acc, but acc)
    if iter ==1 and cap_acc: print("Using Cap")
    cap = 4

    if gain_choice != 'None' and gain_choice is not None:
        if iter==1: print("Using PID gain")
        gain= calc_gain(p_w,i_w,d_w,errs, x_estim, gain_choice, u, x)
    else:
        #if iter == 1: print('None PID --> gain set to 1')
        gain = 0

    if control == 'none': return new_u
    if scenario == 'sit still':
        new_u = -1*x_estim
    elif scenario == 'sit still 2':
        new_u = [0, -x_estim[1]-x_estim[0]] #-2*x[1]-x[0]]
        #new_u = [0,gain] #*(x_estim[1]+x_estim[0])] #control vel based on estim position
    elif scenario == 'sint':
        new_u = [0, gain] #TODO: if robo 'knows' it should follow sint, easy. But say it doesn't...

    elif scen_d[0] == '2D':
        new_u = [0, targets[iter,0] - x_estim[1] - x_estim[0], 0, targets[iter,1] - x_estim[3] - x_estim[2]]

    elif scen_d[0] == '2D3':
        #new_u = [0, 0, (1/3)*targets[iter+1,0]-(1/3)*x_estim[0]-(2/3)*x_estim[1], 0,0, (1/3)*targets[iter+1,1] - (1/3)*x_estim[3] - (2/3)*x_estim[4]]
        new_u = [0, 0, targets[iter+1,0]-x_estim[0]-2*x_estim[1] - x_estim[2] + gain, 0,0, targets[iter+1,1] - x_estim[3] - 2*x_estim[4] - x_estim[5] + gain]
        #print("x_acc_chng, gain = " + str(targets[iter+1,0]-x_estim[0]-2*x_estim[1] - x_estim[2]) + ', ' + str(gain))

        #new_u = [0, 0, (targets[iter + 1, 0] - x_estim[0] - 2 * x_estim[1] - x_estim[2])*gain, 0, 0,(targets[iter + 1, 1] - x_estim[3] - 2 * x_estim[4] - x_estim[5])*gain]

        if cap_acc:
            x_acc, y_acc = new_u[2],new_u[5]
            if x_acc > 0:  x_acc = min(cap,x_acc)
            else: x_acc = max(-1*cap,x_acc)

            if y_acc > 0:  y_acc = min(cap,y_acc)
            else: y_acc = max(-1*cap,y_acc)
            new_u[2], new_u[5] = x_acc, y_acc


    else: #if scenario == 'drift' or scenario == 'rude drift':
        #target = iter+1
        new_u = [0, targets[iter]-x_estim[1]-x_estim[0]]

        new_u = [0, targets[iter] - x_estim[1] - x_estim[0]] #-gain]

    #else: assert(False)

    return new_u


def calc_gain(curr_w, integral_w, deriv_w, errs, X, gain_choice, u, x):
    #discretized PID control, assumes dt = 1
    # TODO: apparently suppoed to use u += gain...but this works horribly
    # TODO: really -= gain, since curr formula is -ve
    # TODO: will need to change deriv term for changing target, use derivative-on-measurement PID form
    # integral term tends to grow over time, ie isn't normzd by lng, but this isn't intuitive at all

    # TODO: err is currently x[0] + x[1], but problems seem to be def'd with err = X[i] for gain[i]

    # TODO: also curr just 1D, should be spc to direction

    verbose = False

    gain_p = errs[-1]*curr_w
    if errs[-1] < 0: gain_p = 1/gain_p

    if len(errs)>1:
        gain_i = -1*np.sum(errs)*integral_w/(len(errs)-1) #all but most recent err term
        gain_d = -1*(errs[-1] - errs[-2]) * deriv_w
    else: return gain_p

    if len(errs) > 2 and False:
        # a lot of these attemps perform similarly, none perform better
        # likely something fundamental missing, but could try higher order terms
        #gain_d2 = -1*(errs[-3]-2*errs[-2]+errs[-1])*deriv_w
        gain_d2 = -1*(errs[-1] - errs[-3])*deriv_w
        if verbose:
            print("\ngain_d, gain_d2,  X[1], t")
            print(gain_d, gain_d2, X[1], len(errs))
        #gain_d2 = (gain_d + 2*gain_d2)/3
        #gain_d2 = (gain_d+gain_d2)/2
        #gain_d2 = -1*(gain_d*gain_d2*X[1])/abs(X[1]) #TODO: seems to control in right dir, but wrong magnitude
        if gain_choice=='PD2': return gain_p + gain_d2
        elif gain_choice=='PID2': return gain_p + gain_i + gain_d2


    if gain_choice=='P': gain = gain_p
    elif gain_choice=='PI': gain = gain_p + gain_i
    elif gain_choice=='PD' or gain_choice == 'PD2': gain= gain_p + gain_d
    elif gain_choice=='PID' or gain_choice == 'PID2': gain = gain_p + gain_i + gain_d



    else: assert(False)
    #print("\nerr = " + str(errs[-1]))
    #print("gain = " + str(gain)) #, gain_p, gain_i, gain_d: ")
    #print(gain, gain_p, gain_i, gain_d)
    return gain
