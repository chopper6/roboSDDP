import numpy as np


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



def update_u(scenario, x_estim, control, gain_choice,errs):
    u=[0,0]
    p_w, i_w, d_w = .2,.2,1
    gain= calc_gain(p_w,i_w,d_w,errs,gain_choice)
    if control == 'none': return u
    if scenario == 'sit still':
        u = -1*x_estim
    elif scenario == 'drift':
        u = [0,gain] #*(x_estim[1]+x_estim[0])] #control vel based on estim position
    else: assert(False)
    return u


def calc_gain(curr_w, integral_w, deriv_w, errs, gain_choice):
    #discretized PID control, assumes dt = 1
    gain_p = -1*errs[-1]*curr_w
    if len(errs)>1:
        gain_i = -1*sum(errs[:-1])*integral_w/(len(errs)-1) #all but most recent err term
        gain_d = -1*(errs[-1] - errs[-2]) * deriv_w
    else: return gain_p

    if gain_choice=='P': gain = gain_p
    elif gain_choice=='PI': gain = gain_p + gain_i
    elif gain_choice=='PD': gain= gain_p + gain_d
    elif gain_choice=='PID': gain = gain_p + gain_i + gain_d
    else: assert(False)
    #print("\nerr = " + str(errs[-1]))
    #print("gain = " + str(gain)) #, gain_p, gain_i, gain_d: ")
    #print(gain, gain_p, gain_i, gain_d)
    return gain
