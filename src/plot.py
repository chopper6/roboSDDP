import os, numpy as np
from matplotlib import pyplot as plt

def state_estim(xs, xestims, control, scenario, gain_choice, use_noise):
    xs, xestims = np.array(xs), np.array(xestims)
    if use_noise: noise = 'noisy'
    else: noise = "noNoise"

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
        plt.plot(ticks,param, color='blue', label='actual', alpha=.8)
        plt.plot(ticks,param_estim, color='green', label='estimated', alpha=.8)
        plt.title('Estimated vs Actual ' + str(titles[i]) + " (" + str(control) + ", " + str(gain_choice) + ',' + str(noise) + ")")
        plt.legend(loc = 'lower left')
        plt.savefig(plot_dir + '/control_' + str(control) + '_gain_' + str(gain_choice) + '_noise_' + str(noise) + '_param_' + titles[i] + '.png')
        plt.clf()



def control_err(us, errs, control_name, scenario, gain_choice, use_noise):
    us, errs = np.array(us), np.array(errs)
    if use_noise: noise = 'noisy'
    else: noise = "noNoise"

    if scenario == 'sit still': titles = ['x','y']
    elif scenario == 'drift': titles = ['position','velocity']
    else: assert(False)

    dirname = os.getcwd()
    plot_dir = dirname + '/output/plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for i in range(len(us[0])):
        control = us[:,i]
        error = errs
        n = len(control)
        ymax = max(max(control),max(error))
        ymin = min(min(control),min(error))
        yabs = max(abs(ymax), abs(ymin))
        ticks = [j for j in range(len(control))]
        plt.plot(ticks,control, color='purple', label='control', alpha=.8)
        plt.plot(ticks,error, color='red', label='error', alpha=.8)
        plt.title('Control Parameter and Error ' + str(titles[i]) + " (" + str(control_name) + ", " + str(gain_choice) + ',' + str(noise) + ")")
        plt.legend(loc = 'lower left')
        plt.savefig(plot_dir + '/control_' + str(control_name) + '_gain_' + str(gain_choice) + '_noise_' + str(noise) + '_param_' + titles[i] + '_controlAndError.png')
        plt.clf()
