import os, numpy as np, math
from matplotlib import pyplot as plt


def key(run_name):

    dirname = os.getcwd()
    plot_dir = dirname + '/output/' + str(run_name) + '_plots/'
    plt.title('TYPE OF PLOT, PARAMETER (control type, PID type, noise)')
    plt.savefig(plot_dir + '/A_KEY.png')
    plt.clf()

def state_estim(run_name, xs, xestims, targets, control, scenario, gain_choice, use_noise):
    xs, xestims = np.array(xs), np.array(xestims)
    if use_noise: noise = 'noisy'
    else: noise = "noNoise"

    titles = gen_titles(scenario)

    dirname = os.getcwd()
    plot_dir = dirname + '/output/' + str(run_name) + '_plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for i in range(len(xs[0])):
        if i==0 or i==3 or i==2 or i==5: #TODO: temp only plot position, add acc now
            param = xs[:,i]
            param_estim = xestims[:,i]
            n = len(param)
            ymax = max(max(param),max(param_estim))
            ymin = min(min(param),min(param_estim))
            yabs = max(abs(ymax), abs(ymin))
            MSE = np.linalg.norm(param-param_estim)/n
            MSD = np.linalg.norm(param)/n
            ticks = [j for j in range(len(param))]
            plt.text(0,ymax,"Avg MSE of estimation = " + str(MSE), color='red')
            #plt.text(0,ymax-yabs/8,"Mean dist from origin = " + str(MSD), color='red')
            plt.plot(ticks,param, color='blue', label='actual', alpha=.6)
            plt.plot(ticks,param_estim, color='green', label='estimated', alpha=.8, linestyle='-.')
            if i==0 or i==3: #ie position
                scen = scenario.split(' ')
                if scen[0] == '2D' or scen[0] == '2D3':
                    if i ==0: target_vec = targets[:,0]
                    else: target_vec = targets[:,1]
                else: target_vec = targets

                plt.plot(ticks, target_vec[:-1], color='purple', label='targets', alpha=.7, linestyle=':')
                #var_targets = np.linalg.norm(targets)
                var_targets = 0
                for k in range(len(target_vec)-1):
                    var_targets += np.linalg.norm(target_vec[k]-target_vec[k+1])
                var_targets /= len(target_vec)
                plt.text(0, ymax - yabs / 8, "Avg variance of target positions = " + str(var_targets), color='red')

            plt.title('Estimated vs Actual ' + str(titles[i]) + " (" + str(control) + ", " + str(gain_choice) + ',' + str(noise) + ")")
            plt.legend(loc = 'lower left')
            plt.savefig(plot_dir + '/control_' + str(control) + '_gain_' + str(gain_choice) + '_noise_' + str(noise) + '_param_' + titles[i] + '.png')
            plt.clf()



def control_err(run_name, us,  estim_errs, actual_errs, targets, control_name, scenario, gain_choice, use_noise):
    us, estim_errs, actual_errs = np.array(us), np.array(estim_errs), np.array(actual_errs)
    if use_noise: noise = 'noisy'
    else: noise = "noNoise"

    titles = gen_titles(scenario)

    dirname = os.getcwd()
    plot_dir = dirname + '/output/' + str(run_name) + '_plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for i in range(len(us[0])):
        if i==2 or i==5: #TODO: temp to only use acc control plots
            control = us[:,i]
            n = len(control)
            ymax = max(max(control),max(estim_errs), max(actual_errs))
            ymin = min(min(control),min(estim_errs), min(actual_errs))
            yabs = max(abs(ymax), abs(ymin))
            if yabs == 0:
                ymax = ymin = yabs =.04
            ticks = [j for j in range(len(control))]
            MSE_actual = np.sum(actual_errs)/len(actual_errs)
            MSE_estim = np.sum(estim_errs)/len(estim_errs)

            plt.plot(ticks, actual_errs, color='red', label='actual error', alpha=.7)
            plt.plot(ticks,control, color='#33cccc', label='control choice', alpha=1, linestyle=':')
            plt.plot(ticks,estim_errs, color='purple', label='perceived error', alpha=.6, linestyle='-.')
            scen = scenario.split(' ')
            if scen[0] == '2D' or scen[0]=='2D3':
                plt.text(0,ymax,"Actual avg MSE of positions (x and y) = " + str(MSE_actual), color='red')
                plt.text(0, ymax - yabs / 8, "Estimated avg MSE of position (x and y) = " + str(MSE_estim), color='purple')

            else:
                plt.text(0,ymax,"Actual avg MSE of position = " + str(MSE_actual), color='red')
                plt.text(0, ymax - yabs / 8, "Estimated avg MSE of position = " + str(MSE_estim), color='purple')

            plt.title('Control Parameter and Error ' + str(titles[i]) + " (" + str(control_name) + ", " + str(gain_choice) + ',' + str(noise) + ")")
            plt.legend(loc = 'lower left')
            plt.savefig(plot_dir + '/control_' + str(control_name) + '_gain_' + str(gain_choice) + '_noise_' + str(noise) + '_param_' + titles[i] + '_controlAndError.png')
            plt.clf()

def Kalman_weight(run_name, Ks, control_name, scenario, gain_choice, use_noise):
    dirname = os.getcwd()
    plot_dir = dirname + '/output/' + str(run_name) + '_plots/'
    Ks = np.array(Ks)
    label = gen_label(control_name, scenario, gain_choice, use_noise)
    titles = gen_titles(scenario)
    ticks = [j for j in range(len(Ks))]
    colors = ['blue','red','green','purple']
    linestyles = ['-.', ':','--','-.', ':','--']
    n = len(Ks[0])
    for i in range(n):
        for j in range(n):
            if i==0:
                a_label = titles[i] + ' --> ' + titles[j]
                plt.plot(ticks, Ks[:,i,j], color = colors[(i*n+j)%4], label = a_label, alpha=.5, linestyle = linestyles[(i*n+j)%6])
    plt.title("Kalman Weight on Measurement Term for each State Variable")
    plt.legend(title = 'Weight of measurement --> on estimate')
    plt.savefig(plot_dir + str(label) + '_Kalman.png')
    plt.clf()



def gen_label(control_name, scenario, gain_choice, use_noise):
    if use_noise: noise = 'noisy'
    else: noise = "noNoise"

    label = '/control_' + str(control_name) + '_gain_' + str(gain_choice) + '_noise_' + str(noise)
    return label

def gen_titles(scenario):

    scen_d = scenario.split(' ')

    if scenario == 'sit still': titles = ['x','y']
    #elif scenario == 'drift' or scenario == 'sint' or scenario=='meanAF': titles = ['position','velocity']

    elif scen_d[0] == '2D':
        titles = ['x position', 'x velocity', 'y position', 'y velocity']

    elif scen_d[0] == '2D3':
        titles = ['x position', 'x velocity', 'x acceleration', 'y position', 'y velocity', 'y acceleration']
    else:  titles = ['position','velocity']
    #else: assert(False)
    return titles
