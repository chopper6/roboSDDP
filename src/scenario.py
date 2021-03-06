#TODO: eventually the scenario should be the path to follow
import numpy as np, math, random as rd



def generate(scenario, use_noise, iters):
    A,B,C,x0, mean, covar, noise_cov_mv, noise_cov_ms, targets = 0,0,0,0,0,0,0,0,0
    if scenario == 'sit still':
        A,B,C = np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]]),np.array([[1,0],[0,1]])
        x0, mean, covar = np.array([0,0]),np.array([0,0]),np.array([[1,0],[0,1]])
        noise_cov_mv, noise_cov_ms = np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])
        targets = np.array([0 for i in range(iters+1)])

    elif scenario == 'sit still 2':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([1, 1]), np.array([1, 1]), np.array([[1, 0], [0, 1]])
        noise_cov_mv, noise_cov_ms = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        targets = np.array([0 for i in range(iters+1)])

    elif scenario == 'sint':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([1, 1]), np.array([1, 1]), np.array([[1, 0], [0, 1]])
        noise_cov_mv, noise_cov_ms = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        assert(False)         #TODO: def targets

    elif scenario == 'drift':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([1, 1]), np.array([1, 1]), np.array([[1, 0], [0, 1]])
        targets = np.array([i+1 for i in range(iters+1)])
        noise_cov_mv, noise_cov_ms = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])

    elif scenario == 'rude drift':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([1, 1]), np.array([1, 1]), np.array([[1, 0], [0, 1]])
        targets = np.array([i+1 for i in range(iters+1)])
        noise_cov_mv, noise_cov_ms = np.array([[1, -1], [1, 1]]), np.array([[1, 1], [-1, 1]])

    elif scenario == 'uniform rd path':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([1, 1]), np.array([1, 1]), np.array([[1, 0], [0, 1]])
        targets = np.array([rd.uniform(-1,1) for i in range(iters+1)])
        noise_cov_mv, noise_cov_ms = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])

    elif scenario == 'rude uniform rd path':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([1, 1]), np.array([1, 1]), np.array([[1, 0], [0, 1]])
        targets = np.array([rd.uniform(-1,1) for i in range(iters+1)])
        noise_cov_mv, noise_cov_ms = np.array([[1, -1], [1, 1]]), np.array([[1, 1], [-2, 1]])

    elif scenario == 'uniform 10 rd path':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([1, 1]), np.array([1, 1]), np.array([[1, 0], [0, 1]])
        targets = np.array([rd.uniform(-1,1)*10 for i in range(iters+1)])
        noise_cov_mv, noise_cov_ms = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])

    elif scenario == 'gauss 10 rd path':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([1, 1]), np.array([1, 1]), np.array([[1, 0], [0, 1]])
        targets = np.array([rd.normalvariate(0,1)*10 for i in range(iters+1)])
        noise_cov_mv, noise_cov_ms = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])

    elif scenario == 'rude uniform rd path with drift':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([1, 1]), np.array([1, 1]), np.array([[1, 0], [0, 1]])
        targets = []
        targets.append(rd.uniform(-2,2))
        for i in range(1,iters+1):
            targets.append(targets[-1]+rd.uniform(-2,2))
        noise_cov_mv, noise_cov_ms = np.array([[-2, -1], [1, 1]]), np.array([[1, 2], [-2, 1]])

    elif scenario == 'uniform rd path with drift 2':
        A, B, C = np.array([[1, 1], [0, 1]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]])
        x0, mean, covar = np.array([1, 1]), np.array([1, 1]), np.array([[1, 0], [0, 1]])
        targets = []
        targets.append(rd.uniform(-1,1))
        for i in range(1,iters+1):
            targets.append(targets[-1]+rd.uniform(-1,2))
        noise_cov_mv, noise_cov_ms = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])

    elif scenario == '2D drift':
        A, B, C = np.array([[1,1,0,0], [0,1,0,0],[0,0,1,1],[0,0,0,1]]), np.array([[0,0,0,0], [0,1,0,0],[0,0,0,0],[0,0,0,1]]), np.eye(4)
        x0, mean, covar = np.array([1,1,1,1]), np.array([1,1,1,1]), np.eye(4)
        targets = np.array([[i+1,i+1] for i in range(iters+1)])
        noise_cov_mv, noise_cov_ms = np.eye(4), np.eye(4)


    elif scenario == '2D rd path with drift':
        A, B, C = np.array([[1,1,0,0], [0,1,0,0],[0,0,1,1],[0,0,0,1]]), np.array([[0,0,0,0], [0,1,0,0],[0,0,0,0],[0,0,0,1]]), np.eye(4)
        x0, mean, covar = np.array([1,1,1,1]), np.array([1,1,1,1]), np.eye(4)
        targets = []
        targets.append([rd.uniform(-1,1),rd.uniform(-1,1)])
        for i in range(1,iters+1):
            x_target = targets[-1][0]+rd.uniform(-1,2)
            y_target = targets[-1][1]+rd.uniform(-1,2)
            targets.append([x_target,y_target])
        targets = np.array(targets)
        noise_cov_mv, noise_cov_ms = np.eye(4), np.eye(4)



    elif scenario == '2D rude rd path with drift':
        A, B, C = np.array([[1,1,0,0], [0,1,0,0],[0,0,1,1],[0,0,0,1]]), np.array([[0,0,0,0], [0,1,0,0],[0,0,0,0],[0,0,0,1]]), np.eye(4)
        x0, mean, covar = np.array([1,1,1,1]), np.array([1,1,1,1]), np.eye(4)
        targets = []
        targets.append([rd.uniform(-1,1),rd.uniform(-1,1)])
        for i in range(1,iters+1):
            x_target = targets[-1][0]+rd.uniform(-1,2)
            y_target = targets[-1][1]+rd.uniform(-1,2)
            targets.append([x_target,y_target])
        targets = np.array(targets)
        noise_cov_mv, noise_cov_ms = np.array([[1,0,.5,0],[0,1,0,.5],[.5,0,1,0],[.5,0,1,0]]),np.array([[1,0,.5,0],[0,1,0,.5],[.5,0,1,0],[.5,0,1,0]])


    elif scenario == '2D3 drift':
        A, B, C = np.array([[1,1,0,0,0,0], [0,1,1,0,0,0],[0,0,1,0,0,0],[0,0,0,1,1,0],[0,0,0,0,1,1],[0,0,0,0,0,1]])\
            , np.array([[0 for i in range(6)], [0 for i in range(6)],[0,0,1,0,0,0],[0 for i in range(6)],[0 for i in range(6)],[0,0,0,0,0,1]])\
            , np.eye(6)
        covar = np.eye(6)
        # static:
        #targets = np.array([[0,0] for i in range(iters+2)])
        #x0, mean = [1, 0,0,1,0,0],[1, 0,0,1,0,0]
        targets = np.array([[i+1,i+1] for i in range(iters+2)])
        x0, mean = [targets[0,0], 0,0,targets[0,1],0,0],[targets[0,0], 0,0,targets[0,1],0,0]

        noise_cov_mv, noise_cov_ms = 1*np.eye(6), 1*np.eye(6)

    elif scenario == '2D3 drift correlated axis':
        A, B, C = np.array([[1,1,0,0,0,0], [0,1,1,0,0,0],[0,0,1,0,0,0],[0,0,0,1,1,0],[0,0,0,0,1,1],[0,0,0,0,0,1]])\
            , np.array([[0 for i in range(6)], [0 for i in range(6)],[0,0,1,0,0,0],[0 for i in range(6)],[0 for i in range(6)],[0,0,0,0,0,1]])\
            , np.eye(6)
        covar = np.eye(6)
        # static:
        #targets = np.array([[0,0] for i in range(iters+2)])
        #x0, mean = [1, 0,0,1,0,0],[1, 0,0,1,0,0]
        targets = np.array([[i+1,i+1] for i in range(iters+2)])
        x0, mean = [targets[0,0], 0,0,targets[0,1],0,0],[targets[0,0], 0,0,targets[0,1],0,0]

        noise_cov_mv, noise_cov_ms = .2*np.array([[1,0,0,1,0,0], [0,1,0,0,1,0],[0,0,1,0,0,1],[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1]]),\
                                     .8*np.array([[1,0,0,1,0,0], [0,1,0,0,1,0],[0,0,1,0,0,1],[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1]])

        noise_cov_mv, noise_cov_ms = np.array(
            [[1, 0, 0, .5, 0, 0], [0, 1, 0, 0,.5, 0], [0, 0, 1, 0, 0, 0], [.5, 0, 0, 1, 0, 0], [0, .5, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]]), \
                                     np.array([[1, 0, 0, .5, 0, 0], [0, 1, 0, 0, .5, 0], [0, 0, 1, 0, 0, 0],
                                                    [.5, 0, 0, 1, 0, 0], [0, .5, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

        #noise_cov_mv = noise_cov_mv.astype(float)*.1

    elif scenario == '2D3 drift ms corr':
        A, B, C = np.array([[1,1,0,0,0,0], [0,1,1,0,0,0],[0,0,1,0,0,0],[0,0,0,1,1,0],[0,0,0,0,1,1],[0,0,0,0,0,1]])\
            , np.array([[0 for i in range(6)], [0 for i in range(6)],[0,0,1,0,0,0],[0 for i in range(6)],[0 for i in range(6)],[0,0,0,0,0,1]])\
            , np.eye(6)
        covar = np.eye(6)
        # static:
        #targets = np.array([[0,0] for i in range(iters+2)])
        #x0, mean = [1, 0,0,1,0,0],[1, 0,0,1,0,0]
        targets = np.array([[i+1,i+1] for i in range(iters+2)])
        x0, mean = [targets[0,0], 0,0,targets[0,1],0,0],[targets[0,0], 0,0,targets[0,1],0,0]

        noise_cov_ms, noise_cov_mv = np.array(
            [[1, 0, 0, .5, 0, 0], [0, 1, 0, 0,.5, 0], [0, 0, 1, 0, 0, 0], [.5, 0, 0, 1, 0, 0], [0, .5, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]]), np.eye(6)
        #noise_cov_mv = noise_cov_mv.astype(float)*.1



    elif scenario == '2D3 drift acc':
        A, B, C = np.array([[1,1,0,0,0,0], [0,1,1,0,0,0],[0,0,1,0,0,0],[0,0,0,1,1,0],[0,0,0,0,1,1],[0,0,0,0,0,1]])\
            , np.array([[0 for i in range(6)], [0 for i in range(6)],[0,0,1,0,0,0],[0 for i in range(6)],[0 for i in range(6)],[0,0,0,0,0,1]])\
            , np.eye(6)
        covar = np.eye(6)
        # static:
        #targets = np.array([[0,0] for i in range(iters+2)])
        #x0, mean = [1, 0,0,1,0,0],[1, 0,0,1,0,0]
        targets = np.array([[math.pow(i+rd.uniform(-2,2),2)+1,math.pow(i+rd.uniform(-2,2),2)+1] for i in range(iters+2)])
        x0, mean = [targets[0,0], 0,0,targets[0,1],0,0],[targets[0,0], 0,0,targets[0,1],0,0]
        print("Scenario 141: using funky scenario, but why is err != 0 with noise?!")
        noise_cov_mv, noise_cov_ms = .2*np.eye(6), .8*np.eye(6)
        #noise_cov_mv = noise_cov_mv.astype(float)*.1




    else: assert(False)

    #if not use_noise:
    #    noise_cov_mv, noise_cov_ms = np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])

    return A, B, C, x0, mean, covar, noise_cov_mv, noise_cov_ms, targets

