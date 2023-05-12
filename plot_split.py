from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.stats
import torch.nn.functional as F
import torch
import warnings


def moving_avg(x,n):
    mv =  np.convolve(x,np.ones(n)/n,mode='valid')
    return np.concatenate(([np.NaN for k in range(n-1)],mv))


def data_from_logfile(logfile, utility, gamma=1.):
    n_objectives = len(logfile['train/reward/'])
    r = np.stack([logfile[f'train/reward/{i}'][:,1] for i in range(n_objectives)], axis=1)
    steps = logfile['train/reward/0'][:,0]
    if n_objectives > 1:
        u = utility(r)
    else:
        NotImplementedError('not multiobjective')
    return np.concatenate((steps.reshape(-1, 1), u), axis=1)

def data_from_logs(p, utility, gamma=1.):
    log_paths = [i for i in p.rglob('*.h5')]

    datas = []
    for lp in log_paths:
        logfile = h5py.File(lp, 'r')
        data = data_from_logfile(logfile, utility, gamma)
        datas.append(data)

    all_steps = sorted(np.unique(np.concatenate([d[:,0] for d in datas])))
    all_utilities = [np.interp(all_steps, d[:,0], moving_avg(d[:,1], 5)) for d in datas]
    data = np.stack((all_steps, np.mean(all_utilities, axis=0), np.std(all_utilities, axis=0)), axis=1)

    return data

def utility_mul(values):
    u = values[:,0]*values[:,1]
    return u.reshape(-1,1)

if __name__ == '__main__':
    import sys
    import pandas as pd

    dirs = [Path(p) for p in sys.argv[1:]]
    labels =['moreinforce', 'mocac', 'moac']

    plt.figure()
    
    for i, dir_ in enumerate(dirs):
        print(labels[i])
        # do not plot every episode
        data = data_from_logs(dir_, utility_mul)
        steps, mean_, std_ = data[::10, 0], data[::10, 1], data[::10, 2]

        plt.plot(steps, mean_, label=labels[i])
        plt.fill_between(steps, 
                    mean_ - std_,
                    mean_ + std_,
                    alpha=0.2)

        all_runs = {}
        all_runs['steps'] = steps
        all_runs['values'] = mean_
        all_runs['std'] = std_
        df = pd.DataFrame(all_runs).fillna(0.).to_csv(f'/tmp/split_{labels[i]}.csv', index=False, header=True)
        
    plt.legend()
    plt.show()

