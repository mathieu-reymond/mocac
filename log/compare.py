import h5py
from pathlib import Path
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np



def compute_utility(logfile, utility_function):
    n_objectives = len(logfile['train/reward'])
    rewards = np.stack([logfile[f'train/reward/{o}'][:,1] for o in range(n_objectives)], axis=1)
    item = utility_function(rewards)
    item = np.stack([logfile['train/reward/0'][:,0], item], axis=1)
    return item


def interpolate_runs(runs):
    if not runs:
        return (np.zeros(1), np.zeros(1))
    all_steps = np.array(sorted(np.unique(np.concatenate([r[:,0] for r in runs]))))
    all_values = np.stack([np.interp(all_steps, r[:,0], r[:,1]) for r in runs], axis=0)
    # all_steps = all_steps[all_steps < 100000]
    # all_values = all_values[:,:len(all_steps)]
    return all_steps, all_values


def moving_avg(x, moving_factor):
    avg = np.zeros_like(x)
    for i in range(1, len(x)):
        avg[i] = avg[i-1]*moving_factor + (1.-moving_factor)*x[i]
    return avg
    # mv =  np.convolve(x,np.ones(n)/n,mode='valid')
    # return np.concatenate(([np.NaN for k in range(n-1)],mv))


def moving_avgs(x, moving_factor):
    mv = np.stack([moving_avg(i, moving_factor) for i in x], axis=0)
    return mv


def make_plots(*logdirs, moving_factor=0.99, utility_function=None):
    assert utility_function is not None, 'should provide a utility function'
    all_runs = {}
    
    # go over all experiments to compare against
    for d_i, logdir in enumerate(logdirs):
        for ui in range(1, 11):
            if f'utility_fishwood_{ui}' in logdir:
                uf = globals()[f'utility_fishwood_{ui}']
                opt = optimal_fishwood(uf)
                utility_function = lambda v, opt=opt, uf=uf: uf(v)/opt
        # find all runs of that experiment
        for i, path in enumerate(Path(logdir).rglob('log.h5')):
            print('making plots for ' + str(path))
            plotdir = path.parents[0] / 'plots'
            plotdir.mkdir(parents=True, exist_ok=True)
            logfile = h5py.File(path, 'r')
            # we are going to get all scalars of each run, to interpolate
            for p, item, type_ in h5py_dataset_iterator(logfile):
                if type_ == 'scalar':
                    if not item.name in all_runs:
                        all_runs[item.name] = [[] for _ in range(len(logdirs))]
                    all_runs[item.name][d_i].append(item.value)
            
            # compute utility based on the vectorial rewards
            if not 'utility' in all_runs:
                all_runs['utility'] = [[] for _ in range(len(logdirs))]
            if len(logfile['train/reward']) > 1:
                value = compute_utility(logfile, utility_function)
            else:
                value = logfile['train/reward/0']/np.array([[1, opt]])
            if (not value is None): # and value[-1, 1] < 1000:
                all_runs['utility'][d_i].append(value)

    for k, v in all_runs.items():
        for i in range(len(v)):
            steps, values = interpolate_runs(v[i])
            values = moving_avgs(values, moving_factor)
            v[i] = (steps, values)
        plot_scalars(k, v)


def plot_scalars(name, runs):
    plt.figure()
    #plt.title(name.replace('/', '_'))
    LABELS = [r'$r_0^2+r_1^2$',r'$r_0^2+r_1$',r'$\max(\lfloor r_0/4\rfloor ,\lfloor r_1/2\rfloor$',r'$r_0 r_1$',r'$(r_0 > 2) r_1$',r'$\log(1+r_0)+\log(1+r_1)$',]
    for i, (steps, values) in enumerate(runs):
        if i == 4: continue 
        avg, std = np.mean(values, axis=0), np.std(values, axis=0)
        plt.plot(steps, avg, label=f'run {i} ({values.shape[0]})')
        # plt.plot(steps, avg, label=LABELS[i])
        plt.fill_between(steps, avg-std, avg+std, alpha=0.2)
    plt.legend()
    plt.xlabel('timesteps')
    plt.ylabel('normalized utility')
    plt.show()
    plt.close()


def h5py_dataset_iterator(g, prefix=''):
    for key in g.keys():
        print(key)
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        yield (path, item, None)
        # if 'type' in item.attrs:
        #     yield (path, item, item.attrs['type'])
        # else:
        #     yield from h5py_dataset_iterator(item, path)
        # if isinstance(item, h5py.Dataset): # test for dataset
        #     yield (path, item)
        # elif isinstance(item, h5py.Group): # test for group (go down)
        #     yield from h5py_dataset_iterator(item, path)


def optimal_fishwood(utility_function):
    wood_prob, fish_prob = 0.65, 0.25
    n_runs = 1000
    max_u = -np.inf
    for i in range(1,14):
        wood = np.random.binomial(i, wood_prob, n_runs)
        fish = np.random.binomial(13-i, fish_prob, n_runs)
        r = np.stack((fish, wood), 1)
        u = utility_function(r)
        u = np.mean(u)
        if u > max_u:
            max_u = u
    return max_u


if __name__ == '__main__':
    import sys
    
    # utility_index = sys.argv[1]
    
    def utility_fishwood_1(values):
        values = (values[:,0] ** 2) + (values[:,1] ** 2)
        return values

    def utility_fishwood_2(values):
        values = ((values[:,0]) + (values[:,1])) ** 2
        return values

    def utility_fishwood_3(values):
        values = (values[:,0] ** 2) + (values[:,1])
        return values

    def utility_fishwood_4(values):
        values = np.exp((values[:,0]) + (values[:,1]))
        return values

    def utility_fishwood_5(values):
        values = np.maximum((values[:,0]//4),(values[:,1] //2))
        return values

    def utility_fishwood_6(values):
        values = (1 + np.exp(values[:,0])) / (1 + np.exp(values[:,1]))
        return values

    def utility_fishwood_7(values):
        values = (np.log(1+values[:,0])) / (np.log(1+values[:,1]))
        return values

    def utility_fishwood_8(values):
        values = values[:,0] * values[:,1]
        return values

    def utility_fishwood_9(values):
        values = (values[:,0] > 2) * values[:,1]
        return values

    def utility_fishwood_10(values):
        values = np.log(1. + values[:,0]) + np.log(1. + values[:,1])
        return values

    #utility_function = globals()[f'utility_fishwood_{utility_index}']
    #opt = optimal_fishwood(utility_function)
    #print('==========optimal utility============')
    #print(opt)
    #print('='*50)
    
    #normalized_utility = lambda v, opt=opt, uf=utility_function: uf(v)/opt
    
    make_plots(*sys.argv[1:], utility_function=utility_fishwood_1)
