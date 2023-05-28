import os 
import pickle as pkl 
from tqdm import tqdm  
import numpy as np 
import matplotlib.pyplot as plt 
from time import time

from utils import * 
from ConfSeqs import * 
from changepoint import CPD_strategy2


def check_ARL(N, Alpha, num_trials, seed=None, save_data=False, save_fig=False):
    if seed is None: 
        seed = int(time()%10000)
    experiment_name = 'BoundedMean'
    #### Source 
    p0 = 0.5 
    p1=0.5
    Source = BoundedSource
    Source_kwargs = {'p0':p0, 'p1':p1, 'nc':None}
    #### FWD CS 
    fwdCS_func = get_Bounded_EB_CS
    fwdCS_kwargs = {}

    ARL = np.zeros(Alpha.shape)
    seed_ = seed
    for _ in tqdm(range(num_trials)):
        for i, alpha in enumerate(Alpha): 
            seed_ = get_new_seed(seed_)
            np.random.seed(seed_)
            XX = Source(N=N, **Source_kwargs)
            Result =  CPD_strategy2(XX, tmax=N, 
                                fwdCS_func=fwdCS_func,
                                fwdCS_kwargs=fwdCS_kwargs,
                                backCS_func=fwdCS_func,
                                backCS_kwargs=fwdCS_kwargs,
                                check_disjoint=check_disjoint_univariate_two_CS2,
                                alpha=alpha,
                                two_sample_data=False, 
                                verbose=False)
            ARL[i] += Result['stopping_time']
    ARL /= num_trials  

    base_dir = './data/BoundedMean/'
    if save_data: 
        data_dict = {
            'Alpha':Alpha, 'ARL':ARL, 'N':N, 'Source':Source, 
            'Source_kwargs':Source_kwargs, 'fwdCS_func':fwdCS_func, 
            'fwdCS_kwargs':fwdCS_kwargs
        }
        filename = base_dir + 'ARL_vs_Alpha'+str(seed)+'.pkl' 
        with open(filename, 'wb') as f:
            pkl.dump(data_dict, f) 

    plt.figure()
    plt.plot(Alpha, ARL)
    plt.title(r'ARL vs $\alpha$', fontsize=15)
    plt.xlabel(r'$\alpha$', fontsize=13)
    plt.ylabel('Average Run Length')
    if save_fig:
        figname = base_dir + 'ARL_vs_Alpha' + str(seed)
        tpl.save(figname + ".tex", axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.savefig(figname+'.png', dpi=450)
    else:
        plt.show()


def check_Delay(N, nc, Delta, num_trials, seed=None, alpha=0.01, 
                    save_data=False, save_fig=False, p0=0.2, 
                    xmax=12, linear_fit=True):
    if seed is None: 
        seed = int(time()%10000)
    Source = BoundedSource
    #### FWD CS 
    fwdCS_func = get_Bounded_EB_CS
    fwdCS_kwargs = {}

    Delay = np.zeros(Delta.shape)
    seed_ = seed
    for _ in tqdm(range(num_trials)):
        for i, delta in enumerate(Delta): 
            p1 = delta 
            Source_kwargs = {'p0':p0, 'p1':p1, 'nc':nc}

            seed_ = get_new_seed(seed_)
            np.random.seed(seed_)
            XX = Source(N=N, **Source_kwargs)
            Result =  CPD_strategy2(XX, tmax=N, 
                                fwdCS_func=fwdCS_func,
                                fwdCS_kwargs=fwdCS_kwargs,
                                backCS_func=fwdCS_func,
                                backCS_kwargs=fwdCS_kwargs,
                                check_disjoint=check_disjoint_univariate_two_CS2,
                                alpha=alpha,
                                two_sample_data=False, 
                                verbose=False)
            Delay[i] += max(0, Result['stopping_time']-nc)
    Delay /= num_trials  

    base_dir = './data/BoundedMean/'
    if save_data: 
        data_dict = {
            'Delta':Delta, 'Delay':Delay, 'N':N, 'nc':nc, 'Source':Source, 
            'Source_kwargs':Source_kwargs, 'fwdCS_func':fwdCS_func, 
            'fwdCS_kwargs':fwdCS_kwargs, 'alpha':alpha
        }
        filename = base_dir + 'Delay_vs_Delta'+str(seed)+'.pkl' 
        with open(filename, 'wb') as f:
            pkl.dump(data_dict, f) 

    # normalize the x-axis 
    assert (xmax>0)
    Delta = Delta /(np.min(Delta)*sqrt(xmax))
    if linear_fit: 
        a_, b_ = np.polyfit(1/(Delta*Delta), Delay, 1)

    plt.figure()
    plt.plot(1/(Delta*Delta), Delay)
    if linear_fit:
        plt.plot(1/(Delta*Delta), a_*(1/(Delta*Delta)) + b_)
    plt.title('Delay vs Change Magnitude', fontsize=15)
    plt.xlabel(r'1/$\Delta^2$', fontsize=13)
    plt.ylabel('Average Detection Delay')
    if save_fig:
        figname = base_dir + 'Delay_vs_Delta' + str(seed)
        tpl.save(figname + ".tex", axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.savefig(figname+'.png', dpi=450)
    else:
        plt.show()



if __name__=='__main__':
    N_arl = 2500
    N = 2000 
    nc = 800
    num_trials = 250


    save_fig = True 
    save_data = save_fig 
    Alpha = np.linspace(0.001, 0.01, 5) 

    Delta = np.zeros((10,))
    for i in range(10):
        Delta[i] = 1/sqrt(i+6)

    # check_ARL(N_arl, Alpha, num_trials, save_data=save_data, save_fig=save_fig)
    check_Delay(N, nc, Delta, num_trials, seed=None, alpha=0.01, 
                save_data=save_data, save_fig=save_fig, p0=0.2)