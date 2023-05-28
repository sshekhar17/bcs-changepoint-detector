import os 
import pickle as pkl 
from tqdm import tqdm  
import numpy as np 
import matplotlib.pyplot as plt 
from time import time

from utils import * 
from ConfSeqs import * 
from changepoint import CPD_strategy2
from baselines import KCuSum



#============================================================================
def check_ARL(N, Alpha, num_trials, seed=None, save_fig=False, save_data=False):
    if seed is None: 
        seed = int(time()%10000)
    d = 10
    mean1 = np.zeros((d,))
    cov1 = np.eye(d)

    Source_1 = partial(
        Multivariate_Gaussian_Source, mean0=mean1, cov0=cov1, nc=None
    )

    Source_2 = partial(
        Multivariate_Gaussian_Source, mean0=mean1, cov0=cov1, nc=None
    )

    Source_3 = partial(
        Multivariate_Gaussian_Source, mean0=mean1, cov0=cov1, nc=None
    )


    Source = partial(TwoSample_Source,  
                                Source1=Source_1, Source1_kwargs={}, 
                                Source2=Source_2, Source2_kwargs={}, 
                                Source3=Source_3, Source3_kwargs={}, 
    )
    Source_kwargs = {'nc':None, 'two_dim_output':True}

    # fwdCS_func = get_MMD_CS_BR 
    fwdCS_func = get_MMD_CS_MR 
    fwdCS_kwargs = {}

    ARL = np.zeros(Alpha.shape)
    seed_ = seed
    for _ in tqdm(range(num_trials)):
        num_stopped = 0
        for i, alpha in enumerate(Alpha): 
            seed_ = get_new_seed(seed_)
            np.random.seed(seed_)
            data = Source(N=N, **Source_kwargs)
            Result =  CPD_strategy2(data, tmax=N, 
                                fwdCS_func=fwdCS_func,
                                fwdCS_kwargs=fwdCS_kwargs,
                                backCS_func=fwdCS_func,
                                backCS_kwargs=fwdCS_kwargs,
                                check_disjoint=check_disjoint_univariate_two_CS2,
                                alpha=alpha,
                                two_sample_data=True, 
                                verbose=False)
            if Result['stopped']: 
                ARL[i] += Result['stopping_time']
                num_stopped += 1
        if num_stopped==0:
            ARL[i] = N
        else:
            ARL[i] /= num_stopped
    # ARL /= num_trials  

    base_dir = './data/TwoSample/'
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


def check_Delay(N, nc, Delta, num_trials, d=10, seed=None, alpha=0.01, 
                    save_data=False, save_fig=False, 
                    xmax=12, linear_fit=True):
    if seed is None: 
        seed = int(time()%10000)

    mean1 = np.zeros((d,))
    cov1 = np.eye(d)
    cov2 = np.eye(d)

    Source_1 = partial(
        Multivariate_Gaussian_Source, mean0=mean1, cov0=cov1, nc=None
    )

    Source_2 = partial(
        Multivariate_Gaussian_Source, mean0=mean1, cov0=cov1, nc=None
    )

    # fwdCS_func = get_MMD_CS_BR 
    fwdCS_func = get_MMD_CS_MR 
    fwdCS_kwargs = {}
    # estimate the mmd distance for the post change distribution 

    Delay = np.zeros(Delta.shape)
    MMD_distance = np.zeros(Delta.shape)
    seed_ = seed
    for trial in tqdm(range(num_trials)):
        for i, delta in enumerate(Delta): 
            mean2 = delta*np.ones((d,))
            Source_3 = partial(
                Multivariate_Gaussian_Source, mean0=mean2, cov0=cov2, nc=None
            )


            Source = partial(TwoSample_Source,  
                                        Source1=Source_1, Source1_kwargs={}, 
                                        Source2=Source_2, Source2_kwargs={}, 
                                        Source3=Source_3, Source3_kwargs={}, 
            )
            Source_kwargs = {'nc':nc, 'two_dim_output':True}
            if trial==0:
                X_, Y_ = Source_1(N=10000), Source_3(N=10000)
                MMD_distance[i] = sqrt(computeMMD(X_, Y_))

            seed_ = get_new_seed(seed_)
            np.random.seed(seed_)
            data = Source(N=N, **Source_kwargs)
            Result =  CPD_strategy2(data, tmax=N, 
                                fwdCS_func=fwdCS_func,
                                fwdCS_kwargs=fwdCS_kwargs,
                                backCS_func=fwdCS_func,
                                backCS_kwargs=fwdCS_kwargs,
                                check_disjoint=check_disjoint_univariate_two_CS2,
                                alpha=alpha,
                                two_sample_data=True, 
                                verbose=False)
            Delay[i] += max(0, Result['stopping_time']-nc)
    #####################################################
    Delay /= num_trials  
    assert(xmax>0)
    MMD_distance = MMD_distance / (np.min(MMD_distance)*sqrt(xmax))
    if linear_fit: 
        a_, b_ = np.polyfit(1/(MMD_distance*MMD_distance), Delay, deg=1)

    base_dir = './data/GaussianMean/'
    if save_data: 
        data_dict = {
            'Delta':Delta, 'Delay':Delay, 'N':N, 'nc':nc, 'Source':Source, 
            'Source_kwargs':Source_kwargs, 'fwdCS_func':fwdCS_func, 
            'fwdCS_kwargs':fwdCS_kwargs, 'alpha':alpha, 
            'MMD_distance':MMD_distance
        }
        filename = base_dir + 'Delay_vs_MMD'+str(seed)+'.pkl' 
        with open(filename, 'wb') as f:
            pkl.dump(data_dict, f) 


    plt.figure()
    plt.plot(1/(MMD_distance*MMD_distance), Delay)
    if linear_fit:
        plt.plot(1/(MMD_distance*MMD_distance), a_*(1/(MMD_distance*MMD_distance)) + b_)
    plt.title('Delay vs Change Magnitude', fontsize=15)
    plt.xlabel(r'1/$\Delta^2$', fontsize=13)
    plt.ylabel('Average Detection Delay')
    if save_fig:
        figname = base_dir + 'Delay_vs_MMD' + str(seed)
        tpl.save(figname + ".tex", axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.savefig(figname+'.png', dpi=450)
    else:
        plt.show()

    print(MMD_distance)
    #####################################################

if __name__=='__main__':
    N_arl = 500
    N = 200 
    nc = 80
    num_trials = 5

    save_fig = False 
    save_data = save_fig 
    Alpha = np.linspace(0.001, 0.01, 5) 

    Delta = np.zeros((10,))
    for i in range(10):
        Delta[i] = 1/sqrt(i+5)
    
    Delta = Delta*2
    check_Delay(N, nc, Delta, num_trials, seed=None, alpha=0.01, 
                save_data=save_data, save_fig=save_fig)
    # check_ARL(N_arl, Alpha, num_trials//10, save_fig=save_fig, save_data=save_data)