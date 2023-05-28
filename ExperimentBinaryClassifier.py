import os 
import pickle as pkl 
from time import time
from tqdm import tqdm  
from math import sin, cos, pi

import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt 
import tikzplotlib as tpl

from utils import * 
from ConfSeqs import * 
from changepoint import CPD_strategy2


###########################################################################
## Generate the source distribution data 
def generate_illustrative_figure():
    radius=1.5
    N = 50
    mu0 = radius*np.array([-1,0])
    Z_0 = stats.multivariate_normal(mean=mu0, cov=np.eye(2)).rvs(size=N)

    mu1 = radius*np.array([1,0])
    Z_1 = stats.multivariate_normal(mean=mu1, cov=np.eye(2)).rvs(size=N)

    plt.figure()
    plt.scatter(Z_0[:,0], Z_0[:,1], marker='o', label=r'$L=0$')
    plt.scatter(Z_1[:,0], Z_1[:,1], marker='x', label=r'$L=1$')
    # circle0 = plt.Circle(mu0, 1, fill=False)
    # plt.gca().add_patch(circle0)
    # plt.axis("equal")
    plt.plot(np.zeros((100,)), np.linspace(-4, 4, 100), 'k--',
                linewidth=2, label=r'$h^*$')
    plt.ylim([-6,6])
    plt.legend(fontsize=13)
    tpl.save("Source.tex", axis_width=r'\figwidth', axis_height=r'\figheight')

## Generate the harmful target distribution data 
    theta = pi/4 
    mu0_ = -radius*np.array([cos(theta), sin(theta)])
    Z_0_ = stats.multivariate_normal(mean=mu0_, cov=np.eye(2)).rvs(size=N)
    mu1_ = radius*np.array([cos(theta), sin(theta)])
    Z_1_ = stats.multivariate_normal(mean=mu1_, cov=np.eye(2)).rvs(size=N)

    plt.figure()
    plt.scatter(Z_0_[:,0], Z_0_[:,1], marker='o', label=r'$L=0$')
    plt.scatter(Z_1_[:,0], Z_1_[:,1], marker='x', label=r'$L=1$')
    # circle0 = plt.Circle(mu0, 1, fill=False)
    # plt.gca().add_patch(circle0)
    # plt.axis("equal")
    plt.plot(np.zeros((100,)), np.linspace(-4, 4, 100), 'k--',
                linewidth=2, label=r'$h^*$')
    ## New classifier 
    xx = np.linspace(-2,2, 100)
    yy = -xx 
    plt.plot(xx, yy, 'r', linestyle='dotted', linewidth=2, label=r'$h^{**}$')
    plt.ylim([-6,6])
    plt.legend(fontsize=13)
    tpl.save("Target.tex", axis_width=r'\figwidth', axis_height=r'\figheight')


    mu0 = 2*radius*np.array([-1,0])
    Z_0 = stats.multivariate_normal(mean=mu0, cov=np.eye(2)).rvs(size=N)
    mu1 = 2*radius*np.array([1,0])
    Z_1 = stats.multivariate_normal(mean=mu1, cov=np.eye(2)).rvs(size=N)


    plt.figure()
    plt.scatter(Z_0[:,0], Z_0[:,1], marker='o', label=r'$L=0$')
    plt.scatter(Z_1[:,0], Z_1[:,1], marker='x', label=r'$L=1$')
    # circle0 = plt.Circle(mu0, 1, fill=False)
    # plt.gca().add_patch(circle0)
    # plt.axis("equal")
    plt.plot(np.zeros((100,)), np.linspace(-4, 4, 100), 'k--',
                linewidth=2, label=r'$h^*$')
    plt.ylim([-6,6])
    plt.legend(fontsize=13)
    tpl.save("Source_new.tex", axis_width=r'\figwidth', axis_height=r'\figheight')


def check_ARL(N, Alpha, num_trials, seed=None, save_data=False, save_fig=False,
                radius=1.5):
    if seed is None: 
        seed = int(time()%10000)
    experiment_name = 'DistributionShift'
    #### Source 
    Source = ClassifierOutput
    Source_kwargs = {'mu':np.array([1,0]), 'radius':radius, 'theta':pi/4, 
                        'nc':None, 'w':np.array([1,0])}
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

    base_dir = './data/DistributionShift/'
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


def check_Delay(N, nc, Theta, num_trials, seed=None, alpha=0.01, 
                    save_data=False, save_fig=False, radius=1.5, 
                    Delta=None, xmax=12, linear_fit=True):
    if seed is None: 
        seed = int(time()%10000)
    Source = ClassifierOutput
    #### FWD CS 
    fwdCS_func = get_Bounded_EB_CS
    fwdCS_kwargs = {}

    Delay = np.zeros(Theta.shape)
    seed_ = seed
    for _ in tqdm(range(num_trials)):
        for i, theta in enumerate(Theta): 
            Source_kwargs = {
                'mu':np.array([1,0]), 'radius':radius,
                'theta':theta, 'nc':nc, 'w':np.array([1,0])
                        }

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

    # Compute the Delta Values 
    if Delta is None:
        loss0 = 1-stats.norm.cdf(radius)

        Delta = np.zeros(Theta.shape)
        for i, th in enumerate(Theta): 
            r1 = radius*(cos(th))
            loss1 =  1-stats.norm.cdf(r1)
            Delta[i] =  loss1-loss0

    for delay_, delta_ in zip(Delay, Delta):
        print(delta_, delay_)

    base_dir = './data/DistributionShift/'
    # Rescale the x-axis 
    Delta = Delta / (np.min(Delta)*sqrt(xmax))
    if linear_fit: 
        a_, b_ = np.polyfit(1/(Delta*Delta), Delay, deg=1)

    if save_data: 
        data_dict = {'Theta':Theta,
            'Delta':Delta, 'Delay':Delay, 'N':N, 'nc':nc, 'Source':Source, 
            'Source_kwargs':Source_kwargs, 'fwdCS_func':fwdCS_func, 
            'fwdCS_kwargs':fwdCS_kwargs, 'alpha':alpha
        }
        filename = base_dir + 'Delay_vs_Delta'+str(seed)+'.pkl' 
        with open(filename, 'wb') as f:
            pkl.dump(data_dict, f) 

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
    N_arl = 200
    N = 2000 
    nc = 800
    num_trials = 250
    radius = 1.5

    save_fig = True 
    save_data = True 
    Alpha = np.linspace(0.001, 0.01, 5)*10

    n_theta = 10 
    Theta = np.zeros((n_theta,))
    Delta = np.zeros(Theta.shape)
    theta0 = pi/5
    for i in range(n_theta):
        theta = theta0 + ((i+1)/n_theta)*(pi/2 - theta0)
        loss0 = 1-stats.norm.cdf(radius)
        r1 = radius*(cos(theta))
        loss1 =  1-stats.norm.cdf(r1)

        Theta[i] = theta
        Delta[i] = loss1 - loss0

        print(f'{Theta[i]:.2f}, {1/(Delta[i]**2):.2f}')

    # check_ARL(N_arl, Alpha, num_trials, save_data=save_data, save_fig=save_fig)
    check_Delay(N, nc, Theta, num_trials, seed=None, alpha=0.01, 
                save_data=save_data, save_fig=save_fig)


    
