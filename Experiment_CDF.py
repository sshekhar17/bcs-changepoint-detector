import os 
import pickle as pkl 
from tqdm import tqdm  
import numpy as np 
import matplotlib.pyplot as plt 
from time import time

from utils import * 
from ConfSeqs import * 
from changepoint import CPD_strategy2
import seaborn as sns

############################################
def get_KS_CS_width(t, alpha, factor=0.5):
    """
        with of the confidence sequence for KS distance 
        derived by Howard-Ramdas 2021  
    """
    t = max(1, t)
    return factor*1.7*sqrt((log (1 + log(t)) + 0.8*log(1612/alpha) )/t) 

def get_KS_dist(Source, Source_kwargs, N=100000): 
    Source_kwargs['nc']=N//2
    XX = Source(N=N, **Source_kwargs) 
    d, _ = stats.ks_2samp(XX[:N//2], XX[N//2:]) 
    return d

def get_empirical_CDF(grid, X):
    assert(len(X)>0)
    F = np.array([
        np.sum(X<=x) for x in grid 
    ])
    F = F/len(X) 
    return F 

############################################
def CPD_strategyKS(XX, tmax, alpha=0.01,factor=0.5):
    assert 4<=tmax<=len(XX) 
    Result = {'stopped':False, 'stopping_time':tmax, 'change_magnitude':0, 
                'change_point':tmax}
    for n in range(2, tmax+1): 
        X = XX[:n]
        for t in range(1, n-1): 
            stat, _ = stats.ks_2samp(X[:t], X[t:])
            w = (get_KS_CS_width(t, alpha=alpha, factor=factor) +
                get_KS_CS_width(n-t, alpha=alpha, factor=factor)) 
            if stat>w:
                Result['stopping_time'] = n 
                Result['change_point'] = t+1
                Result['change_magnitude'] = stat 
                Result['stopped'] = True 
                return Result 
    return Result 
                  


############################################
def check_Delay(N, nc, Delta, num_trials, p=0.5,  alpha=0.01, 
                    seed=None, save_data=False, save_fig=False, 
                    xmax=12, linear_fit=True, factor=0.5):
    if seed is None: 
        seed = int(time()%10000)

    experiment_name = 'CDFexperiment'
    #### Source 
    df1, loc1, scale1 = 3, 0.0, 1.0 
    df2, loc2, scale2 = 3, 0.0, 2.0 
    Source = Univariate_T_Source
    #### FWD CS 

    Delay = np.zeros(Delta.shape)
    KSdist = np.zeros(Delta.shape)
    seed_ = seed
    for _ in tqdm(range(num_trials)):
        for i, delta in enumerate(Delta): 
            loc2 = delta 
            Source_kwargs = {
                'df1':df1, 'loc1':loc1, 'scale1':scale1, 
                'df2':df2, 'loc2':loc2, 'scale2':scale2, 
                'nc':nc 
            }

            seed_ = get_new_seed(seed_)
            np.random.seed(seed_)
            XX = Source(N=N, **Source_kwargs)
            Result =  CPD_strategyKS(XX, tmax=N, 
                                alpha=alpha,
                                factor=factor)
            Delay[i] += max(0, Result['stopping_time']-nc)
            KSdist[i] = get_KS_dist(Source, Source_kwargs)
    Delay /= num_trials  
    # normalize the x-axis 
    assert (xmax>0)
    Delta = Delta /(np.min(Delta)*sqrt(xmax))

    base_dir = './data/CDF/'
    if save_data: 
        data_dict = {
            'Delta':Delta, 'Delay':Delay, 'N':N, 'nc':nc, 'Source':Source, 
            'Source_kwargs':Source_kwargs,'alpha':alpha
        }
        filename = base_dir + 'Delay_vs_Delta'+str(seed)+'.pkl' 
        with open(filename, 'wb') as f:
            pkl.dump(data_dict, f) 
    
    # fit the linear curve 
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


#########################################################################
###################################################################
def run_KS_CPD_trial(Source, Source_kwargs, 
                N=500, alpha=0.01,
                seed=None, tmax=None, 
                plot_results=False, 
                save_fig=False, 
                figname=None, 
                factor=0.5
                ):
    # print('Source Kwargs inside trial', Source_kwargs)
    tmax = N if tmax is None else tmax 
    # set the random seed 
    if seed is not None:
        np.random.seed(seed) 
    # get the data 
    data = Source(N=N, **Source_kwargs)
    Result = CPD_strategyKS(data, tmax=tmax, alpha=alpha, factor=factor)
    nc = Source_kwargs['nc'] 
    ###################################################################
    tau = Result['stopping_time']
    change_point = Result['change_point']
    t, t1 = change_point+1, tau-change_point    
    if Result['stopped']:
        XX = data[:tau+1]
        X1, X2 = XX[:t], XX[t:] 
        grid = np.linspace(np.min(XX), np.max(XX), 1000) 
        F1, F2 = get_empirical_CDF(grid, X1), get_empirical_CDF(grid, X2)
        ### shorten the width to make the non-overlapping parts of the 
        ### figure visible. Note that this is only being done for 
        ### plotting the figures. 
        w1 = 0.8*get_KS_CS_width(t, alpha, factor)
        w2 = 0.8*get_KS_CS_width(t1, alpha, factor) 
        L1, L2 = np.maximum(0, F1-w1), np.maximum(0, F2-w2)
        U1, U2 = np.minimum(1, F1+w1), np.minimum(1, F2+w2)

    if plot_results and Result['stopped']:
        palette = sns.color_palette(n_colors=10)
        plt.figure() 
        plt.plot(grid, F1, color=palette[0]) 
        plt.fill_between(grid, L1, U1, color=palette[0], alpha=0.3)
        plt.plot(grid, F2, color=palette[1]) 
        plt.fill_between(grid, L2, U2, color=palette[1], alpha=0.3)
        plt.title(f'Changepoint at T={nc}', fontsize=15)

        if save_fig: 
            if figname is None: 
                figname = 'temp' 
            elif figname[-4:] == '.tex':
                figname = figname[:-4]
            
            plt.savefig(figname+'.png', dpi=450)
            tpl.save(figname+'.tex', axis_width=r'\figwidth', axis_height=r'\figheight')
        else:
            plt.show()
    ###################################################################
    ### approximately compute teh true KS distance 
    KSdist= get_KS_dist(Source, Source_kwargs)
    Result['KSdist'] = KSdist
    return Result


def run_CPD_experimentKS(Source, Source_kwargs, 
                N=500, alpha=0.01, nc=250,
                seed=None, tmax=500, 
                plot_results=False, 
                num_trials=200, 
                epsilon=None, 
                save_fig=False, 
                experiment_name = None, 
                base_dir = None
                ):
    ###-------------
    if epsilon is None: 
        epsilon = get_KS_dist(Source, Source_kwargs)
    ### Some setup for saving figures and data 
    if experiment_name is None: 
        experiment_name = 'TempExperiment' 
    if seed is not None: 
        experiment_name = experiment_name + str(seed)
    if base_dir is None and save_fig: 
        base_dir = os.path.join('./data', experiment_name, '')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    ###-------------
    ## Initialization 
    ChangePoints = np.zeros((num_trials,)) 
    Epsilon = np.zeros((num_trials, ))
    Delay = np.zeros((num_trials,))

    for trial in tqdm(range(num_trials)):
        # update the seed 
        seed = get_new_seed(seed)
        #### TODO: this should not be needed 
        Source_kwargs['nc'] = nc
        # print(Source_kwargs)
        Result = run_KS_CPD_trial(Source, Source_kwargs, 
                N=N, alpha=alpha,
                seed=None, 
                plot_results=False, 
                save_fig=False, 
                factor=0.5
                )
        
        # Extract the information from the result 
        T = Result['change_point']
        eps = Result['change_magnitude']
        ChangePoints[trial] = T  
        Epsilon[trial] = eps 
        if T is not None:
            Delay[trial] = max(0, T-nc)
        else:
            Delay[trial] = N
    # Plot the results 
    if plot_results:
        if save_fig:
            base_figname = os.path.join(base_dir, experiment_name)
        else:
            base_figname=None 
        # plot the experiment results 
        plot_experiment_results1(nc, ChangePoints, Epsilon, Delay,  epsilon,
                                save_fig=save_fig, base_figname=base_figname)
        # save the data 
        if save_fig: 
            data_path = os.path.join(base_dir, experiment_name + '.pkl')
            data_dict = {
                'Source':Source, 'Source_kwargs':Source_kwargs, 
                'N':N, 'num_trials':num_trials, 
                'experiment_name':experiment_name, 
                'ChangePoints':ChangePoints, 
                'Epsilon':Epsilon, 'epsilon':epsilon, 
                'Delay':Delay, 'seed':seed 
            }
            with open(data_path, 'wb') as f:
                pkl.dump(data_dict, f) 


def visualize_performance(delta=0.5, nc=400, N=1000, alpha=0.01, 
                            num_trials=10, save_fig=False, 
                            plot_results=True, seed=None, 
                            experiment_name=None):

    if experiment_name is None: 
        experiment_name = 'CDF'
    if seed is None:
        seed = int(time()%10000)
    #########################################
    experiment_name_ =experiment_name + str(seed)
    if save_fig:
        base_dir = os.path.join('./data', experiment_name_, '')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        figname = base_dir + 'CS'
    else:
        base_dir = None 
        figname = None 
    #########################################


    #### Source 
    df1, loc1, scale1 = 3, 0.0, 1.0 
    df2, loc2, scale2 = 3, delta, 2.0 
    Source = Univariate_T_Source

    Source_kwargs = {
        'df1':df1, 'loc1':loc1, 'scale1':scale1, 
        'df2':df2, 'loc2':loc2, 'scale2':scale2, 
        'nc':nc 
    }
    ### One trial to plot the CSs
    Result = run_KS_CPD_trial(Source, Source_kwargs, N=N, 
    alpha=alpha, save_fig=save_fig, plot_results=plot_results, figname=figname)
    #### Several trials to plot the histograms
    run_CPD_experimentKS(Source, Source_kwargs, N=N, alpha=alpha, nc=nc, 
                            tmax=N, plot_results=plot_results,
                            save_fig=save_fig, num_trials=num_trials,
                            epsilon=None, 
                            experiment_name='CDF', 
                            seed=seed, 
                            base_dir=base_dir 
                            )
    return Result


def Delay_vs_Delta_KS():
    N_arl = 50
    N = 1000 
    nc = 400
    num_trials = 250

    save_fig = True 
    save_data = save_fig 
    Alpha = np.linspace(0.001, 0.01, 10) 

    Delta = np.zeros((10,))
    for i in range(10):
        Delta[i] = 4/sqrt(i+1)

    # check_ARL(N_arl, Alpha, num_trials, save_fig=save_fig, save_data=save_data)
    check_Delay(N, nc, Delta, num_trials, seed=None, alpha=0.01, 
                save_data=save_data, save_fig=save_fig)

#########################################################################
if __name__=='__main__':

    Result = visualize_performance(delta=1.0, N=1000, nc=400, num_trials=200, save_fig=True)
    # Delay_vs_Delta_KS()