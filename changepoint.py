"""
Implementation of our two changepoint detection strategies based on confidence sequences
"""

import os 
import pickle as pkl 
from tqdm import tqdm  
import numpy as np 
import matplotlib.pyplot as plt 
from time import time
from math import cos, pi 
from scipy.stats import chi2

from utils import * 
from ConfSeqs import * 


###################################################################
def CPD_strategy1(XX, tmax, CS_func, CS_kwargs, check_disjoint,
                    alpha=0.05, verbose=True):
    """ 
    CS based online CPD strategy: uses only the forward CS

    Inputs: 
        XX      :numpy array of all the observations 
        tmax    :the time up to which we have to check for stopping 
                    usually should be set to the number of 
                    observations in XX 
        CS_func : function handle 
                    given XX, it computes the confidence sequence 
                    Has two arguments XX, and alpha, 
                    plus some additional keyword args 
        CS_kwargs   : dict
                        keyword arguments for CS_func 
        check_disjoint  : function handle. Takes in the output of CS_func, 
                            and tmax. Returns two values:
                                :stopped    bool, True if null is 
                                            rejected before tmax
                                :stopping_time  int, time at which 
                                            the null is rejected, 
                                            otherwise equal to tmax 
        alpha       :float, the upper bound on false positive rate 
        verbose     :bool, prints out the results if true 
    """
    # get the forward CS 
    forwardCS = CS_func(XX=XX, alpha=alpha, **CS_kwargs)
    # find out the stopping condition: 
    stopped, stopping_time = check_disjoint(forwardCS, tmax)
    # print out the result 
    if verbose:
        if stopped:
            print(f'Completed Experiment: Rejected Null at t={stopping_time}')
        else:
            print(f'Completed Experiment: Null not rejected')
    
    # Create the results dictionary 
    Result = {
        'XX':XX, 'forwardCS':forwardCS, 'stopped':stopped, 
        'stopping_time':stopping_time
    }

    return Result


###################################################################
def CPD_strategy2(data, tmax, 
                    fwdCS_func, fwdCS_kwargs,
                    backCS_func, backCS_kwargs,
                    check_disjoint, alpha=0.01,
                    two_sample_data=False, 
                    verbose=True, initial=10):
    """
    CS based online CPD strategy: uses one forward CS and a new 
    backward CS every time step. 
    """
    if two_sample_data:
        XX, YY = data 
        # get the forward CS 
        forwardCS = fwdCS_func(XX=XX, YY=YY, alpha=alpha, **fwdCS_kwargs)
    else: 
        XX = data
        # get the forward CS 
        forwardCS = fwdCS_func(XX=XX, alpha=alpha, **fwdCS_kwargs)
    # run the main loop, creating a new backward CS every step 
    stopped, stopping_time = False, tmax 
    for i in range(initial, tmax):
        if two_sample_data:
            data_ = (XX[:i+1], YY[:i+1])
        else:
            data_ = XX[:i+1]
        # create a new backward CS 
        backCS = get_backward_CS(data_, backCS_func, backCS_kwargs, alpha=alpha,
                                    two_sample_data=two_sample_data)
        # check if disjoint 
        stopped = check_disjoint(forwardCS, backCS, t=i+1)
        if stopped: 
            stopping_time = i+1 
            break 
    
    if verbose:
        if stopped:
            print(f'Completed Experiment: Rejected Null at t={stopping_time}')
        else:
            print(f'Completed Experiment: Null not rejected')

    # Create the results dictionary 
    Result = {
        'XX':data, 'forwardCS':forwardCS, 'backCS':backCS, 
         'stopped':stopped, 'stopping_time':stopping_time
    }
    return Result


###################################################################
def run_CPD_trial(Source, Source_kwargs, 
                fwdCS_func, fwdCS_kwargs,
                check_disjoint,
                backCS_func=None, backCS_kwargs=None,
                N=500, alpha=0.001, use_back_CS=True,
                seed=None, tmax=500, 
                verbose=False, 
                initial_phase=20, 
                plot_results=False, 
                mean = None, 
                save_fig=False, 
                figname=None, 
                fig_kwargs={}, 
                two_sample_data=False
                ):
    # some preprocessing
    if use_back_CS:
        if backCS_func is None:
            backCS_func = fwdCS_func
        if backCS_kwargs is None:
            backCS_kwargs = fwdCS_kwargs
    # set the random seed 
    if seed is not None:
        np.random.seed(seed) 
    # get the data 
    data = Source(N=N, **Source_kwargs)
    if use_back_CS:
        # use strategy 2 with both fwd and back CSs
        Result = CPD_strategy2(data, tmax, 
                    fwdCS_func, fwdCS_kwargs,
                    backCS_func, backCS_kwargs,
                    check_disjoint, alpha=alpha, verbose=verbose,
                    two_sample_data=two_sample_data)
        fwdCS, backCS = Result['forwardCS'], Result['backCS']
        change_point, change_magnitude, I0 = get_change_time_and_magnitude2(fwdCS, backCS)
        # update the result dictionary 
        Result['change_point'] = change_point
        Result['change_magnitude'] = change_magnitude
        Result['disjoint_interval'] = I0
    else: 
        #use strategy 1 with a single fwd CS 
        Result = CPD_strategy1(data, tmax, fwdCS_func, fwdCS_kwargs, check_disjoint,
                    alpha=alpha, verbose=verbose)
    # plot results if using back CS 
    if plot_results and use_back_CS: 

        plot_CPD_results1(data=data, Result=Result, mean=mean, nc=nc,
                            save_fig=save_fig, figname=figname, **fig_kwargs)
    return Result


def run_CPD_experiment(Source, Source_kwargs, 
                fwdCS_func, fwdCS_kwargs,
                check_disjoint,
                backCS_func=None, backCS_kwargs=None,
                N=500, alpha=0.001, nc=250,
                use_back_CS=True,
                seed=None, tmax=500, 
                verbose=False, 
                initial_phase=20, 
                plot_results=False, 
                num_trials=200, 
                epsilon=0.5, 
                save_fig=False, 
                two_sample_data=False, 
                experiment_name = None, 
                base_dir = None
                ):
    ###-------------
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
    LeftEndPoint = np.zeros((num_trials,))
    RightEndPoint = np.zeros((num_trials,))
    Delay = np.zeros((num_trials,))


    for trial in tqdm(range(num_trials)):
        # update the seed 
        seed = get_new_seed(seed)

        Result = run_CPD_trial(Source, Source_kwargs, 
                fwdCS_func, fwdCS_kwargs,
                check_disjoint,
                backCS_func=backCS_func,
                backCS_kwargs=backCS_kwargs,
                N=N, alpha=alpha,
                use_back_CS=use_back_CS,
                seed=seed,
                tmax=tmax, 
                verbose=False, 
                initial_phase=initial_phase, 
                plot_results=False, 
                two_sample_data=two_sample_data
            )
        # Extract the information from the result 
        I0 = Result['disjoint_interval']
        T = Result['change_point']
        eps = Result['change_magnitude']
        ChangePoints[trial] = T  
        Epsilon[trial] = eps 
        if len(I0):
            LeftEndPoint[trial] = min(I0)
            RightEndPoint[trial] = max(I0)
        else:
            LeftEndPoint[trial] = 0
            RightEndPoint[trial] = N
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
                'fwdCS_func':fwdCS_func, 'fwdCS_kwargs':fwdCS_kwargs, 
                'N':N, 'num_trials':num_trials, 
                'experiment_name':experiment_name, 
                'ChangePoints':ChangePoints, 
                'Epsilon':Epsilon, 'epsilon':epsilon, 
                'Delay':Delay, 'seed':seed 
            }
            with open(data_path, 'wb') as f:
                pkl.dump(data_dict, f) 






if __name__=='__main__':
    ###################################################
    ## General parameters 
    N=2000
    nc = 800 
    alpha=0.01 
    # random seed 
    seed = int(time()%10000)
    initial_phase = 10
    Gaussian_Experiment = False
    BoundedMean_Experiment = False  
    TwoSample_Experiment=False 
    Quantile_Experiment=False
    DistributionShift_Experiment=True
    num_trials = 250
    save_fig = True

    ###################################################

    if Gaussian_Experiment:
        experiment_name = 'GaussianMean'
        #### Source 
        m0 = 0 
        m1=0.35
        Source = Univariate_Gaussian_Source
        Source_kwargs = {'mean0':m0, 'mean1':m1, 'nc':nc}
        #### FWD CS 
        fwdCS_func = get_Gaussian_CS
        fwdCS_kwargs = {}
        epsilon = abs(m1-m0)
        mean = np.ones((N,))*m0
        mean[nc:] = m1 
        two_sample_data = False
        if nc<N:
            title = f'Gaussian Source with Changepoint at T={nc}'
        else:
            title = f'Gaussian Source with no Changepoint'
        fig_kwargs = {
            'title':title
        }



    elif BoundedMean_Experiment: # Bernoulli Experiment 
        experiment_name = 'BoundedMean'
        p0, p1 = 0.5, 0.7
        Source = BoundedSource
        Source_kwargs = {'p0':p0, 'p1':p1, 'nc':nc}
        #### FWD CS 
        fwdCS_func = get_Bounded_EB_CS
        fwdCS_kwargs = {}
        epsilon = abs(p1-p0)
        mean = np.ones((N,))*p0
        mean[nc:] = p1 
        two_sample_data = False
        if nc<N:
            title = f'Bounded Source with Changepoint at T={nc}'
        else:
            title = f'Bounded Source with No Changepoint'
        fig_kwargs = {
            'title':title
        }
    
    elif TwoSample_Experiment: ### TWO SAMPLE EXPERIMENT 
        experiment_name = 'TwoSample'
        d = 10
        mean1 = np.zeros((d,))
        cov1 = np.eye(d)

        mean2 = 1.5*np.ones((d,))
        cov2 = np.diag(np.random.random((d,))*5) 

        Source_1 = partial(
            Multivariate_Gaussian_Source, mean0=mean1, cov0=cov1, nc=None
        )

        Source_2 = partial(
            Multivariate_Gaussian_Source, mean0=mean1, cov0=cov1, nc=None
        )

        Source_3 = partial(
            Multivariate_Gaussian_Source, mean0=mean2, cov0=cov2, nc=None
        )


        Source = partial(TwoSample_Source,  
                                    Source1=Source_1, Source1_kwargs={}, 
                                    Source2=Source_2, Source2_kwargs={}, 
                                    Source3=Source_3, Source3_kwargs={}, 
        )
        Source_kwargs = {'nc':nc, 'two_dim_output':True}

        fwdCS_func = get_MMD_CS_BR 
        # fwdCS_func = get_MMD_CS_MR 
        fwdCS_kwargs = {}
        # estimate the mmd distance for the post change distribution 
        X_, Y_ = Source_1(N=10000), Source_3(N=10000)
        epsilon = computeMMD(X_, Y_)
        two_sample_data = True
        mean=np.zeros((N,))
        mean[nc:] = epsilon
        if nc<N:
            title = f'Two-Sample Source with Changepoint at T={nc}'
        else:
            title = f'Two-Sample Source with no Changepoint'
        fig_kwargs = {
            'title':title
        }



    elif Quantile_Experiment:
        experiment_name = 'QuantileExperiment'
        #### Source 
        df1, loc1, scale1 = 1, 0.0, 1.0 
        df2, loc2, scale2 = 2, 1.0, 1.0 
        p=0.5
        quantile1 = tdist.ppf(q=p, df=df1, loc=loc1, scale=scale1)
        quantile2 = tdist.ppf(q=p, df=df2, loc=loc2, scale=scale2)

        Source = Univariate_T_Source
        Source_kwargs = {
            'df1':df1, 'loc1':loc1, 'scale1':scale1, 
            'df2':df2, 'loc2':loc2, 'scale2':scale2, 
            'nc':nc 
        }
        #### FWD CS 
        fwdCS_func = get_Quantile_CS_HR
        fwdCS_kwargs = {'p':p}
        epsilon = abs(quantile1 - quantile2)
        mean = np.ones((N,))*quantile1
        mean[nc:] = quantile2 
        two_sample_data = False
        if nc<N:
            title = f'T-Source with Changepoint at T={nc}'
        else:
            title = f'T-Source with no Changepoint'

        fig_kwargs = {
            'title':title, 'ymin':min(loc1, loc2)-1, 'ymax':max(loc1, loc2)+1
        }
    elif DistributionShift_Experiment: 
        experiment_name = 'DistributionShift'
        #### Source 
        radius, theta = 1.5, pi/3
        Source = ClassifierOutput
        Source_kwargs = {'mu':np.array([1,0]), 'epsilon':radius, 'theta':theta, 
                            'nc':nc, 'w':np.array([1,0])}
        #### FWD CS 
        fwdCS_func = get_Bounded_EB_CS
        fwdCS_kwargs = {}
        # compute the losses 
        loss0 = 1-stats.norm.cdf(radius)
        r1 = radius*(cos(theta))
        loss1 =  1-stats.norm.cdf(r1)
        epsilon = loss1 - loss0
        mean = np.ones((N,))*loss0
        mean[nc:] = loss1 
        two_sample_data = False
        if nc<N:
            title = f'Changepoint at T={nc}'
        else:
            title = f'No Changepoint'

        fig_kwargs = {
            'title':title, 'ymin':0, 'ymax':1
        }




    ### Back CS 
    backCS_func = fwdCS_func 
    backCS_kwargs = fwdCS_kwargs
    ### check_disjoint function
    check_disjoint = check_disjoint_univariate_two_CS2

    # run the experiment 

    #########################################
    if seed is not None:
        experiment_name_ =experiment_name + str(seed)
    else:
        experiment_name_ = experiment_name 
    if save_fig:
        base_dir = os.path.join('./data', experiment_name_, '')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        figname = base_dir + 'CS'
    else:
        base_dir = None 
        figname = None 
    #########################################
    Result = run_CPD_trial(Source=Source, 
                            Source_kwargs=Source_kwargs, 
                            fwdCS_func=fwdCS_func, 
                            fwdCS_kwargs=fwdCS_kwargs, 
                            check_disjoint=check_disjoint,
                            backCS_func=backCS_func, 
                            backCS_kwargs=backCS_kwargs,
                            N=N,
                            alpha=alpha,
                            use_back_CS=True, 
                            tmax=N,
                            verbose=True,
                            plot_results=True, 
                            mean=mean, 
                            save_fig=save_fig, 
                            fig_kwargs=fig_kwargs, 
                            two_sample_data=two_sample_data, 
                            seed=seed, 
                            figname=figname
                            )

    ############## result over several trials 
    run_CPD_experiment(Source, Source_kwargs, 
                    fwdCS_func, fwdCS_kwargs,
                    check_disjoint,
                    backCS_func=None, backCS_kwargs=None,
                    N=N, alpha=alpha, nc=nc,
                    use_back_CS=True, tmax=N,
                    seed=seed, 
                    verbose=False, 
                    initial_phase=10, 
                    plot_results=True, 
                    num_trials=num_trials, 
                    epsilon=epsilon, 
                    save_fig=save_fig, 
                    two_sample_data=two_sample_data, 
                    experiment_name=experiment_name
                    )

