from functools import partial 
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 
from time import time

from utils import * 
from ConfSeqs import * 
from changepoint import CPD_strategy2



def KCuSum(XX, YY, kernel, threshold=0.1, delta=1e-5, 
            continue_sampling=False):
    n = len(XX) 
    assert(n==len(YY))
    kcusum_stat = np.zeros((n,))
    stat = 0
    stopped = False
    for i in range(n): 
        temp = 0
        if (i%2==1) and (i>=1): 
            x1, x2 = XX[i], XX[i-1] 
            y1, y2 = YY[i], YY[i-1]  
            temp = (
                kernel(x1, x2) + kernel(y1, y2) - kernel(x1, y2) - kernel(x2, y1)
                - delta 
            )
        # update the statistic 
        # print(temp)
        stat = max(0, stat+temp) 
        kcusum_stat[i] = stat 
        # check if it exceeds the rejection threshold 
        if not continue_sampling:
            if stat>threshold: 
                stopped = True
                return stopped, i+1 # stopping time 

    # otherwise return the array of statistics 
    if continue_sampling:
        return kcusum_stat
    else: 
        return stopped, i+1
def get_average_threshold_crossing(arr, th): 
    num_trials, N = arr.shape 
    arr_bool = arr>=th
    stopping_times = np.argmax(arr_bool, axis=1) 
    # compute the average 
    idx_ = (stopping_times > 0) 
    s_ = stopping_times[idx_]  
    if len(s_)==0: 
        print("no threshold crossing")
        return N 
    else: 
        return s_.mean()


def get_kcusum_threshold(N,  Source, Source_kwargs, kernel,
                        alpha, num_trials=100, num_iters=20, 
                        tol=1e-3, verbose=False): 
    Stat_vals = np.zeros((num_trials, N)) 
    for i in tqdm(range(num_trials)): 
        XX, YY = Source(N=N, **Source_kwargs)
        Stat_vals[i] = KCuSum(XX, YY, kernel, continue_sampling=True) 
    # find the appropriate threshold
    arl = 1/alpha 
    assert arl<=N 
    # do a binary search to find the best threshold 
    def func(th): 
        return -arl + get_average_threshold_crossing(Stat_vals, th) 
    th_min, th_max = np.min(Stat_vals), np.max(Stat_vals)
    th = simple_binary_search(func, th_min, th_max, 
                            num_iters=num_iters,
                            tol=1e-3, verbose=verbose)
    return th 
       

def calibrate_SCD_scheme(N, Source, Source_kwargs, fwdCS_func, alpha, num_trials, 
                        num_iters=20):

    def func(f):
        # fwdCS_func = get_MMD_CS_MR 
        fwdCS_kwargs = {'factor':f}
        arl = 0 
        for _ in tqdm(range(num_trials)):
            num_stopped = 0
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
                arl += Result['stopping_time']
                num_stopped += 1
        if num_stopped==0:
            arl = N
        else:
            arl /= num_stopped
        return arl -1/alpha
    factor = simple_binary_search(func, th_min=0.05, th_max=1.0, 
                                  num_iters=num_iters, verbose=False)    
    return factor 



def compare_delay(Source, N, nc, num_trials, d=10, seed=None,
                    alpha=0.01, learn_threshold=False, th_default=0.8, factor=0.1):
    if seed is None: 
        seed = int(time()%10000)
    if nc is None: 
        nc_ = 0 # used in calculating detection delay 
    else: 
        nc_ = nc
    kernel = GaussianKernel

    ### 
    ### The theoretical MR CS results in very conservative SCD methods, 
    ### In practice, reducing the width of the CS by a factor of 
    ### around 10 to 20 results in an ARL of 500 to 1000. 
    fwdCS_func = get_MMD_CS_MR 
    fwdCS_kwargs = {'factor':factor} 
    # find the appropriate threshold
    if learn_threshold:
        ## calibrate the CS based on data 
        # (only use for larger datasets that have enough prechange data)
        Source_kwargs = {'nc':None, 'two_dim_output':True}
        th = get_kcusum_threshold(N,  Source, Source_kwargs, kernel,
                            alpha, num_trials=num_trials, num_iters=50, 
                            tol=1e-3, verbose=True) 
    else:
        th = th_default
    # compare the performance of the two methods 
    Source_kwargs = {'nc':nc, 'two_dim_output':True}
    delay_BCS = 0
    delay_Kcusum= 0
    seed_ = seed
    num_stopped_BCS, num_stopped_Kcusum = 0, 0
    for _ in tqdm(range(num_trials)):
        # set the seed
        seed_ = get_new_seed(seed_)
        np.random.seed(seed_)
        # get the data 
        data = Source(N=N, **Source_kwargs)
        # run the main BCS experiment 
        Result_BCS =  CPD_strategy2(data, tmax=N, 
                            fwdCS_func=fwdCS_func,
                            fwdCS_kwargs=fwdCS_kwargs,
                            backCS_func=fwdCS_func,
                            backCS_kwargs=fwdCS_kwargs,
                            check_disjoint=check_disjoint_univariate_two_CS2,
                            alpha=alpha,
                            two_sample_data=True, 
                            verbose=False)
        if Result_BCS['stopped']:
            if Result_BCS['stopping_time']>=nc_:
                delay_BCS += max(0, Result_BCS['stopping_time']-nc_)
                num_stopped_BCS += 1
                # print(f"BCS STOPPING TIME {Result_BCS['stopping_time']}")
        # run the KCusum experiment 
        XX, YY = data
        kernel = GaussianKernel
        stopped_, st_=  KCuSum(XX, YY, kernel, threshold=th, delta=1e-5, 
            continue_sampling=False)      
        if stopped_ and st_>=nc_: 
            delay_Kcusum += max(0, st_-nc_)
            num_stopped_Kcusum += 1
            # print(f"KCUSUM STOPPING TIME {st_}")
    #####################################################
    # print(f"Number stopped: BCS={num_stopped_BCS}, and Kcusum={num_stopped_Kcusum}")
    if num_stopped_BCS==0: 
        print(f"BCS Detector detected 0 changes in {num_trials} trials")
        delay_BCS = N
    else: 
        delay_BCS /= num_stopped_BCS
    if num_stopped_Kcusum==0: 
        print(f"Kcusum Detector detected 0 changes in {num_trials} trials")
        delay_Kcusum = N
    else: 
        delay_Kcusum /= num_stopped_Kcusum
    # return the values 
    return delay_BCS, delay_Kcusum

if __name__=='__main__':
    N = 500 
    nc = 100 
    num_trials = 20 
    alpha = 0.002

    # Source = Higgs_Source
    Source = Occupancy_Source
    # Source = Banknote_Source

    #### To calibrate the two schemes (BCS and kernelCUSUM) to have  
    #### an ARL of around 500, we need to shrink the width of the 
    #### MR CS by a factor of 0.1 to 0.2; Similarly, for kernelCusum 
    #### we can manually select the rejection threshold. Some good 
    #### values are in the range 10 to 20 in experiments. 
    delay_BCS, delay_Kcusum = compare_delay(Source, N, nc,  
                                            num_trials, d=10, seed=None,
                                            alpha=alpha,
                                            learn_threshold=False,
                                            th_default=12, factor=0.1) 




    print(f"BCS {delay_BCS}, Kcusum {delay_Kcusum}")