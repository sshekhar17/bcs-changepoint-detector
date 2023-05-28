import os 
from functools import partial 
from math import pi, cos, sin 
from math import sqrt, log
import numpy as np 
from scipy.stats import t as tdist 
from scipy.stats import chi2
import scipy.stats as stats 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.spatial.distance import cdist, pdist 
import tikzplotlib as tpl 
import pandas as pd 
from sklearn.utils import shuffle

def clip_CSs(L, U, Lb, Ub, amin=0, amax=1):
    """
    clip given confidence sequences to lie in the 
    range [amin, amax] 
    """
    L= np.clip(L, a_min=amin, a_max=amax)
    U = np.clip(U, a_min=amin, a_max=amax)
    Lb = np.clip(Lb, a_min=amin, a_max=amax)
    Ub = np.clip(Ub, a_min=amin, a_max=amax)
    return L, U, Lb, Ub


def Bernoulli_Source(N=500, p0=0.3, p1=0.5, nc=None):
    """
    Return a binary stream of size (N,):
        first nc points are iid Bernoulli(p0) 
        remaining points are iid Bernoulli(p1) 
    """
    # nc = change point 
    nc = N if nc is None else nc 
    assert nc<=N
    X = np.random.random((N,))
    X[:nc] = (X[:nc]<=p0)*1.0  
    X[nc:] = (X[nc:]<=p1)*1.0
    return X 

def BoundedSource(N=500, p0=0.5, p1=0.6, nc=None):
    nc = N if nc is None else nc 
    assert nc<=N 
    # generate the observations
    U, V = np.random.random((N,)), np.random.random((N,))
    p = np.concatenate(
        (p0*np.ones((nc,)),  p1*np.ones((N-nc,)))
    )
    X = p*(p + (1-p)*U) + (1-p)*p*V 
    return X 
        

def Univariate_Gaussian_Source(N=500, mean0=0.0, std0=1.0, mean1=2.0, std1=1.0,
        nc=None, mean_vec=None, std_vec=None):
    if (mean_vec is None) or (std_vec is None):
        # nc = change point 
        nc = N if nc is None else nc 
        assert nc<=N
        # default values 
        X = np.random.randn(N)
        X[:nc] = X[:nc]*std0 + mean0
        X[nc:] = X[nc:]*std1 + mean1
    else:
        assert len(std_vec)==len(mean_vec)==N
        X = np.random.randn(N)*std_vec + mean_vec
    return X 


def Multivariate_Gaussian_Source(N=500, mean0=np.zeros((5,)), cov0=np.eye(5), 
                                    mean1= np.ones((5,)), cov1= np.eye(5), nc=None):
    # nc = change point 
    nc = N if nc is None else nc 
    assert nc<=N
    # default values 
    X1 = np.random.multivariate_normal(mean=mean0, cov=cov0, size=nc)
    if N-nc>0:
        X2 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=N-nc)
        X = np.vstack((X1, X2))
    else: 
        X = X1
    return X 


def Univariate_T_Source(N=500, df1=1.0, loc1=0.0, scale1=1.0, 
                        df2=2.0, loc2=1.0, scale2=1.0, nc=None):

    XX = tdist.rvs(df=df1, loc=loc1, scale=scale1, size=N)
    if nc is None:
        return XX 
    else:
        if N-nc<=0:
            return XX 
        else:
            XX[nc:] = tdist.rvs(df=df2, loc=loc2, scale=scale2, size=N-nc)
    return XX 


def TwoSample_Source(N, 
                        Source1, Source1_kwargs,
                        Source2, Source2_kwargs,
                        Source3, Source3_kwargs,
                        nc=None, two_dim_output=True):

    XX = Source1(N=N, **Source1_kwargs)
    YY = Source2(N=N, **Source2_kwargs)
    if nc is not None: 
        YY_ = Source3(N=N-nc, **Source3_kwargs)
        YY[nc:] = YY_
    if two_dim_output:
        if len(XX.shape)==1:
            XX = XX.reshape((-1,1))
        if len(YY.shape)==1:
            YY = YY.reshape((-1,1))
    return XX, YY


###########################################################################
### Define the source of data 
def ClassificationData(N=500, mu=np.array([1, 0]), radius=1.5, 
    theta=pi/4, nc=250):
    if nc is None: 
        nc = N 
    assert nc <= N 
    # get the pre-change observations 
    L1 = (np.random.random((nc,)) >= 0.5)*1 
    mu = radius*mu
    X1 = stats.multivariate_normal(mean=mu, cov=np.eye(2)).rvs(size=nc) 
    temp = (2*L1-1).reshape((-1,1))
    temp = np.concatenate((temp, temp), axis=1)
    X1 = X1 * temp
    # get the post-change distribution 
    if N-nc>0:
        L2 = (np.random.random((N-nc,)) >= 0.5)*1 
        mu2 = radius*np.array([cos(theta), sin(theta)])
        X2 = stats.multivariate_normal(mean=mu2, cov=np.eye(2)).rvs(size=N-nc) 
        temp =(2*L2-1).reshape((-1,1))
        temp = np.concatenate((temp, temp), axis=1)
        X2 = X2 * temp
        X = np.vstack((X1, X2)) 
        L = np.concatenate((L1, L2))
    else: 
        X = X1 
        L = L1 
    return X, L

###########################################################################
# definition of a linear classifier 
def linear_classifier(X, w = np.array([1,0])):
    predicted = (np.sum(X*w, axis=1)>0)*1.0 
    return predicted 

### 
def ClassifierOutput(N=500, mu=np.array([1, 0]), radius=1.5, 
    theta=pi/4, nc=250, w= np.array([1, 0])):
    X, L = ClassificationData(N=N, mu=mu, radius=radius, theta=theta, nc=nc)
    Lpred = linear_classifier(X=X, w=w) 
    Y = (L!=Lpred)*1.0
    return Y
###########################################################################


def check_interval_intersection(l1, u1, l2, u2):
    """
        Checks if [l1, u1]  and [l2, u2] intersect or not
    """
    # some sanity checks
    if l1>u1:
        raise Exception(f'l1={l1:.2f}, and u1={u1:.2f}!!')
    if l2>u2:
        raise Exception(f'l2={l2:.2f}, and u2={u2:.2f}!!')
    # 
    if l1>u2 or l2>u1:
        # no intersection 
        return False 
    else:
        # the two intervals [l1, u1] and [12, u2] intersect
        return True 

def check_disjoint_univariate(CS, tmax):
    """
        Checks for consistency of given CS up to time tmax

        parameters:
            CS      :ndarray    Univariate Confidence Sequences
            tmax    :int        time index 
        
        returns:
            stopped         :bool   True if CS is inconsistent before tmax
            stopping_time   :int    time at which CS is first inconsistent
                                        if not stopped, then tmax = len(CS)
    """
    L, U = CS 
    n = len(L) 
    # sanity checks
    assert n==len(U)
    assert tmax<=n
    # 
    l_max, u_min = -float('inf'), float('inf')
    stopped=False
    stopping_time=tmax
    for t in range(tmax):
        l, u = L[t], U[t]
        if not check_interval_intersection(
                l1=l_max, u1=u_min, l2=l, u2=u
            ):
            stopped=True 
            stopping_time = t+1
            break 
        # update l_max and u_min 
        l_max = max(l_max, l)
        u_min = min(u_min, u)
        if l_max>u_min:
            stopped=True
            stopping_time=t+1
    return stopped, stopping_time 


def check_disjoint_univariate_two_CS(fwdCS, backCS, t, verbose=False):
    """
        Check whether fwdCS and backCS are disjoint at time t 
    """
    L, U = fwdCS 
    L, U = L[:t], U[:t]
    L_, U_ = backCS 
    # sanity check 
    assert len(L_)==t and len(U_)==t
    l_max, l_max_ = -np.float('inf'), -np.float('inf')
    u_min, u_min_ = np.float('inf'), np.float('inf')
    stopped = False 
    for i in range(t):
        l, u, l_, u_ = L[i], U[i], L_[i], U_[i] 
        # update the running intersections 
        l_max = max(l_max, l)
        l_max_ = max(l_max_, l_)
        u_min = min(u_min, u)
        u_min_ = min(u_min_, u_)
        # check for non-intersections 
        if l_max>u_min: 
            stopped = True 
            if verbose:
                print(f'Rejecting: l_max={l_max:.2f}>u_min={u_min:.2f}')
            return stopped 
        elif  l_max_>u_min_:
            stopped = True 
            if verbose:
                print(f'Rejecting: l_max_={l_max_:.2f}>u_min_={u_min_:.2f}')
            return stopped 

        elif not check_interval_intersection(l_max, u_min, l_max_, u_min_):
            # finally check if the running intersections of the 
            # two CSs intersect or not 
            if verbose:
                print('Rejecting because the intersections do not intersect')
            stopped = True 
            return stopped 

    return stopped 

def check_disjoint_univariate_two_CS2(fwdCS, backCS, t, verbose=False):
    """
        Check whether fwdCS and backCS are disjoint at time t 
    """
    L, U = fwdCS 
    L, U = L[:t], U[:t]
    L_, U_ = backCS 
    ### get the intersected versions of the CS 
    L, U = get_intersected_CS(L, U) 
    L_, U_ = get_intersected_CS(L_, U_, reversed=True)
    # now check if these two CSs ever are disjoint 
    l, u = np.max(L), np.min(U)
    l_, u_ = np.max(L_), np.min(U_)
    return ( (l>u_) or (l_>u) )


def get_change_time_and_magnitude(fwdCS, backCS, verbose=False, use_intersected_CS=True):
    L, U = fwdCS 
    Lb, Ub = backCS
    if use_intersected_CS:
        L, U = get_intersected_CS(L, U)
        Lb, Ub = get_intersected_CS(Lb, Ub, reversed=True)
    t=len(Lb)
    # get the points of intersection 
    intersections = np.array([
                            check_intersect(L[i], U[i], Lb[i], Ub[i]) for i in range(t)
        ])
    # check if the CS's are disjoint at the stopping time
    disjoint_cs=False
    # store the instances at which the CSs (at time t+1) are disjoint
    I0 = [] 
    change_point = None 
    change_magnitude = 0
    max_CS_dist = 0
    for i, intersect_i in enumerate(intersections):
        if not intersect_i:
            disjoint_cs=True
            I0.append(i+1)
            # compute the point at which the difference b/w CSs is the largest 
            temp = max(Lb[i]-U[i], L[i]-Ub[i])
            if temp>=max_CS_dist:
                # returns the rightmost point of argmax
                change_point = i+1 
            # update the estimate of the change magnitude 
            temp_ = 0.5*abs(Lb[i]+Ub[i] - L[i]-U[i])
            change_magnitude = max(change_magnitude, temp_) 
    # convert to np-array 
    if disjoint_cs:
        I0 = np.array(I0)
    
    if verbose:
        print('\n')
        print(f'Changepoint of magnitude {change_magnitude:.2f} at T={change_point}') 
        print('\n')
    return change_point, change_magnitude, I0


def get_change_time_and_magnitude2(fwdCS, backCS, verbose=False, use_intersected_CS=True):
    L, U = fwdCS 
    Lb, Ub = backCS
    if use_intersected_CS:
        L, U = get_intersected_CS(L, U)
        Lb, Ub = get_intersected_CS(Lb, Ub, reversed=True)
    t=len(Lb)
    
    I0 = []
    max_CS_dist = 0 
    change_magnitude = 0 
    change_point = None
    for i in range(t):
        # 
        l, u, lb, ub = L[i], U[i], Lb[i], Ub[i]
        # check if there is no intersection: 
        if l>ub or lb>u: # no intersect
            I0.append(i) 
            # update the changepoint estimate if needed
            d_ = max(l-ub, lb-u)  
            if d_ >= max_CS_dist: 
                change_point = i+1 
                max_CS_dist = d_
            # update the changepoint magnitude if needed 
            eps_ = 0.5*abs( (u+l) - (ub+lb))
            change_magnitude =  max(eps_, change_magnitude)
    
    if verbose:
        if change_point is None:
            print('No changepoint detected!!')
        else: 
            print(f'Changepoint of magnitude {change_magnitude:.2f} at T={change_point}') 

    return change_point, change_magnitude, I0
            

####################################################################
def check_intersect(l, u, lb, ub):
    if (l<=lb<=u) or (lb <= l <= ub):
        return True 
    else:
        return False 


#===============================================================================
#===============================================================================
def get_intersected_CS(L, U, reversed=False):
    l_max = -float('inf')
    u_min = float('inf')
    if reversed:
        L_, U_ = L[::-1], U[::-1]
    else:
        L_, U_ = L, U
    for i, (l, u) in enumerate(zip(L_, U_)):
        l_max = max(l, l_max)
        u_min = min(u, u_min)
        L_[i] = min(l_max, u_min)
        U_[i] = max(u_min, l_max) 
    if reversed:
        L_, U_ = L_[::-1], U_[::-1]
    return L_, U_ 


def plot_CPD_results1(data, Result, mean,save_fig=False, 
                        figname=None, nc=100, 
                        ymin=-1, ymax=1, title=None):
    fwdCS = Result['forwardCS']
    backCS = Result['backCS']
    I0 = Result['disjoint_interval']
    L, U = fwdCS
    Lb, Ub = backCS

    N = len(L)
    NN = np.arange(1, N+1)
    T_=len(Lb)
    TT = np.arange(1, T_+1)

    T = Result['change_point']

    print(Lb.shape, L.shape)
    # get the intersected versions of the CS 
    L, U = get_intersected_CS(L, U)
    Lb, Ub = get_intersected_CS(Lb, Ub, reversed=True)
    palette = sns.color_palette(n_colors=10)
    plt.figure()
    if mean is not None:
        plt.plot(NN, mean, 'k', alpha=0.8)

    plt.plot(TT, Lb, color=palette[0])
    plt.plot(TT, Ub, color=palette[0])
    plt.fill_between(x=TT, y1=Lb, y2=Ub, alpha=0.3, color=palette[0])

    plt.plot(NN, L, color=palette[1])
    plt.plot(NN, U, color=palette[1])
    plt.fill_between(x=NN, y1=L, y2=U, alpha=0.3, color=palette[1])

    if len(I0):
        plt.axvspan(xmin=min(I0), xmax=max(I0), color='gray', alpha=0.3)
    plt.ylim([ymin,ymax])
    if title is not None: 
        plt.title(title, fontsize=15)
    if save_fig: 
        if figname is None: 
            figname = 'temp' 
        elif figname[-4:] == '.tex':
            figname = figname[:-4]
        
        plt.savefig(figname+'.png', dpi=450)
        tpl.save(figname+'.tex', axis_width=r'\figwidth', axis_height=r'\figheight')
    else:
        plt.show()


def plot_experiment_results1(nc, ChangePoints, Epsilon, Delay,  epsilon,
                                save_fig=False, base_figname=None):
    if save_fig:
        if base_figname is None:
            base_figname = './data/TempExperiment'

    # plot the distribution of estimated changepoints 
    plt.figure()
    plt.hist(ChangePoints, density=True, alpha=0.6) 
    plt.axvline(x=nc, color='k', linestyle='--') 
    plt.title('Distribution of Estimated Changepoint', fontsize=15)
    if save_fig: 
        figname = base_figname + 'EstimatedChangepoint'
        tpl.save(figname + '.tex', axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.savefig(figname+'.png', dpi=450)

    # Plot the distribution of estimated change magnitude
    plt.figure()
    plt.hist(Epsilon, density=True, alpha=0.6)
    plt.axvline(x=epsilon, color='k', linestyle='--') 
    plt.title('Distribution of Estimated Change Magnitude', fontsize=15)
    if save_fig: 
        figname = base_figname +'EstimatedChangeMagnitude'
        tpl.save(figname + ".tex", axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.savefig(figname+'.png', dpi=450)


    # plot the detection delay 
    plt.figure()
    plt.hist(Delay, density=True, alpha=0.6)
    plt.axvline(x=Delay.mean(),  linestyle='--') 
    plt.title('Distribution of Detection Delay', fontsize=15)
    if save_fig: 
        figname = base_figname + 'DetectionDelay'
        tpl.save(figname + ".tex", axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.savefig(figname+'.png', dpi=450)



def GaussianKernel(XX, YY=None, bw=1.0):
    # 
    YY = XX if YY is None else YY
    if len(XX.shape)==1: 
        XX = XX.reshape((1, -1))
    if len(YY.shape)==1: 
        YY = YY.reshape((1, -1))
    # print('Shape', XX.shape, YY.shape, '\n')
    dists = cdist(XX, YY)
    squared_dists = dists * dists 
    k = np.exp( -(1/(2*bw*bw)) * squared_dists ) 
    return k 


def get_new_seed(old_seed):
    if old_seed is None:
        return None 
    else: 
        return int(8217*old_seed % 10000)

def simple_binary_search(func, th_min, th_max, num_iters=50,
                            tol=1e-3, verbose=False):
    for _ in range(num_iters): 
        th = (th_max+th_min)/2 
        val = func(th)
        if val>0: 
            th_max = th 
        else: 
            th_min = th 
    th_ = (th_min+th_max)/2
    if verbose: 
        print(f'ending binary search  (th, val) = {th, func(th_)}')
    return th_ 
 
def Higgs_Source(N=1000, nc=200, two_dim_output=True):
    def helper(N__):
        skiprows = np.random.randint(0, 100000, size=1)
        df = pd.read_csv('./data/HIGGS.csv', sep=',', header=None, 
                        skiprows=skiprows, nrows=N__)
        data = df.to_numpy()
        labels = data[:, 0]
        features = data[:, 1:]
        idx0 = labels==0
        idx1 = labels==1
        features0 = shuffle(features[idx0])
        features1 = shuffle(features[idx1])
        
        return features0, features1
    
    while True: 
        f0, f1 = helper(10*N) 
        if len(f0)>=2*N and len(f1)>=N: 
            break 
    # create the data 
    if two_dim_output:
        XX = f0[:N] 
        if nc is None:
            YY = f0[N:2*N]
        else:
            YY = np.vstack((f0[N:N+nc], f1[:N-nc])) 
        # print(XX.shape, YY.shape)
        return XX, YY
    else:
        return YY

def Occupancy_Source(N=1000, nc=200, two_dim_output=True):
    df = pd.read_csv('./data/occupancy.txt', sep=',',skiprows=2,
                        header=None)

    data = df.to_numpy()
    labels, features = data[:, -1], data[:, :-1] 
    features = np.delete(features, 1, 1)
    features = features.astype(float)
    m, s = features.mean(axis=0, keepdims=True), features.std(axis=0, keepdims=True)
    features = (features-m)/s
    f0, f1 = features[labels==0], features[labels==1]
    # m0, s0 = f0.mean(axis=0, keepdims=True), f0.std(axis=0, keepdims=True)
    # m1, s1 = f1.mean(axis=0, keepdims=True), f1.std(axis=0, keepdims=True)
    # f0, f1 = (f0-m0)/s0, (f1-m1)/s1
    n0, n1 = len(f0), len(f1) 
    assert (2*N<=n0) and (N<=n1)
    # create the dataset 
    f0_, f1_ = shuffle(f0), shuffle(f1)
    if two_dim_output:
        XX = f0_[:N] 
        if nc is None:
            YY = f0_[N:2*N]
        else:
            YY = np.vstack((f0_[N:N+nc], f1_[:N-nc])) 
        return XX, YY
    else:
        return YY

def Banknote_Source(N=500, nc=100, two_dim_output=True):
    df = pd.read_csv('./data/data_banknote_authentication.txt', sep=',',skiprows=2,
                    header=None)
    assert N<=700
    data = df.to_numpy()
    labels, features = data[:, -1], data[:, :-1] 
    features = features.astype(float)
    # scale the features 
    m, s = features.mean(axis=0, keepdims=True), features.std(axis=0, keepdims=True)
    features = (features-m)/s
    f0, f1 = features[labels==0], features[labels==1]
    # create the dataset 
    if two_dim_output:
        XX = shuffle(f0)[:N] 
        if nc is None:
            YY = shuffle(f0)[:N]
        else:
            y1 = shuffle(f0)[:nc]
            y2 = shuffle(f1)[:N-nc]
            YY = np.vstack((y1, y2)) 
        return XX, YY
    else:
        return YY



def create_two_sample_source(d, delta=0.1):
    mean1 = np.zeros((d,))
    mean2 = np.ones((d,))*delta
    cov1 = np.eye(d)
    cov2 = np.eye(d)

    Source_1 = partial(
        Multivariate_Gaussian_Source, mean0=mean1, cov0=cov1, nc=None
    )

    Source_2 = partial(
        Multivariate_Gaussian_Source, mean0=mean1, cov0=cov1, nc=None
    )
    Source_3 = partial(
        Multivariate_Gaussian_Source, mean0=mean2, cov0=cov1, nc=None
    )
    Source = partial(TwoSample_Source,  
                                Source1=Source_1, Source1_kwargs={}, 
                                Source2=Source_2, Source2_kwargs={}, 
                                Source3=Source_3, Source3_kwargs={}, 
    )

    return Source 



if __name__=='__main__':
    # L_, U_ = np.random.random(100,), np.random.random(100,)
    # L, U = np.minimum(L_, U_), np.maximum(L_, U_)
    # stopped, t = check_disjoint_univariate([L,U], 50) 

    # plt.plot(np.arange(1, t+11), L[:t+10])
    # plt.plot(np.arange(1, t+11), U[:t+10])
    # plt.axvline(x=t, color='k')
    # plt.show()

    X = BoundedSource(N=1000, p0=0.5, p1=0.6, nc=250)
    plt.plot(np.arange(1, len(X)+1), X, '.')