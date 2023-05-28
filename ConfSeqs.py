import numpy as np 
from utils import * 
from math import sqrt, log, log2, floor, ceil, pi
from functools import partial

#===============================================================================
#===============================================================================
def get_backward_CS(data, CS_func, CS_kwargs, alpha=0.001, initial=5, initial_min=-20, 
                        initial_max=20, two_sample_data=False):
    """
        Construct a backward CS using a forward CS function 
    """
    ## some preprocessing 
    if initial_min is None: 
        initial_min = -float('inf')
    if initial_max is None:
        initial_min = float('inf')

    if two_sample_data:
        XX, YY = data 
        XX_, YY_ = XX[::-1], YY[::-1]
        # get the CS for reversed observations 
        res = CS_func(XX_, YY_, alpha=alpha, **CS_kwargs)
        L_, U_ = res
    else:
        XX = data 
        XX_ = XX[::-1]
        # get the CS for reversed observations 
        L_, U_ = CS_func(XX_, alpha=alpha, **CS_kwargs)

    initial = min(initial, len(XX))
    # discard the initial vlaues  of L_ and U_
    L_[:initial], U_[:initial] = initial_min, initial_max
    # Reverse the CS again to get backward CS
    L, U = L_[::-1], U_[::-1]
    return L, U 

def get_2sample_backward_CS(XX, YY, CS_func, CS_kwargs, alpha=0.001,
                            initial=5, initial_min=-20, initial_max=20):
    """
        Construct a two-sample backward CS using a forward CS function 
    """
    ## some preprocessing 
    if initial_min is None: 
        initial_min = -float('inf')
    if initial_max is None:
        initial_min = float('inf')
    initial = min(initial, len(XX))
    # reverse the observations 
    XX_ = XX[::-1]
    YY_ = YY[::-1]
    # get the CS for reversed observations 
    L_, U_ = CS_func(XX_, YY_, alpha=alpha, **CS_kwargs)
    # discard the initial vlaues  of L_ and U_
    L_[:initial], U_[:initial] = initial_min, initial_max
    # Reverse the CS again to get backward CS
    L, U = L_[::-1], U_[::-1]
    return L, U 

#===============================================================================
# Empirical Bernstein CS for Bounded Random variables
#===============================================================================
def get_Psi_E(a):
    assert 0<=a<1 
    return (-log(1-a) - a)/4

def get_mu_hat(XX):
    n = len(XX)
    mu_hat = (np.cumsum(XX) + 0.5)/np.arange(2, n+2)
    return mu_hat

def get_sigma_hat(XX):
    n = len(XX)
    mu_hat = get_mu_hat(XX)
    return (np.cumsum( (XX-mu_hat)**2) + 0.25 )/np.arange(2, n+2)

def get_v(XX):
    M_ = get_mu_hat(XX)
    M = np.zeros(M_.shape)
    M[0] = 0.5
    M[1:] = M_[:-1]
    # compute the v values 
    v = 4*((XX-M)**2)
    return v 

def get_lambda(XX, alpha=0.05, c=0.5):
    n = len(XX)
    S = get_sigma_hat(XX)
    T = np.arange(1, n+1)
    Lambda = np.sqrt( (2*log(2/alpha))/(S*T*np.log(1+T)) )
    # clip the values below c 
    Lambda = np.clip(a=Lambda, a_min=0, a_max=c) 
    return Lambda 

def get_weighted_mean(XX, alpha=0.05, Lambda=None, c=0.5): 
    if Lambda is None:
        Lambda = get_lambda(XX, alpha=alpha, c=c)
    L = np.cumsum(Lambda)        
    LX = Lambda * XX 
    return np.cumsum(LX)/L


def get_Bounded_EB_CS(XX, Lambda=None, v=None, alpha=0.05, c=0.5, initial=5, 
                        initial_min=0, initial_max=1):
    """
        Construct an Empirical Bernstein (EB) confidence sequence 
        for bounded random variables
    """
    ## some preprocessing 
    if initial_min is None: 
        initial_min = -float('inf')
    if initial_max is None:
        initial_min = float('inf')
    initial = min(initial, len(XX))

    if Lambda is None:
        Lambda = get_lambda(XX, alpha=alpha, c=c) 
    if v is None:
        v = get_v(XX)
    # get the \Psi_E values 
    Psi_E = np.fromiter(map(get_Psi_E, Lambda), dtype=float)
    # get the weighted mean 
    wt_mean = get_weighted_mean(XX, alpha=alpha, Lambda=Lambda, c=c)
    # get the v values 
    C = (log(2/alpha) + np.cumsum(v*Psi_E))/np.cumsum(Lambda) 

    L, U = wt_mean - C, wt_mean + C
    ### 
    L[:initial], U[:initial] = initial_min, initial_max
    return L, U 


def get_backward_Bounded_EB_CS(XX, Lambda=None, v=None, alpha=0.05, c=0.5):
    # reverse the observations 
    XX_ = XX[::-1]
    # get the CS for the reversed observations 
    L, U = get_Bounded_EB_CS(XX_, Lambda=Lambda, v=v, alpha=alpha, c=c)
    # Reverse the confidence sequence 
    L, U = L[::-1], U[::-1]
    return L, U


#===============================================================================
## GAUSSIAN CS based on Howard-Ramdas 2021
#===============================================================================
def get_Gaussian_CS(XX, alpha=0.05, intersect=False, initial=5, 
                        initial_min=-20, initial_max=20):
    ## some preprocessing 
    if initial_min is None: 
        initial_min = -float('inf')
    if initial_max is None:
        initial_min = float('inf')
    initial = min(initial, len(XX))

    NN = np.arange(1, len(XX)+1)
    mean = np.cumsum(XX)/NN 
    temp = (np.log(np.log(2*NN)) + 0.72*log(10.4/alpha))/NN 
    w = 1.7*np.sqrt(temp) 
    L = mean - w 
    U = mean + w 
    if intersect:
        lmax, umin = L[0], U[0]
        for i, (l, u) in enumerate(zip(L, U)):
            lmax =  max(lmax, l)
            L[i] = lmax 
            umin = min(umin, u)
            U[i] = umin 
    # 
    L[:initial], U[:initial] = initial_min, initial_max
    return L, U 

def get_backward_Gaussian_CS(XX, alpha=0.05, intersect=False):
    # reverse the observations 
    XX_ = XX[::-1]
    # get the CS for the reversed observations 
    L, U = get_Gaussian_CS(XX_, alpha=alpha, intersect=intersect)

    # Reverse the confidence sequence 
    L, U = L[::-1], U[::-1]
    return L, U

#===============================================================================
#===============================================================================
## MMD Utils 
def computeMMD(XX, YY, kernel_func=None, unbiased=True): 
    """
        Compute the quadratic time kernel-MMD statistic 
        between the samples XX and YY of same size, using 
        the kernel defined by the function handle 'kernel_func' 
    """
    n, d = XX.shape 
    n_, d_ = YY.shape 
    assert (n_==n) and (d_==d)
    if n<=1:
        return 0 
    if kernel_func is None:
        kernel_func = partial(GaussianKernel, bw=sqrt(d)) 
    
    KX = kernel_func(XX)
    KY = kernel_func(YY)
    KXY = kernel_func(XX, YY)
    KYX = np.transpose(KXY) 

    H = KX + KY - KXY - KYX
    if unbiased:
        temp = np.sum(H) - np.trace(H) 
        mmd = temp / (n*(n-1))
    else:
        mmd = H.mean()
    return mmd 

def computeMMD2(XX, YY, kernel_func=None, unbiased=True): 
    """
        Compute the quadratic time kernel-MMD statistic 
        between the samples XX and YY of same size, using 
        the kernel defined by the function handle 'kernel_func' 
    """
    n, d = XX.shape 
    n_, d_ = YY.shape 
    assert (n_==n) and (d_==d)
    if n<=1:
        return 0 
    if kernel_func is None:
        kernel_func = partial(GaussianKernel, bw=sqrt(d)) 
    #########
    KX = kernel_func(XX)
    KY = kernel_func(YY)
    KXY = kernel_func(XX, YY)
    KYX = np.transpose(KXY) 
    #########
    H = KX + KY - KXY - KYX
    mmd = 0 
    MMDvals = np.zeros((n,))
    for i in range(1, n):
        ### update mmd 
        mmd = mmd + np.sum(H[i, :i+1]) + np.sum(H[:i+1, i]) - H[i, i]
        if unbiased:
            mmd -= H[i,i]
        MMDvals[i] = mmd 
    # normalize the MMD values appropriately
    nn = np.arange(1, n+1)
    if unbiased:
        normalize_ = nn*(nn-1)
        normalize_[0] =  1 # to prevend division by zero 
    else:
        normalize_ = nn*nn
    return MMDvals/normalize_

#===============================================================================
#===============================================================================
###Reverse-submartingale kernel-MMD CS of Manole-Ramdas 
def get_gamma_t(t, alpha=0.01):
    """
        the term gamma_t defined in Corollary 14 of Manole-Ramdas~(2021)  
    """
    assert t>=1
    pi_ = np.pi
    term1 = 2*sqrt(2/t) 
    term2_1 = log( log( (pi_*pi_*(max(1, log2(t)))**2)/6 ) )     
    term2_2 = log(2/alpha) 
    term2 = 4*sqrt((2/t)* (term2_1 + term2_2))
    gamma_t = term1 + term2 
    return gamma_t 

def get_gamma_t2(n, alpha=0.01):
    """
        the term gamma_t defined in Corollary 14 of Manole-Ramdas~(2021)  
    """
    assert n>=1
    nn = np.arange(1, n+1)
    pi_ = np.pi
    term1 = 2*np.sqrt(2/nn) #2*sqrt(2/t) 
    term2_1 = np.log( np.log( (pi_*pi_*(np.maximum(1, np.log2(nn)))**2)/6 ) )     
    term2_2 = log(2/alpha) 
    term2 = 4*np.sqrt((2/nn)* (term2_1 + term2_2))
    gamma_t = term1 + term2 
    return gamma_t 


def get_kappa_t(t, alpha=0.01):
    """
        the term kappa_t defined in Eq.~(23) of Manole-Ramdas~(2021) 
    """
    pi_ = np.pi
    term1 = log( log( (pi_*pi_*(max(1, log2(t)))**2)/6 ) )     
    term2 = log(4/alpha) 
    kappa_t = sqrt( (term1 + term2)/t) 
    return kappa_t

def get_kappa_t2(n, alpha=0.01):
    """
        the term kappa_t defined in Eq.~(23) of Manole-Ramdas~(2021) 
    """
    assert n>=1 
    nn = np.arange(1, n+1)
    pi_ = np.pi
    term1 = np.log( np.log( (pi_*pi_*(np.maximum(1, np.log2(nn)))**2)/6 ) )     
    term2 = log(4/alpha) 
    kappa_t = np.sqrt( (term1 + term2)/nn) 
    return kappa_t

def get_MMD_CS_MR(XX, YY, alpha=0.05,  initial=5, 
                        initial_min=0, initial_max=10, factor=0.2):
    """
        Construct an CS for MMD distance based on reverse submartingales 
        derived by Manole-Ramdas (2021) arxiv: 2103:09267
    """
    ## some preprocessing 
    if initial_min is None: 
        initial_min = -float('inf')
    if initial_max is None:
        initial_min = float('inf')
    initial = min(initial, len(XX))

    # Get the MMD distances 
    n, d = XX.shape 
    kernel_func = partial(GaussianKernel, bw=sqrt(d))
    MMD = computeMMD2(XX, YY, unbiased=False) 
    MMD = np.sqrt(np.maximum(0, MMD))
    # Get the boundaries 
    Gamma = get_gamma_t2(n, alpha) 
    Kappa = get_gamma_t2(n, alpha)
    # Get the lower and uppper CS 
    ## the term factor is used to shrink the size of CS, 
    ## because in practice MR CS can often be quite conservative
    L = MMD - factor*Gamma 
    U = MMD + factor*Kappa 
    # Some postprocessing 
    L[:initial] = initial_min
    L = np.maximum(L, 0) #MMD metric is non-negative 
    U[:initial] = initial_max 

    return L, U 

#===============================================================================
#===============================================================================   
###Empirical Bernstein Linear-MMD CS of Balsubramani-Ramdas 
### the 'practical' version, stated in Eq.~(7) 
def get_block_mmd_vals(XX, YY, kernel_func=None, b=2): 
    n, d = XX.shape 
    if n<b:
        return np.zeros((n,)) 
    if kernel_func is None: 
        kernel_func = partial(GaussianKernel, bw=sqrt(d))
    # compute the block mmd values 
    H_unique = np.zeros((n//b, ))
    H = np.zeros((b,))
    for i in range(n//b):
        hval =  computeMMD(XX=XX[b*i:b*(i+1)], YY=YY[b*i:b*(i+1)])
        H_unique[i] = hval 
        H[i*b:(i+1)*b] = hval
    # some postprocessing 
    if n%b: 
        # copy the last computed MMD value in the remainng places 
        H[-(n%b):] = H_unique[-1] 
    return H_unique, H

def get_BR_boundary(n, H_unique, alpha=0.01, b=2):
    V_unique = np.cumsum(H_unique*H_unique)
    logV = np.maximum(1, np.log(V_unique))
    B_unique = log(1/alpha) + np.sqrt(2*V_unique*np.log(logV/alpha))
    B = np.zeros((n,))
    for i in range(n//b):
        B[i*b:(i+1)*b] = B_unique[i]/(i+1) 
    r = n%b 
    if r: 
        B[-r:] = B_unique[-1]
    return B 

def get_MMD_CS_BR(XX, YY, alpha=0.05, b=2, initial=5, 
                        initial_min=0, initial_max=10, 
                        factor=0.1):
    """
        Construct an CS for MMD distance based on reverse submartingales 
        derived by Balsubramani-Ramdas (2015)
    """
    ## some preprocessing 
    if initial_min is None: 
        initial_min = -float('inf')
    if initial_max is None:
        initial_min = float('inf')
    initial = min(initial, len(XX))

    # Get the MMD distances 
    n, d = XX.shape 
    if n<b: 
        L, U = np.zeros((n,)), np.zeros((n,))
        return L, U
    kernel_func = partial(GaussianKernel, bw=sqrt(d))
    H_unique, H = get_block_mmd_vals(XX, YY, kernel_func, b=b)
    H_sum = np.cumsum(H_unique)
    MMD = np.zeros((n,))
    for i in range(n//b):
        MMD[i*b:(i+1)*b] = H_sum[i]/(i+1)
    if n%b:
        MMD[-(n%b):] = H_sum[-1] / (n//b) 
    # Get the boundaries 
    B = get_BR_boundary(n, H_unique, alpha, b)
    # Get the lower and uppper CS 
    L = np.maximum(MMD - B, 0)
    U = MMD + B
    # Some postprocessing 
    L[:initial] = initial_min
    U[:initial] = initial_max 
    # a heuristic reduction in the width of the CS!!!
    L, U = factor*L, factor*U
    return L, U 


#===============================================================================
#===============================================================================
###TODO: supermartingale quantile-CS of Howard-Ramdas 
def get_Q(XX, p):
    """
    return \sup \{x: \hat{F}(x) \leq p \}
    """

    XX_ = np.sort(XX)
    n = XX.size
    assert 0<p<1
    idx = floor(n*p)  
    q_val = XX_[idx]
    return q_val 


def get_Q_minus(XX, p): 
    """
    return \sup\{x: \hat{F}(x) < p \}
    """
    XX_ = np.sort(XX)
    n = XX_.size
    idx = ceil( n*p -1) 
    q_minus_val = XX_[idx]
    return q_minus_val


def get_ft_func(N, p, alpha):
    """
    ft = 1.5*\sqrt{p(1-p)*ell(t)} + 0.8*\ell(t), 
    where \ell(t) = (1/t)*(1.4*log(log(2.1t)) + log(10/alpha)) 
    Eq. (9) in Howard-Ramdas
    """
    NN = np.arange(1, N+1)
    ell_vals = (1.4*np.log(np.log(2.1*NN)) + log(10/alpha))/ NN
    ft = 1.5*np.sqrt(p*(1-p)*ell_vals) + 0.8*ell_vals
    return ft

def get_Quantile_CS_HR(XX, p=0.5,
                        alpha=0.01,
                        initial=20,
                        initial_min=-10,
                        initial_max=10, 
                        tol=1e-5):
    ## some preprocessing 
    if initial_min is None: 
        initial_min = -float('inf')
    if initial_max is None:
        initial_min = float('inf')
    initial = min(initial, len(XX))

    N = len(XX) 
    # Get the f_t values 
    ft = get_ft_func(N=N, p=p, alpha=alpha) 
    
    # upper and lower "CS" in the p-space 
    # force to keep it in the range [tol, 1-tol] \subset (0,1)
    lower_p = np.maximum(tol, p - ft) 
    upper_p = np.minimum(1-tol, p + ft)
    # Transform from p-space to the Real line 
    U = np.array([
        get_Q_minus(XX[:t+1], p_) for t, p_ in enumerate(upper_p)
    ])
    L = np.array([
        get_Q(XX[:t+1], _p) for t, _p in enumerate(lower_p)
    ])
    # Some postprocessing 
    L[:initial] = initial_min
    U[:initial] = initial_max
    return L, U 

#===============================================================================
#===============================================================================
###TODO: Changepoint detection by Podkopoev-Ramdas

#===============================================================================
#===============================================================================
if __name__=='__main__':
    n = 5000
    nc = 1000
    Source = Higgs_Source
    XX, YY = Source(N=n, nc=nc)
    # XX = np.random.randn(n) + 10
    # YY = np.random.randn(n)
    # YY[n//2:] = 1000 + 10*np.random.random((n//2,))

    # XX = XX.reshape((-1,1))
    # YY = YY.reshape((-1,1))
    alpha=0.002
    L, U = get_MMD_CS_BR(XX, YY, alpha=0.01, factor=0.1) 
    ## L, U = get_MMD_CS_MR(XX, YY, alpha=alpha)
    # L, U = get_Quantile_CS_HR(XX, p=0.4)
    # # L, U = get_Gaussian_CS(XX, alpha=alpha)
    L_, U_ = get_2sample_backward_CS(XX, YY, get_MMD_CS_BR, {'factor':0.1} ,alpha=alpha)
    NN = np.arange(1, n+1)
    plt.plot(NN, L,'b')
    plt.plot(NN, U,'b')
    plt.plot(NN, L_,'r')
    plt.plot(NN, U_,'r')
    plt.ylim([-0.1, 0.1])
    # plt.show()


    # X1 = Univariate_Gaussian_Source(N=n)
    # X2 = Univariate_Gaussian_Source(N=n, mean0=10) 
    # X1 = X1.reshape((-1,1))
    # X2 = X2.reshape((-1,1))

    # print(computeMMD(X1, X2))


