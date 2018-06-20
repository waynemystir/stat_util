import scipy.stats as stats
import numpy as np
from numpy import power as pw
import random

flip=lambda p=1/2: random.random()<p

flipn=lambda p=1/2,n=10: [flip(p) for i in range(n)]

sample_mean=lambda data: sum(data)/len(data)

def sample_var(data, ht=None, opt='unbiased'):
    # data: data
    # ht: sample mean (hat theta)
    # opt: type of sv
    n=len(data)
    if ht is None:
        ht = sample_mean(data)
    if opt=='unbiased':
        return 1/(n-1)*sum(pw(data[i]-ht,2) for i in range(n))
    if opt=='biased':
        return 1/(n)*sum(pw(data[i]-ht,2) for i in range(n))
    if opt=='bernoulli-1':
        return ht*(1-ht)
    if opt=='bernoulli-2':
        return 1/4

# standard normal cdf
phi=lambda x: stats.norm(0,1).cdf(x)

# alternative to stats.norm.ppf(q=q,loc=ht,scale=se)
def find_z(x, mu=0, var=1, tol=1e-8):
    err,z,cnt=9.9,1.0,0
    while abs(err)>tol:
        cnt+=1
        err=x-stats.norm(mu,var).cdf(z)
        z+=err
    return (z,cnt,err)

'''
In [1271]: find_z(.975)
Out[1271]: (1.9599638252009011, 246, 9.8906491885486503e-09)

In [1272]: stats.norm.ppf(.975)
Out[1272]: 1.959963984540054

In [1273]: stats.norm.ppf(.975,loc=17,scale=3)
Out[1273]: 22.879891953620163

In [1274]: find_z(.975,mu=17,var=3)
Out[1274]: (22.879891457848469, 782, 9.8503739609512309e-09)
'''

def confidence_interval(data=None, ht=None, hv=None, n=None, cl=0.95, nort='n', svopt='unbiased', prnt=False):
    # data is list of numpy array
    # if you supply data parameter, then ht, hv, and n are overridden if you provide those as well
    # ht: sample mean (hat theta)
    # hv: variance or sample variance
    # n: number of samples
    # cl: confidence level
    # nort: 'n' is normal dist, 't' is t-dist
    # svopt: type of sample variance

    if data is None and (ht is None or hv is None or n is None):
        print("Unsuccessful. Please provide data or sample mean and variance.")
        return None

    if data is not None:
        n=len(data)
        ht=sample_mean(data)
        hv=sample_var(data,ht=ht,opt=svopt)

    nortb=nort=='t'
    norts='t' if nort=='t' else 'Normal'

    # quantile: defaults to (1+.95)/2=0.975
    q=(1+cl)/2

    # se: standard error of the sample mean ht
    # https://en.wikipedia.org/wiki/Variance#Sum_of_uncorrelated_variables_(Bienaym%C3%A9_formula)
    # https://en.wikipedia.org/wiki/Standard_error
    se=np.sqrt(hv/n)

    # we can compute the confidence interval in two ways:
    # normalize after the ppf (percent point function):
    z1=stats.t.ppf(q=q,df=n-1) if nortb else stats.norm.ppf(q=q)
    ci1=(ht-z1*se,ht+z1*se)

    # or we can normalize inside the ppf:
    if se!=0:
        z2=stats.t.ppf(q=q,df=n-1,loc=ht,scale=se) if nortb else stats.norm.ppf(q=q,loc=ht,scale=se)
        ci2=(2*ht-z2,z2)

    if prnt:
        print("ci: nort={} n={} ht={} hv={} se={} z1={} z2={}\nci1={}\nci2={}"
            .format(norts, n, ht, hv, round(se,3), round(z1,3), round(z2,3), ci1, ci2))

    return (ci1,ht,hv)

if __name__ == "__main__":
    confidence_interval()
