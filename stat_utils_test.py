import stat_utils as su
import numpy as np
import random

def test_confidence_interval():
    print("********************** Test 1 n=100, Normal, ht=1/2 ********************")
    ht=1/2;hv=ht*(1-ht);su.confidence_interval(ht=ht,hv=hv,n=100,nort='n',cl=.95,prnt=True)
    print("************************************************************************\n")

    print("******************** Test 2 n=100, t-dist, ht=1/2 ********************")
    ht=1/2;hv=ht*(1-ht);su.confidence_interval(ht=ht,hv=hv,n=100,nort='t',cl=.95,prnt=True)
    print("************************************************************************\n")

    print("******************** Test 3 n=1000, Normal, ht=1/2 ********************")
    ht=1/2;hv=ht*(1-ht);su.confidence_interval(ht=ht,hv=hv,n=1000,nort='n',cl=.95,prnt=True)
    print("************************************************************************\n")

    print("******************** Test 4 n=1000, t-dist, ht=1/2 ********************")
    ht=1/2;hv=ht*(1-ht);su.confidence_interval(ht=ht,hv=hv,n=1000,nort='t',cl=.95,prnt=True)
    print("************************************************************************\n")

    print("********************** Test 5 n=100, Normal, ht=1/100 ********************")
    ht=1/100;hv=ht*(1-ht);su.confidence_interval(ht=ht,hv=hv,n=100,nort='n',cl=.95,prnt=True)
    print("************************************************************************\n")

    print("********************** Test 6 n=100, Normal, ht=1/10 ********************")
    ht=1/10;hv=ht*(1-ht);su.confidence_interval(ht=ht,hv=hv,n=100,nort='n',cl=.95,prnt=True)
    print("************************************************************************\n")

    print("********************** Test 7 n=100, Normal, ht=7/10 ********************")
    ht=7/10;hv=ht*(1-ht);su.confidence_interval(ht=ht,hv=hv,n=100,nort='n',cl=.95,prnt=True)
    print("************************************************************************\n")

    print("********************** Test 8 n=100, Normal, ht=9/10 ********************")
    ht=9/10;hv=ht*(1-ht);su.confidence_interval(ht=ht,hv=hv,n=100,nort='n',cl=.95,prnt=True)
    print("************************************************************************\n")

    print("********************** Test 9 n=100, Normal, ht=99/100 ********************")
    ht=99/100;hv=ht*(1-ht);su.confidence_interval(ht=ht,hv=hv,n=100,nort='n',cl=.95,prnt=True)
    print("************************************************************************\n")

    print("*\n*\n*\n*\n*\n")

    print("****** Test 10 conf_int_smoke_test n=10000, trials=1000, ht=1/2 ************")
    w,trs,s=conf_int_smoke_test(t=1/2,n=10000,trials=1000)
    print("Test 10 Smoke Test: w={} trials={} Perc Inside CI={}".format(w,trs,s))
    print("************************************************************************\n")

    print("****** Test 11 conf_int_smoke_test n=10000, trials=1000, ht=1/2 ************")
    w,trs,s=conf_int_smoke_test(t=1/2,n=10000,trials=1000)
    print("Test 11 Smoke Test: w={} trials={} Perc Inside CI={}".format(w,trs,s))
    print("************************************************************************\n")

    print("****** Test 12 conf_int_smoke_test n=10000, trials=1000, ht=1/2 ************")
    w,trs,s=conf_int_smoke_test(t=1/2,n=10000,trials=1000)
    print("Test 12 Smoke Test: w={} trials={} Perc Inside CI={}".format(w,trs,s))
    print("************************************************************************\n")

    print("****** Test 13 conf_int_smoke_test n=10000, trials=1000, ht=1/2 ************")
    w,trs,s=conf_int_smoke_test(t=1/2,n=10000,trials=1000)
    print("Test 13 Smoke Test: w={} trials={} Perc Inside CI={}".format(w,trs,s))
    print("************************************************************************\n")

    print("****** Test 14 conf_int_smoke_test n=10000, trials=1000, ht=1/2 ************")
    w,trs,s=conf_int_smoke_test(t=1/2,n=10000,trials=1000)
    print("Test 14 Smoke Test: w={} trials={} Perc Inside CI={}".format(w,trs,s))
    print("************************************************************************\n")

    print("*\n*\n*\n*\n*\n")

    print("****** Test 15 conf_int_test n=1000, trials=1000 ************")
    w,trs,s=conf_int_smoke_test(n=1000,trials=1000)
    print("Test 15: w={} trials={} Perc Inside CI={}".format(w,trs,s))
    print("************************************************************************\n")

    print("****** Test 16 conf_int_test n=1000, trials=1000 ************")
    w,trs,s=conf_int_smoke_test(n=1000,trials=1000)
    print("Test 16: w={} trials={} Perc Inside CI={}".format(w,trs,s))
    print("************************************************************************\n")

    print("****** Test 17 conf_int_test n=1000, trials=1000 ************")
    w,trs,s=conf_int_smoke_test(n=1000,trials=1000)
    print("Test 17: w={} trials={} Perc Inside CI={}".format(w,trs,s))
    print("************************************************************************\n")


def conf_int_smoke_test(t=1/2,n=10000,trials=100,pr=False,prci=True):
    w=0
    f0=su.flipn(p=t,n=n)
    c,ht,hv=su.confidence_interval(data=f0,nort=True)
    hs2=su.sample_var(f0,ht=ht,opt='bernoulli-1')
    if prci: print("ht={} hv={} hs2={} ci={} n={}".format(ht,hv,hs2,c,n))
    for i in range(trials):
        f=su.flipn(p=t,n=n)
        r=sum(f)/n
        if r>=c[0] and r<=c[1]: w+=1
        elif pr:
            print("no:f={} sf={} r={} n={} c0={} c1={}"
                .format(np.array(f).astype(int),sum(f),r,n,round(c[0],3),round(c[1],3)))
    return w,trials,w/trials

def conf_int_test(n=1000,trials=100):
    w=0
    for i in range(trials):
        p=random.random()
        f=su.flipn(p=p,n=n)
        c,ht,hv=su.confidence_interval(data=f,nort=True)
        if p>=c[0] and p<=c[1]: w+=1
    return w,trials,w/trials
