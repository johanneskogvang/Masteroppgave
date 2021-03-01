# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:37:34 2021

@author: Johan
"""

import scipy.special as sc
from itertools import chain



def u12equalj(j,i,ui,gamma,kappa):
#P(U_i+Z_i = j | U_i=u_i)
    #ui <= j <= 12
    #ui <= i <= 12   
    #j <= 12- (i-ui) 
    if j > ui + 12-i: #cant have more red boxes than ui + the numer of boxes that are left to open
        return 0
    if ui>j:
        return 0
    if j>12:
        raise ValueError('j has to be smaller than or equal to 12')
    if i>12:
        raise ValueError('i has to be smaller than or equal to 12')
    if ui>i:
        raise ValueError('i has to be bigger than or equal to ui')
    if i>12:
        raise ValueError('i has to be smaller than or equal to 12')
    if ui>12:
        raise ValueError('i has to be smaller than or equal to 12')
    
    
    nume = sc.beta(j-ui+gamma,12-i-(j-ui)+kappa)*sc.binom(12-i,j-ui)
    denume = sc.beta(gamma,kappa)
    return nume/denume
    
#print(u12equalj(9,7,3,1,1))
    

def redmajority(i,ui,gamma,kappa):
    #P(U_i+V_i >= 7 | U_i=u_i, U_i+V_i neq 6)
    #ui <= i <= 12 
    nume = 0
    for j in [7,8,9,10,11,12]:
        nume += u12equalj(j,i,ui,gamma,kappa)
    
    denume=0
    for j in [0,1,2,3,4,5]:
        denume += u12equalj(j,i,ui,gamma,kappa)
    denume += nume
    
    return nume/denume

#print(redmajority(7,3,1,1))
    




def nextisred(gamma,kappa):
    #P(X_i+1 = 1 | U_i=u_i) = P(X_i+1 = 1)
    return gamma/(gamma+kappa)


def viplus1_equalj(j,i,ui,gamma,kappa):
    #P(U_i+V_i=j | U_i=u_i, X_i+1 = |) = P(V_i+1=j)
    if j>11-i:
        return 0
    nume = sc.beta(j+gamma,11-i-j+kappa)*sc.binom(11-i,j)
    denume = sc.beta(gamma,kappa)
    return nume/denume

def majority_givennextisred(i,ui,gamma,kappa):
    #P(U_i+V_i neq 6 | U_i=u_i, X_i+1 = 1)
    if ui>i:
        raise ValueError('i has to be bigger than or equal to ui')
    jrange = chain(range(5-ui),range(5-ui+1,11-i+1))
    prob=0
    for j in jrange:
        prob += viplus1_equalj(j,i,ui,gamma,kappa)
    return prob

def nextisred_givenmajority(i,ui,gamma,kappa):
    #P(X_i+1 = 1 | U_i=u_i, U_i+V_i neq 6)
    denume = 0
    for j in [0,1,2,3,4,5,7,8,9,10,11,12]:
        denume += u12equalj(j,i,ui,gamma,kappa)
    return (majority_givennextisred(i,ui,gamma,kappa)*nextisred(gamma,kappa))/denume

print(nextisred_givenmajority(4,4,2,1))


    