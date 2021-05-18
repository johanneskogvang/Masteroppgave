# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:19:07 2021

@author: Johan
"""



import numpy as np
import scipy as sp 
import pandas as pd
from math import inf
from scipy.optimize import minimize, brute
from ryddet_prosjektoppgave import make_matrix_unlim
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import matplotlib.pyplot as plt

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            firstly finding the negative loss function in the unlimited case. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def neg_log_likel_unlim(x,df,person):
    #df contains the data from the participants having done the box task.
    eta = x[0]
    alpha = x[1]
    gamma = 1
    kappa = 1
    
    """
    Starting to find the expected losses for trial 2 which is the first unlimited trial.
    """
    #finding the matrix with the expected loss for all possible combinations of red and blue boxes.
    exp_loss_mat = make_matrix_unlim(12,alpha,gamma,kappa,True)
    
    
    #the first unlimited trial is the second trial. 
    trial2 = [1,1,1,0,1,1,1,1,0,1,0,1] #blue=0 and red=1. 
    trial3 = [0,1,0,1,0,0,1,0,1,0,1,0] #Yellow=0, Green=1
    trial4 = [0,0,1,0,1,0,1,0,0,1,0,0] #Purple=0, White=1
    
    #need to find the expected losses for the case we have here:
    #first initializing
    expL_2 = np.zeros(12,dtype=dict) #these are the exp loss for trial 2. 
    expL_3 = np.zeros(12,dtype=dict) #these are the exp loss for trial 3. 
    expL_4 = np.zeros(12,dtype=dict) #these are the exp loss for trial 4. 

    #The second entry for the matrix for trial 2:
    j2=0    
    #the second entry for the matrix for trial 3:
    j3=0    
    #The second entry for the matrix for trial 4:
    j4=0
    
    k=0 #the elements in expL_2, expL_3 and expL4 and the first entry in the matrix
    for a in range(len(trial2)):
        expL_2[k] = exp_loss_mat[k][j2] #finding the expected losses for trial 2 in this situation in the matrix with all the exp losses.
        expL_3[k] = exp_loss_mat[k][j3] 
        expL_4[k] = exp_loss_mat[k][j4]
        k+=1 #for finding the next element in expL2,3,4 and going down one row in the matrix
        #i+=1 #going down one row in the loss matrix
        if trial2[a] == 1: #if the box that is opened is red, we also go one element to the right in the matrix
            j2+=1
        if trial3[a] == 1: #if the box that is opened is green, we also go one element to the right in the matrix
            j3+=1
        if trial4[a] == 1: #if the box that is opened is white, we also go one element to the right in the matrix
            j4+=1
     #dette ser riktig ut når jeg sjekker om riktig element er kommet med. 
    #all_expL = [expL_2]
    all_expL = [expL_2,expL_3,expL_4]
    
    
    
    """
    finding the choices that the participant make. firslty only for the second participant in the first unlimited trial (=trial2)
    """

    #person=0
    #only finding the data for the first participant now., that is row 0.
    dtd2 = df['BoxNormExtDtD2'][person] #dette er riktig
    choice2 = df['BoxNormExtChoice2'][person] #dette er riktig
    all_choices2 = 2*np.ones(dtd2+1,dtype=int) #making an array of the decisions that the participant make (2 untli last choice)
    if choice2=='Blue': #we denothe the choice of blue as majority colour as '0'
        all_choices2[-1]=0
    else: #choosing res as majority colour is denoted '1'
        all_choices2[-1]=1
    #dette ser også riktig ut for trial2 for første participant. 
        
        
     #finding the decisions for the third trial:
    dtd3 = df['BoxNormExtDtD3'][0]
    all_choices3 = 2*np.ones(dtd3+1,dtype=int)
    if df['BoxNormExtChoice3'][0] == "Yellow":
        all_choices3[-1]=0
    else:
        all_choices3[-1]=1
    
    #decisions for the  fourth trial
    dtd4 = df['BoxNormExtDtD4'][0]
    all_choices4 = 2*np.ones(dtd4+1,dtype=int)
    if df['BoxNormExtChoice4'][0] == 'Purple':
        all_choices4[-1]=0
    else:
        all_choices4[-1]=1
        
        
    #all_choices = [all_choices2]
    all_choices = [all_choices2,all_choices3,all_choices4]
    
    
    """
    now that we have the exp losses and the choices that the participant make, we can start finding the negative log likelihood
    """
    neg_l = 0
    for trial in range(3):
        choices = all_choices[trial]
        expL = all_expL[trial]
        i=0
        for choice in choices:
            loss = 'Loss'+str(choice)
            e_delta = expL[i][loss]
            e0=expL[i]['Loss0']
            e1=expL[i]['Loss1']
            e2=expL[i]['Loss2']
            const = max(-eta*e0,-eta*e1,-eta*e2) #blir dette riktig????
            neg_l += eta*e_delta + const + np.log(np.exp(-eta*e0-const)
                                               +np.exp(-eta*e1-const)
                                              +np.exp(-eta*e2-const))
            
            
            i+=1
            
        
    """    
    trial = 0 #dettte må bli en for-løkke når alle tre trials er med
    choices = all_choices[trial]
    expL = all_expL[trial]
    i=0
    for choice in choices:
        loss = 'Loss'+str(choice)
        e_delta = expL[i][loss]
        e0=expL[i]['Loss0']
        e1=expL[i]['Loss1']
        e2=expL[i]['Loss2']
        const = max(-eta*e0,-eta*e1,-eta*e2) #blir dette riktig????
        neg_l += eta*e_delta + const + np.log(np.exp(-eta*e0-const)
                                           +np.exp(-eta*e1-const)
                                          +np.exp(-eta*e2-const))
        
        
        i+=1
    """

    
    return neg_l



#start by loading the data:
data = pd.read_csv(r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\data1.csv',sep=',')
data = data.rename(columns={'Unnamed: 0':'ID'}) #putting ID as the name of the first column. 

person=0 #whcih person we want to find the mles for


#finding the mle estimators of eta and alpha.
#that is, we are minimizing the negative log likelihood function
bnds = ((-inf,inf),(0,inf))
x0_unlim = [0,0.01]
mles = minimize(neg_log_likel_unlim,args=(data,person),x0=x0_unlim,bounds=bnds)
#mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(data,person),bounds=bnds,method='L-BFGS-B')
#opt = {'ftol':1e-100,'gtol':1e-100,'maxiter':1500000,'maxfun':15000000}
#opt = {'ftol':1e-100,'gtol':1e-100,'maxiter':150000,'maxfun':150000}
#mles = minimize(neg_log_likel_unlim,args=(data,person),x0=x0_unlim,bounds=bnds,method='L-BFGS-B',options=opt)
#mles = minimize(neg_log_likel_unlim,args=(data,person),x0=x0_unlim,bounds=bnds,method='BFGS',options={'g-tol':1e-40})
#mles = minimize(neg_log_likel_unlim,args=(data,person),x0=x0_unlim,bounds=bnds,method='Nelder-Mead')
print(mles)


'''
scipy fmin, kan evt bruke dette isf minimize.
toleranmsen er på gradienten
siden funk er veldgi flat, gir det forskjellig svar i femte desimal.
hvis gradienten er liten nok vil den bare stoppe der. 
sjekk for flere personer. 
'''














