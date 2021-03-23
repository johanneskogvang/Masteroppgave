# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 09:46:08 2021

@author: Johan
"""


import numpy as np
import pandas as pd
from math import inf
from scipy.optimize import minimize
from ryddet_prosjektoppgave import make_matrix_unlim, make_matrix_lim



'''
-----------------Unlimited--------------------------------
'''
def neg_likelihood_unlim(x):
    eta=x[0]
    alpha=x[1]
    trial2 = [1,1,1,0,1,1,1,1,0,1,0,1] #blue and red
    trial3 = [0,1,0,1,0,0,1,0,1,0,1,0] #yellow and green
    trial4 = [0,0,1,0,1,0,1,0,0,1,0,0] #purple and white
    loss_mat = make_matrix_unlim(12,alpha,1,1,False) #matrix for losses of all different combination of boxes. 
    
    
    
    losses_trial2 = np.zeros(12,dtype=dict)
    #print(losses_trial2)
    h=0 #number of boxes that are opened (index of losses_trial2)
    f=0 #first index in matrix
    g=0 #second index in matrix of losses
    losses_trial2[h] = loss_mat[f][g]
    for elem in trial2:
        h += 1
        f += 1 #for each new box we go one element down in the matrix
        if elem == 1:
            g += 1 #if the new box is red, we also go one elemetn to the right. 
            #if blue, we stay in the same column
        losses_trial2[h] = loss_mat[f][g]
        if h==11:
            break
    #print(losses_trial2)
    
    
    
    losses_trial3 = np.zeros(12,dtype=dict)
    #print(losses_trial3)
    h=0 #number of boxes that are opened (index of losses_trial2)
    f=0 #first index in matrix
    g=0 #second index in matrix of losses
    losses_trial3[h] = loss_mat[f][g]
    for elem in trial3:
        h += 1
        f += 1 #for each new box we go one element down in the matrix
        if elem == 1:
            g += 1 #if the new box is red, we also go one elemetn to the right. 
            #if blue, we stay in the same column
        losses_trial3[h] = loss_mat[f][g]
        if h==11:
            break
    #print(losses_trial3)
    
    
    losses_trial4 = np.zeros(12,dtype=dict)
    #print(losses_trial4)
    h=0 #number of boxes that are opened (index of losses_trial2)
    f=0 #first index in matrix
    g=0 #second index in matrix of losses
    losses_trial4[h] = loss_mat[f][g]
    for elem in trial4:
        h += 1
        f += 1 #for each new box we go one element down in the matrix
        if elem == 1:
            g += 1 #if the new box is red, we also go one elemetn to the right. 
            #if blue, we stay in the same column
        losses_trial4[h] = loss_mat[f][g]
        if h==11:
            break
    #print(losses_trial4)
    
    
    
    
    
    data = "C:\\Users\\Johan\\OneDrive\\Documents\\Masteroppgave\\Data\\data1.csv"
    df = pd.read_csv(data,sep=",")
    df.head()
    
    #for the first participant in the first real tiral = trial2 in the data:
    dtd2 = df['BoxNormExtDtD2'][0]
    choices2 = np.ones(dtd2,dtype=int)*2
    final_choice2 = df['BoxNormExtChoice2'][0]
    if final_choice2 == 'Blue':
        choices2 = np.append(choices2,0)
    else:
        choices2 = np.append(choices2,1)
    #print(choices2)
    #you need to find the losses of each of the choices
    
    #still first participant, second trial, unlimited:
    dtd3 = df['BoxNormExtDtD3'][0]
    choices3 = np.ones(dtd3,dtype=int)*2
    final_choice3 = df['BoxNormExtChoice3'][0]
    if final_choice3 == 'Yellow':
        choices3 = np.append(choices3,0)
    else:
        choices3 = np.append(choices3,1)
    #print(choices3)
    
    #still first participant, third trial, unlimited:
    dtd4 = df['BoxNormExtDtD4'][0]
    choices4 = np.ones(dtd4,dtype=int)*2
    final_choice4 = df['BoxNormExtChoice4'][0]
    if final_choice4 == 'Purple':
        choices4 = np.append(choices4,0)
    else:
        choices4 = np.append(choices4,1)
    #print(choices4)
    
    list_of_choices = [choices2,choices3,choices4]
    #print(list_of_choices)
    
    
    
    neg_l = 0
    for liste in list_of_choices:
        i=0
        for choice in liste:
            loss = "Loss" + str(choice)
            neg_l += eta*losses_trial2[i][loss] + np.log(np.exp(-eta*losses_trial2[i]['Loss0'])+
                                                  np.exp(-eta*losses_trial2[i]['Loss1'])+
                                                  np.exp(-eta*losses_trial2[i]['Loss2']))
            i+=1
    return neg_l


bnds = ((-inf,inf),(0,inf)) #bounds for eta and alpha. 
#bnds = ((0,1),(0,inf)) #bounds for eta and alpha. 
x0 = [1,0.001] #initial guess of eta and alpha
#opt_result = minimize(neg_likelihood_unlim,x0,bounds=bnds) #the optimal eta and alpha for this function. 
#opt_result_unlim = minimize(neg_likelihood_unlim,x0,method='TNC',bounds=bnds)
#print(opt_result_unlim)





'''
-----------------Limited--------------------------------
'''


def neg_likelihood_lim(x):
    eta=x[0]
    alpha=x[1]
    beta=x[2]
    
    trial5 = [0,0,0,1,0,0,0,0,1] #Light blue and Red
    trial6 = [1,0,0,1,0,0] #Yellow and Light Green
    trial7 = [1,0,0,0,1,0,1,0,0] #Light purple and Yellowish white
    loss_mat = make_matrix_lim(12,alpha,beta,1,1,False) #matrix for losses of all different combination of boxes. 
    
    
    
    losses_trial5 = np.zeros(10,dtype=dict)
    #print(losses_trial2)
    h=0 #number of boxes that are opened (index of losses_trial2)
    f=0 #first index in matrix
    g=0 #second index in matrix of losses
    losses_trial5[h] = loss_mat[f][g]
    for elem in trial5:
        h += 1
        f += 1 #for each new box we go one element down in the matrix
        if elem == 1:
            g += 1 #if the new box is red, we also go one elemetn to the right. 
            #if blue, we stay in the same column
        losses_trial5[h] = loss_mat[f][g]
        if h==9:
            break
    #print(losses_trial2)
    
    
    '''
    losses_trial3 = np.zeros(12,dtype=dict)
    #print(losses_trial3)
    h=0 #number of boxes that are opened (index of losses_trial2)
    f=0 #first index in matrix
    g=0 #second index in matrix of losses
    losses_trial3[h] = loss_mat[f][g]
    for elem in trial3:
        h += 1
        f += 1 #for each new box we go one element down in the matrix
        if elem == 1:
            g += 1 #if the new box is red, we also go one elemetn to the right. 
            #if blue, we stay in the same column
        losses_trial3[h] = loss_mat[f][g]
        if h==11:
            break
    #print(losses_trial3)
    
    
    losses_trial4 = np.zeros(12,dtype=dict)
    #print(losses_trial4)
    h=0 #number of boxes that are opened (index of losses_trial2)
    f=0 #first index in matrix
    g=0 #second index in matrix of losses
    losses_trial4[h] = loss_mat[f][g]
    for elem in trial4:
        h += 1
        f += 1 #for each new box we go one element down in the matrix
        if elem == 1:
            g += 1 #if the new box is red, we also go one elemetn to the right. 
            #if blue, we stay in the same column
        losses_trial4[h] = loss_mat[f][g]
        if h==11:
            break
    #print(losses_trial4)
    
    '''
    
    
    
    data = "C:\\Users\\Johan\\OneDrive\\Documents\\Masteroppgave\\Data\\data1.csv"
    df = pd.read_csv(data,sep=",")
    #df.head()
    
    #for the first participant in the first limited tiral = trial5 in the data:
    dtd5 = df['BoxNormExtDtD5'][0]
    choices5 = np.ones(dtd5,dtype=int)*2
    final_choice5 = df['BoxNormExtChoice5'][0]
    if final_choice5 == 'Light blue':
        choices5 = np.append(choices5,0)
    elif final_choice5 == '-1':
        choices5 = np.append(choices5,-1)
    else:
        choices5 = np.append(choices5,1)
    #print(choices5)
    #you need to find the losses of each of the choices
    
    
    #still first participant, second trial, limited:
    dtd6 = df['BoxNormExtDtD6'][0]
    choices6 = np.ones(dtd6,dtype=int)*2
    final_choice6 = df['BoxNormExtChoice6'][0]
    if final_choice6 == 'Yellow':
        choices6 = np.append(choices6,0)
    elif final_choice6 == '-1':
        choices6 = np.append(choices6,-1)
    else:
        choices6 = np.append(choices6,1)
    #print(choices6)
    
    
    #still first participant, third trial, limited:
    dtd7 = df['BoxNormExtDtD7'][0]
    choices7 = np.ones(dtd7,dtype=int)*2
    final_choice7 = df['BoxNormExtChoice7'][0]
    if final_choice7 == 'Light purple':
        choices7 = np.append(choices7,0)
    elif final_choice7 == '-1':
        choices7 = np.append(choices7,-1)
    else:
        choices7 = np.append(choices7,1)
    #print(choices4)
    
    list_of_choices = [choices5,choices6,choices7]
    print(list_of_choices)
    
    
    
    neg_l = 0
    
    for liste in list_of_choices:
        i=0
        for choice in liste:
            if choice == -1: #if the test has terminated, the loss is beta. 
                neg_l += eta*beta + np.log(np.exp(-eta*losses_trial5[i]['Loss0'])+
                                    np.exp(-eta*losses_trial5[i]['Loss1'])+
                                    np.exp(-eta*beta))
            else:
                loss = "Loss" + str(choice)
                neg_l += eta*losses_trial5[i][loss] + np.log(np.exp(-eta*losses_trial5[i]['Loss0'])+
                                                      np.exp(-eta*losses_trial5[i]['Loss1'])+
                                                      np.exp(-eta*losses_trial5[i]['Loss2']))
            i+=1
    '''
    i=0
    for choice in choices5:
        if choice == -1: #if the test has terminated, the loss is beta. 
            neg_l += eta*beta + np.log(np.exp(-eta*losses_trial5[i]['Loss0'])+
                                np.exp(-eta*losses_trial5[i]['Loss1'])+
                                np.exp(-eta*beta))
        else:
            loss = "Loss" + str(choice)
            neg_l += eta*losses_trial5[i][loss] + np.log(np.exp(-eta*losses_trial5[i]['Loss0'])+
                                                  np.exp(-eta*losses_trial5[i]['Loss1'])+
                                                  np.exp(-eta*losses_trial5[i]['Loss2']))
       
        i+=1 
        '''
    return neg_l



bnds = ((-inf,inf),(0,inf),(0,inf)) #bounds for eta and alpha and beta 
#bnds = ((0,1),(0,inf)) #bounds for eta and alpha. 
x0_lim = [1,0.001,0.8] #initial guess of eta and alpha
opt_result_lim = minimize(neg_likelihood_lim,x0_lim,bounds=bnds) #the optimal eta, alpha and beta for this function. 
#opt_result_lim = minimize(neg_likelihood_lim,x0_lim,method='TNC',bounds=bnds)
print(opt_result_lim)













