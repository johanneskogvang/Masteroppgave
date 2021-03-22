# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:31:53 2021

@author: Johan
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ryddet_prosjektoppgave import make_matrix_unlim, make_matrix_lim

#finding the mle og eta for the first participant:
#prøver først å lese inn dataene

#values for alpha and beta:
alpha = 0.001
beta = 1


#fidning the three losses in each step for trial2, trial3 and trial4:
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
print(losses_trial2)



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
print(losses_trial3)


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
print(losses_trial4)




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




#finding the likelihood, eta is unknown, only for the first trial for the first person
"""
def neg_likelihood_func(eta):
    neg_likelihood=0
    i=0
    for choice in choices2:
        loss = "Loss" + str(choice)
        neg_likelihood += eta*losses_trial2[i][loss] + np.log(np.exp(-eta*losses_trial2[i]['Loss0'])+
                                                           np.exp(-eta*losses_trial2[i]['Loss1'])+
                                                           np.exp(-eta*losses_trial2[i]['Loss2']))
        i+=1
    return neg_likelihood
"""    
#likelihood for the three unlimited trials for the first person
def neg_likel_func(eta):
    neg_likel = 0
    for liste in list_of_choices:
        i=0
        for choice in liste:
            loss = "Loss" + str(choice)
            neg_likel += eta*losses_trial2[i][loss] + np.log(np.exp(-eta*losses_trial2[i]['Loss0'])+
                                                           np.exp(-eta*losses_trial2[i]['Loss1'])+
                                                           np.exp(-eta*losses_trial2[i]['Loss2']))
            i+=1
    return neg_likel

#print(neg_likelihood_func(0.5))
print(neg_likel_func(0.5))


eta_0 = 1 #initial guess of eta
opt_result = minimize(neg_likel_func,eta_0) #the optimal eta for this function. 
print(opt_result)



print(neg_likel_func(2.56376899))
print(neg_likelihood_func(3))
print(neg_likelihood_func(2))








