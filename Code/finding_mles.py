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


#fidning the three losses in each step for trial2:
trial2 = ['r','r','r','b','r','r','r','r','b','r','b','r']

loss_mat = make_matrix_unlim(12,alpha,1,1,False) #matrix for losses of all different combination of boxes. 

losses_trial2 = np.zeros(12,dtype=dict)
print(losses_trial2)

h=0 #number of boxes that are opened (index of losses_trial2)
f=0 #first index in matrix
g=0 #second index in matrix of losses
losses_trial2[h] = loss_mat[f][g]

for elem in trial2:
    h += 1
    f += 1 #for each new box we go one element down in the matrix
    if elem == 'r':
        g += 1 #if the new box is red, we also go one elemetn to the right. 
        #if blue, we stay in the same column
    losses_trial2[h] = loss_mat[f][g]
    if h==11:
        break
print(losses_trial2)






data = "C:\\Users\\Johan\\OneDrive\\Documents\\Masteroppgave\\Data\\data1.csv"
df = pd.read_csv(data,sep=",")
df.head()
df['BoxNormExtDtD2'][0]
df['BoxNormExtChoice2'][0]

#for the first participant in the first real tiral = trial2 in the data:
dtd = df['BoxNormExtDtD2'][0]
choices = np.ones(dtd,dtype=int)*2
final_choice = df['BoxNormExtChoice2'][0]
print(final_choice)
print(choices)
choices[0]
if final_choice == 'Blue':
    choices = np.append(choices,0)
else:
    choices = np.append(choices,1)

print(choices)
#you need to find the losses of each of the choices






#finding the likelihood, eta is unknown
def neg_likelihood_func(eta):
    neg_likelihood=0
    i=0
    for choice in choices:
        loss = "Loss" + str(choice)
        neg_likelihood += eta*losses_trial2[i][loss] + np.log(np.exp(-eta*losses_trial2[i]['Loss0'])+
                                                           np.exp(-eta*losses_trial2[i]['Loss1'])+
                                                           np.exp(-eta*losses_trial2[i]['Loss2']))
    return neg_likelihood
    


#likelihood_func(0.5)



eta_0 = 1 #initial guess of eta
opt_result = minimize(neg_likelihood_func,eta_0) #the optimal eta for this function. 
print(opt_result)













