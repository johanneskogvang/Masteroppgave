# -*- coding: utf-8 -*-
"""
Created on Wed May 19 18:42:18 2021

@author: Johan
"""


import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


theta = np.linspace(0.02,0.98,100)

a = 1
b=1
#theta1 = np.linspace(ss.beta.ppf(0.01, a, b), ss.beta.ppf(0.99, a, b), 100)
prob1 = ss.beta.pdf(theta,a,b)
#print(prob1)
#print(theta1)


a = 0.5
b=0.5
#theta05 = np.linspace(ss.beta.ppf(0.01, a, b), ss.beta.ppf(0.99, a, b), 100)
prob05 = ss.beta.pdf(theta,a,b)
print(prob05)

a = 0.1
b=0.1
#theta01 = np.linspace(ss.beta.ppf(0.01, a, b), ss.beta.ppf(0.99, a, b), 100)
prob01 = ss.beta.pdf(theta,a,b)
print(prob01)

a=2
b=2
prob2 = ss.beta.pdf(theta,a,b)

a=5
b=5
prob5 = ss.beta.pdf(theta,a,b)


plt.plot(theta,prob01,color='darkorange')
plt.plot(theta,prob05,color='darkorchid')
plt.plot(theta,prob1,color='deeppink')
plt.plot(theta,prob2,color='black')
plt.plot(theta,prob5,color='grey')
#plt.title('The beta distribution')
plt.xlabel(r'$\theta$')
plt.ylabel('Probability density')
plt.legend([r'$\gamma=\kappa=0.1$', r'$\gamma=\kappa=0.5$',r'$\gamma=\kappa=1$',r'$\gamma=\kappa=2$',r'$\gamma=\kappa=5$'],bbox_to_anchor=(1.05, 1))
#plt.legend([r'$\gamma=\kappa=0.1$', r'$\gamma=\kappa=0.5$',r'$\gamma=\kappa=1$',r'$\gamma=\kappa=2$',r'$\gamma=\kappa=5$'],
#           bbox_to_anchor=(0., 1.02, 1., .102),ncol=5,loc='center')
plt.show()




'''
%%%%%%%%%%%%%%%%%%% The data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
data = pd.read_csv(r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\data1.csv',sep=',')
data = data.rename(columns={'Unnamed: 0':'ID'}) #putting ID as the name of the first column. 
data.head()

print(data['BoxNormExtDtD2'])



trial2 = data['BoxNormExtDtD2']
trial2 = trial2.tolist()
plt.hist(trial2,bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5],color='mediumseagreen')
plt.title('Trial 2')
plt.xlabel('Draws to decision')
plt.ylabel('Number of participants')
plt.show


trial3 = data['BoxNormExtDtD3']
trial3 = trial3.tolist()
plt.hist(trial3,bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5],color='mediumseagreen')
plt.title('Trial 3')
plt.xlabel('Draws to decision')
plt.ylabel('Number of participants')
plt.show

trial4 = data['BoxNormExtDtD4']
trial4 = trial4.tolist()
plt.hist(trial4,bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5],color='mediumseagreen')
plt.title('Trial 4')
plt.xlabel('Draws to decision')
plt.ylabel('Number of participants')
plt.show


trial5 = data['BoxNormExtDtD5']
trial5 = trial5.tolist()
plt.hist(trial5,bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5],color='mediumseagreen')
plt.title('Trial 5')
plt.xlabel('Draws to decision')
plt.ylabel('Number of participants')
plt.show

trial6 = data['BoxNormExtDtD6']
trial6 = trial6.tolist()
plt.hist(trial6,bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5],color='mediumseagreen')
plt.title('Trial 6')
plt.xlabel('Draws to decision')
plt.ylabel('Number of participants')
plt.show

trial7 = data['BoxNormExtDtD7']
trial7 = trial7.tolist()
plt.hist(trial7,bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5],color='mediumseagreen')
plt.title('Trial 7')
plt.xlabel('Draws to decision')
plt.ylabel('Number of participants')
plt.show

trial8 = data['BoxNormExtDtD8']
trial8 = trial8.tolist()
plt.hist(trial8,bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5],color='mediumseagreen')
plt.title('Trial 8')
plt.xlabel('Draws to decision')
plt.ylabel('Number of participants')
plt.show

trial9 = data['BoxNormExtDtD9']
trial9 = trial9.tolist()
plt.hist(trial9,bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5],color='mediumseagreen')
plt.title('Trial 9')
plt.xlabel('Draws to decision')
plt.ylabel('Number of participants')
plt.show

trial10 = data['BoxNormExtDtD10']
trial10 = trial10.tolist()
plt.hist(trial10,bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5],color='mediumseagreen')
plt.title('Trial 10')
plt.xlabel('Draws to decision')
plt.ylabel('Number of participants')
plt.show




