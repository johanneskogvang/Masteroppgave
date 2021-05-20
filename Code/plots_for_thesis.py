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



plt.plot(theta,prob01)
plt.plot(theta,prob05)
plt.plot(theta,prob1)
plt.show()




'''
%%%%%%%%%%%%%%%%%%% The data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
data = pd.read_csv(r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\data1.csv',sep=',')
data = data.rename(columns={'Unnamed: 0':'ID'}) #putting ID as the name of the first column. 
data.head()

print(data['BoxNormExtDtd2'])





































