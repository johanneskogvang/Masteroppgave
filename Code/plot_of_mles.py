# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:39:02 2021

@author: Johan
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
%%%%%%%%%%%%%%%%% Unlimited %%%%%%%%%%%%%%%%%%%%%%%%
'''

path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\bootstrap_1000_unlim.csv'
df_unlim = pd.read_csv(path,sep=',')
df_unlim=df_unlim.drop('Unnamed: 0',axis=1)

print(df_unlim.columns)
%matplotlib inline 



eta_vec = list(df_unlim['eta'])
print(eta_vec)

alpha_vec = list(df_unlim['alpha'])
print(alpha_vec)

#Plotting eta and alpha values:
#Not zoomed
plt.scatter(alpha_vec,eta_vec,s=10,color='mediumseagreen')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\eta$')
plt.show()


#zoomed
plt.scatter(alpha_vec,eta_vec,s=10,color='mediumseagreen')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
plt.xlabel(r'$\hat{\alpha}$')
plt.ylabel(r'$\hat{\eta}$')
plt.xlim([-0.01,0.175])
plt.ylim([-5,300])
plt.show()

#zoomed even more:
plt.scatter(alpha_vec,eta_vec,s=10,color='mediumseagreen')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
plt.xlabel(r'$\hat{\alpha}$')
plt.ylabel(r'$\hat{\eta}$')
plt.xlim([-0.002,0.065])
plt.ylim([0,65])
plt.show()




'''
%%%%%%%%%%%%%%%%%%%%%%%%% LIMITED %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\bootstrap_limited_fixed1.csv'
df_lim = pd.read_csv(path,sep=',')
df_lim=df_lim.drop('Unnamed: 0',axis=1)
df_lim.head()

eta_vec = list(df_lim['eta'])
print(eta_vec)

alpha_vec = list(df_lim['alpha'])
print(alpha_vec)

beta_vec = list(df_lim['beta'])
print(beta_vec)

''' alpha and eta '''
#Plotting eta and alpha values:
#Not zoomed
plt.scatter(alpha_vec,eta_vec,s=10,color='mediumseagreen')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
plt.xlabel(r'$\hat{\alpha}$')
plt.ylabel(r'$\hat{\eta}$')
plt.show()


#zoomed
plt.scatter(alpha_vec,eta_vec,s=10,color='mediumseagreen')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
plt.xlabel(r'$\hat{\alpha}$')
plt.ylabel(r'$\hat{\eta}$')
plt.xlim([-0.002,0.06])
plt.ylim([-5,132])
plt.show()


''' alpha and beta'''
plt.scatter(alpha_vec,beta_vec,s=10,color='mediumseagreen')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
plt.xlabel(r'$\hat{\alpha}$')
plt.ylabel(r'$\hat{\beta}$')
plt.show()

#zoomed in on beta axis. 
plt.scatter(alpha_vec,beta_vec,s=10,color='mediumseagreen')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
plt.xlabel(r'$\hat{\alpha}$')
plt.ylabel(r'$\hat{\beta}$')
#plt.xlim([-0.01,0.2])
plt.ylim([-0.1,1.1])
plt.show()



'''beta and eta '''
plt.scatter(beta_vec,eta_vec,s=10,color='mediumseagreen')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
plt.xlabel(r'$\hat{\beta}$')
plt.ylabel(r'$\hat{\eta}$')
plt.show()


plt.scatter(beta_vec,eta_vec,s=10,color='mediumseagreen')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
plt.xlabel(r'$\hat{\beta}$')
plt.ylabel(r'$\hat{\eta}$')
plt.xlim([-0.05,0.7])
plt.ylim([-10,130])
plt.show()















