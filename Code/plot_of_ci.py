# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:18:13 2021

@author: Johan
"""

import numpy as np
import pandas as pd


#plotting mles and cis in one figure


'''
%%%%%%%%%%%%%%% Unlimited %%%%%%%%%%%%%%%%%%%%%%
'''

#first loading data:
path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\bootstrap_1000_unlim.csv"
df_unlim=pd.read_csv(path,usecols=['ID', 'eta', 'alpha', 'fun', 'eta_0', 'alpha_0',
       'alpha_5_p', 'alpha_95_p', 'eta_5_p', 'eta_95_p', 'eta200', 'alpha200'])

#print(df_unlim.columns)
#df_unlim.head()
#in unlim we plot for person 58, 13, 61.
#and maybe one that has length 0 of the CI? this si person 58

person = 61
e = df_unlim['eta200'][person]
e = e[1:-1]
#print(e)
e_vec = e.split()
#print(e_vec)
e_float = list(map(float,e_vec))
#print(e_float)

a = df_unlim['alpha200'][person]
a = a[1:-1]
a_vec = a.split()
a_float = list(map(float,a_vec))


#The CIs:
a_5 = df_unlim['alpha_5_p'][person]
a_95 = df_unlim['alpha_95_p'][person]
a_ci = [a_5,a_95]
a_ci_mid = (a_95+a_5)/2

e_5 = df_unlim['eta_5_p'][person]
e_95 = df_unlim['eta_95_p'][person]
e_ci = [e_5,e_95]
e_ci_mid = (e_95+e_5)/2




plt.scatter(a_float,e_float,s=10,color='mediumseagreen')
#plt.plot(a_ci,e_ci)
plt.plot(a_ci,[e_ci_mid,e_ci_mid],'.-',color='black')
plt.plot([a_ci_mid,a_ci_mid],e_ci,'.-',color='black')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
#plt.xlim([-0.02,4])
#plt.ylim([-100,120])
plt.xlabel(r'$\hat{\alpha}$')
plt.ylabel(r'$\hat{\eta}$')
plt.show()



#zoomed
plt.scatter(a_float,e_float,s=10,color='mediumseagreen')
#plt.plot(a_ci,e_ci)
plt.plot(a_ci,[e_ci_mid,e_ci_mid],'.-',color='black')
plt.plot([a_ci_mid,a_ci_mid],e_ci,'.-',color='black')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=0.5$.")
plt.xlim([-0.001,0.053])
plt.ylim([-20,300])
plt.xlabel(r'$\hat{\alpha}$')
plt.ylabel(r'$\hat{\eta}$')
plt.show()
















'''
%%%%%%%%%%%%%%%%%%%%%% Limited %%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\bootstrap_limited_fixed1.csv"
df_lim=pd.read_csv(path,usecols=['ID', 'eta', 'alpha', 'beta','eta_0', 'alpha_0','beta_0',
       'alpha_5_p', 'alpha_95_p', 'beta_5_p','beta_95_p', 'eta_5_p', 'eta_95_p', 'eta200', 'alpha200','beta200'])

df_lim.columns
#df_lim.head()

person = 40
e = df_lim['eta200'][person]
e = e[1:-1]
#print(e)
e_vec = e.split()
#print(e_vec)
e_float = list(map(float,e_vec))
#print(e_float)

a = df_lim['alpha200'][person]
a = a[1:-1]
a_vec = a.split()
a_float = list(map(float,a_vec))
#print(a_float)

b = df_lim['beta200'][person]
b = b[1:-1]
b_vec = b.split()
#print(b_vec)
b_float = list(map(float,b_vec))
#print(b_float)



a_5 = df_lim['alpha_5_p'][person]
a_95 = df_lim['alpha_95_p'][person]
a_ci = [a_5,a_95]
a_ci_mid = (a_95+a_5)/2

e_5 = df_lim['eta_5_p'][person]
e_95 = df_lim['eta_95_p'][person]
e_ci = [e_5,e_95]
e_ci_mid = (e_95+e_5)/2

b_5 = df_lim['beta_5_p'][person]
b_95 = df_lim['beta_95_p'][person]
b_ci = [b_5,b_95]
b_ci_mid = (b_95+b_5)/2


#alpha and eta
%matplotlib inline
plt.scatter(a_float,e_float,s=10,color='mediumseagreen')
#plt.plot(a_ci,e_ci)
plt.plot(a_ci,[e_ci_mid,e_ci_mid],'.-',color='black')
plt.plot([a_ci_mid,a_ci_mid],e_ci,'.-',color='black')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
#plt.xlim([-50,1400])
plt.ylim([-170,4300])
plt.xlabel(r'$\hat{\alpha}$')
plt.ylabel(r'$\hat{\eta}$')
plt.show()



plt.scatter(a_float,b_float,s=10,color='mediumseagreen')
#plt.plot(a_ci,e_ci)
plt.plot(a_ci,[b_ci_mid,b_ci_mid],'.-',color='black')
plt.plot([a_ci_mid,a_ci_mid],b_ci,'.-',color='black')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
#plt.xlim([-50,1400])
#plt.ylim([-5,76])
plt.xlabel(r'$\hat{\alpha}$')
plt.ylabel(r'$\hat{\beta}$')
plt.show()


#beta and eta
plt.scatter(b_float,e_float,s=10,color='mediumseagreen')
#plt.plot(a_ci,e_ci)
plt.plot(b_ci,[e_ci_mid,e_ci_mid],'.-',color='black')
plt.plot([b_ci_mid,b_ci_mid],e_ci,'.-',color='black')
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
#plt.xlim([-1,30])
plt.ylim([-50,4200])
plt.xlabel(r'$\hat{\beta}$')
plt.ylabel(r'$\hat{\eta}$')
plt.show()














