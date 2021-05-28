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
path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\bootstrap_1000_unlim.csv"
df_unlim=pd.read_csv(path,usecols=['ID', 'eta', 'alpha', 'fun', 'eta_0', 'alpha_0',
       'alpha_5_p', 'alpha_95_p', 'eta_5_p', 'eta_95_p', 'eta200', 'alpha200'])

#print(df_unlim.columns)
#df_unlim.head()
#in unlim we plot for person 58, 13, 61.
#and maybe one that has length 0 of the CI?

person = 0
e = df_unlim['eta200'][person]
e = e[1:-1]
print(e)
e_vec = e.split()
print(e_vec)
e_float = list(map(float,e_vec))
print(e_float)

a = df_unlim['alpha200'][person]
a = a[1:-1]
a_vec = a.split()
a_float = list(map(float,a_vec))


#The CIs:
a_5 = df_unlim['alpha_5_p'][person]
a_95 = df_unlim['alpha_95_p'][person]
a_ci = [a_5,a_95]

e_5 = df_unlim['eta_5_p'][person]
e_95 = df_unlim['eta_95_p'][person]
e_ci = [e_5,e_95]





plt.scatter(a_float,e_float,s=10,color='mediumseagreen')
#plt.plot(a_ci,e_ci)
#plt.yscale('log') #for log scale of eta.
#plt.title(r"MLE of $\alpha$ and $\eta$ for the 76 participants, unlimited. $\gamma=\kappa=1$.")
plt.xlabel(r'$\hat{\alpha}$')
plt.ylabel(r'$\hat{\eta}$')
plt.show()




