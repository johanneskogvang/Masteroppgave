# -*- coding: utf-8 -*-
"""
Created on Sun May 30 10:32:12 2021

@author: Johan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



'''
%%%%%%%%%%%%%%%%%%%%%%%%% Unlimited %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\bootstrap_1000_unlim.csv"
df=pd.read_csv(path,usecols=['ID', 'eta', 'alpha','alpha_5_p', 'alpha_95_p', 'eta_5_p', 'eta_95_p'])
df.head()

df_c = df.drop(df.loc[df['alpha_95_p']>1].index)
df_c.head()
#print(df.loc[df['alpha_95_p']>1].index)
#df_c[:][0:20]
df_small = df_c.drop(df_c.loc[df_c['alpha_95_p']>0.04].index)
df_big = df_c.drop(df_c.loc[df_c['alpha_95_p']<=0.04].index)


#plotting all cis, not for person 13. 
%matplotlib qt
for lower,upper,y in zip(df_c['alpha_5_p'],df_c['alpha_95_p'],range(len(df_c))):
    plt.plot((lower,upper),(y,y),'r|-',color='seagreen')
plt.yticks(range(len(df_c)),df_c.index)

#plotting small values of alpha
%matplotlib qt
for lower,upper,y in zip(df_small['alpha_5_p'],df_small['alpha_95_p'],range(len(df_small))):
    plt.plot((lower,upper),(y,y),'r|-',color='seagreen')
plt.yticks(range(len(df_small)),df_small.index)

#plotting big values of alpha
%matplotlib qt
for lower,upper,y in zip(df_big['alpha_5_p'],df_big['alpha_95_p'],range(len(df_big))):
    plt.plot((lower,upper),(y,y),'r|-',color='seagreen')
plt.yticks(range(len(df_big)),df_big.index)

#plotting all cis alpha:
%matplotlib qt
for lower,upper,y in zip(df['alpha_5_p'],df['alpha_95_p'],range(len(df))):
    plt.plot((lower,upper),(y,y),'r|-',color='seagreen')
plt.yticks(range(len(df)),df.index)
plt.axvline(x=0, color='black',linewidth=0.5)




#plotting all cis eta:
%matplotlib qt
for lower,upper,y in zip(df['eta_5_p'],df['eta_95_p'],range(len(df))):
    plt.plot((lower,upper),(y,y),'r|-',color='seagreen')
plt.yticks(range(len(df)),df.index)
plt.axvline(x=0, color='black',linewidth=0.5)


#trying to include a line at eta=0 and zoomed:
%matplotlib qt
for lower,upper,y in zip(df['eta_5_p'],df['eta_95_p'],range(len(df))):
    plt.plot((lower,upper),(y,y),'r|-',color='seagreen')
plt.axvline(x=0, color='black',linewidth=0.5)
#plt.xlim(-100,1000)
plt.yticks(range(len(df)),df.index)






#plotting all eta except high values
df_c = df.drop(df.loc[df['eta_95_p']>10000].index)
df_c.head()
#plotting all cis, not for person with really high value 
%matplotlib qt
for lower,upper,y in zip(df_c['eta_5_p'],df_c['eta_95_p'],range(len(df_c))):
    plt.plot((lower,upper),(y,y),'r|-',color='seagreen')
plt.yticks(range(len(df_c)),df_c.index)



'''
%%%%%%%%%%%%%%%%%%%%%%%%% Limited %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\bootstrap_limited_fixed1.csv"
df=pd.read_csv(path,usecols=['ID', 'eta', 'alpha','alpha_5_p', 'alpha_95_p', 'eta_5_p', 'eta_95_p','beta_5_p', 'beta_95_p'])
df.head()


#plotting all cis eta:
%matplotlib qt
for lower,upper,y in zip(df['eta_5_p'],df['eta_95_p'],range(len(df))):
    plt.plot((lower,upper),(y,y),'r|-',color='seagreen')
plt.yticks(range(len(df)),df.index)
plt.axvline(x=0, color='black',linewidth=0.5)





#alpha plotting all cis alpha:
%matplotlib qt
for lower,upper,y in zip(df['alpha_5_p'],df['alpha_95_p'],range(len(df))):
    plt.plot((lower,upper),(y,y),'r|-',color='seagreen')
plt.yticks(range(len(df)),df.index)
plt.axvline(x=0, color='black',linewidth=0.5)




#alpha: plotting all cis, not for person 11. 
df_c = df.drop(df.loc[df['alpha_95_p']>1].index)
df_c.head()
#print(df.loc[df['alpha_95_p']>1].index)
#df_c[:][0:20]
df_small = df_c.drop(df_c.loc[df_c['alpha_95_p']>0.04].index)
df_big = df_c.drop(df_c.loc[df_c['alpha_95_p']<=0.04].index)


%matplotlib qt
for lower,upper,y in zip(df_c['alpha_5_p'],df_c['alpha_95_p'],range(len(df_c))):
    plt.plot((lower,upper),(y,y),'r|-',color='seagreen')
plt.yticks(range(len(df_c)),df_c.index)



#plotting all cis beta:
%matplotlib qt
for lower,upper,y in zip(df['beta_5_p'],df['beta_95_p'],range(len(df))):
    plt.plot((lower,upper),(y,y),'r|-',color='seagreen')
plt.yticks(range(len(df)),df.index)
plt.axvline(x=0, color='black',linewidth=0.5)



#plotting all cis beta except person 11 and 75:
df_c = df.drop(df.loc[df['beta_95_p']>10].index)
df_c.head()
#print(df.loc[df['alpha_95_p']>1].index)
#df_c[:][0:20]
#df_small = df_c.drop(df_c.loc[df_c['alpha_95_p']>0.04].index)
#df_big = df_c.drop(df_c.loc[df_c['alpha_95_p']<=0.04].index)


%matplotlib qt
for lower,upper,y in zip(df_c['beta_5_p'],df_c['beta_95_p'],range(len(df_c))):
    plt.plot((lower,upper),(y,y),'r|-',color='seagreen')
plt.yticks(range(len(df_c)),df_c.index)





















