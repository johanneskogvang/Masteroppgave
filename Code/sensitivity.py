# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:01:04 2021

@author: Johan
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
%%%%%%%%%%%%%%%%% Unlimited %%%%%%%%%%%%%%%%%%%%%%%%
'''

''' MLEs '''
path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\bootstrap_1000_unlim.csv'
df_unlim1 = pd.read_csv(path,sep=',')
df_unlim1=df_unlim1.drop('Unnamed: 0',axis=1)

eta1 = list(df_unlim1['eta'])
alpha1 = list(df_unlim1['alpha'])


path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\bootstrap_1000_unlim.csv'
df_unlim05 = pd.read_csv(path,sep=',')
df_unlim05=df_unlim05.drop('Unnamed: 0',axis=1)

eta05 = list(df_unlim05['eta'])
alpha05 = list(df_unlim05['alpha'])




#mles unlim, not zoomed
#%matplotlib inline
fig, ax = plt.subplots()
x1 = df_unlim1['alpha']
y1 = df_unlim1['eta']
x2 = df_unlim05['alpha']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 15, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 10, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.003,headwidth=6, headlength=8)
ax.set_xlabel(r'$\hat{\alpha}$')
ax.set_ylabel(r'$\hat{\eta}$')
#path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Masteroppgave\pictures\Sensitivity\mles_unlim.pdf'
#plt.savefig(path)
plt.show()


#mles unlim, zoomed once
%matplotlib inline
fig, ax = plt.subplots()
x1 = df_unlim1['alpha']
y1 = df_unlim1['eta']
x2 = df_unlim05['alpha']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 10, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 10, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.002,headwidth=8, headlength=10)
ax.set_xlim([-0.01, 0.19])
ax.set_ylim([-10, 400])
ax.set_xlabel(r'$\hat{\alpha}$')
ax.set_ylabel(r'$\hat{\eta}$')
path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Masteroppgave\pictures\Sensitivity\mles_unlim_zoom1.pdf'
plt.savefig(path)
plt.show()

%matplotlib inline
#mles unlim, zoomed twice
fig, ax = plt.subplots()
x1 = df_unlim1['alpha']
y1 = df_unlim1['eta']
x2 = df_unlim05['alpha']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 20, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 20, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.002,headwidth=8, headlength=10)
ax.set_xlim([-0.005, 0.07])
ax.set_ylim([0, 70])
ax.set_xlabel(r'$\hat{\alpha}$')
ax.set_ylabel(r'$\hat{\eta}$')
path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Masteroppgave\pictures\Sensitivity\mles_unlim_zoom2.pdf'
plt.savefig(path)
plt.show()


%matplotlib inline
#mles unlim, zoomed three times
fig, ax = plt.subplots()
x1 = df_unlim1['alpha']
y1 = df_unlim1['eta']
x2 = df_unlim05['alpha']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 20, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 20, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.002,headwidth=8, headlength=10)
ax.set_xlim([-0.001, 0.002])
ax.set_ylim([5, 10])
ax.set_xlabel(r'$\hat{\alpha}$')
ax.set_ylabel(r'$\hat{\eta}$')
plt.show()




#test wiht different arrow sizes
%matplotlib qt
#mles unlim, not zoomed 
fig, ax = plt.subplots()
x1 = df_unlim1['alpha']
y1 = df_unlim1['eta']
x2 = df_unlim05['alpha']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 30, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 30, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.0003, headwidth=10, headlength=15)
plt.show()

%matplotlib qt
#mles unlim, zoomed once
fig, ax = plt.subplots()
x1 = df_unlim1['alpha']
y1 = df_unlim1['eta']
x2 = df_unlim05['alpha']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 30, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 30, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.0003, headwidth=10, headlength=15)
ax.set_xlim([-0.01, 0.22])
ax.set_ylim([-10, 400])
plt.show()

%matplotlib qt
#mles unlim, zoomed twice
fig, ax = plt.subplots()
x1 = df_unlim1['alpha']
y1 = df_unlim1['eta']
x2 = df_unlim05['alpha']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 30, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 30, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.0003, headwidth=10, headlength=15)
ax.set_xlim([-0.005, 0.07])
ax.set_ylim([0, 70])
plt.show()



''' Confidence Intervals '''
#alpha
%matplotlib qt
for l1,u1,y,l2,u2 in zip(df_unlim1['alpha_5_p'],df_unlim1['alpha_95_p'],range(len(df_unlim1)),df_unlim05['alpha_5_p'],df_unlim05['alpha_95_p']):
    plt.plot((l2,u2),(y,y),'ro-',color='orange')
    plt.plot((l1,u1),(y,y),'r|-',color='seagreen')
    #plt.plot((l2,u2),(y,y),'ro',color='orange')
plt.yticks(range(len(df_unlim1)),df_unlim1.index)
#plt.axvline(x=0, color='black',linewidth=0.5)

#eta
%matplotlib qt
for l1,u1,y,l2,u2 in zip(df_unlim1['eta_5_p'],df_unlim1['eta_95_p'],range(len(df_unlim1)),df_unlim05['eta_5_p'],df_unlim05['eta_95_p']):
    plt.plot((l2,u2),(y,y),'ro-',color='orange')
    plt.plot((l1,u1),(y,y),'r|-',color='seagreen')
    #plt.plot((l2,u2),(y,y),'ro',color='orange')
plt.yticks(range(len(df_unlim1)),df_unlim1.index)






'''
%%%%%%%%% LIMITED %%%%%%%%%%%%%%
'''
path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\bootstrap_limited_fixed1.csv'
df_unlim1 = pd.read_csv(path,sep=',')
df_unlim1=df_unlim1.drop('Unnamed: 0',axis=1)

path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\bootstrap_limited_fixed1000.csv'
df_unlim05 = pd.read_csv(path,sep=',')
df_unlim05=df_unlim05.drop('Unnamed: 0',axis=1)




#mles lim, alpha, eta, not zoomed
%matplotlib inline
fig, ax = plt.subplots()
x1 = df_unlim1['alpha']
y1 = df_unlim1['eta']
x2 = df_unlim05['alpha']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 15, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 10, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.003,headwidth=6, headlength=8)
ax.set_xlabel(r'$\hat{\alpha}$')
ax.set_ylabel(r'$\hat{\eta}$')
#path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Masteroppgave\pictures\Sensitivity\mles_lim_a_e.pdf'
#plt.savefig(path)
plt.show()

#mles lim, alpha, eta, zoomed
%matplotlib inline
fig, ax = plt.subplots()
x1 = df_unlim1['alpha']
y1 = df_unlim1['eta']
x2 = df_unlim05['alpha']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 15, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 10, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.003,headwidth=6, headlength=8)
ax.set_xlabel(r'$\hat{\alpha}$')
ax.set_ylabel(r'$\hat{\eta}$')
ax.set_xlim([-0.001, 0.01])
ax.set_ylim([-10, 100])
plt.show()








#mles lim, alpha, beta 
%matplotlib inline
fig, ax = plt.subplots()
x1 = df_unlim1['alpha']
y1 = df_unlim1['beta']
x2 = df_unlim05['alpha']
y2 = df_unlim05['beta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 10, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 10, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.002,headwidth=8, headlength=10)
ax.set_xlabel(r'$\hat{\alpha}$')
ax.set_ylabel(r'$\hat{\beta}$')
path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Masteroppgave\pictures\Sensitivity\mles_lim_a_b.pdf'
plt.savefig(path)
plt.show()

#mles lim, alpha, beta. Zoomed
%matplotlib inline
fig, ax = plt.subplots()
x1 = df_unlim1['alpha']
y1 = df_unlim1['beta']
x2 = df_unlim05['alpha']
y2 = df_unlim05['beta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 10, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 10, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.002,headwidth=8, headlength=10)
ax.set_xlim([-0.001, 0.001])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel(r'$\hat{\alpha}$')
ax.set_ylabel(r'$\hat{\beta}$')
plt.show()







%matplotlib inline
#mles lim, beta, eta, 
fig, ax = plt.subplots()
x1 = df_unlim1['beta']
y1 = df_unlim1['eta']
x2 = df_unlim05['beta']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 20, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 20, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.002,headwidth=8, headlength=10)
#ax.set_xlim([-0.005, 0.07])
#ax.set_ylim([0, 70])
ax.set_xlabel(r'$\hat{\beta}$')
ax.set_ylabel(r'$\hat{\eta}$')
#path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Masteroppgave\pictures\Sensitivity\mles_lim_b_e.pdf'
#plt.savefig(path)
plt.show()


#mles lim, beta, eta, zoomed once
fig, ax = plt.subplots()
x1 = df_unlim1['beta']
y1 = df_unlim1['eta']
x2 = df_unlim05['beta']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 20, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 20, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.002,headwidth=8, headlength=10)
ax.set_xlim([-0.02, 0.98])
ax.set_ylim([-10, 140])
ax.set_xlabel(r'$\hat{\beta}$')
ax.set_ylabel(r'$\hat{\eta}$')
path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Masteroppgave\pictures\Sensitivity\mles_lim_b_e_zoomed.pdf'
plt.savefig(path)
plt.show()


#mles lim, beta, eta, zoomed twice
fig, ax = plt.subplots()
x1 = df_unlim1['beta']
y1 = df_unlim1['eta']
x2 = df_unlim05['beta']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 20, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 20, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.002,headwidth=8, headlength=10)
ax.set_xlim([-0.01, 0.7])
ax.set_ylim([-10, 90])
ax.set_xlabel(r'$\hat{\beta}$')
ax.set_ylabel(r'$\hat{\eta}$')
plt.show()

#mles lim, beta, eta, zoomed bottom left corner
fig, ax = plt.subplots()
x1 = df_unlim1['beta']
y1 = df_unlim1['eta']
x2 = df_unlim05['beta']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 20, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 20, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.002,headwidth=8, headlength=10)
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-5, 0])
ax.set_xlabel(r'$\hat{\beta}$')
ax.set_ylabel(r'$\hat{\eta}$')
plt.show()



''' Confidence Intervals '''
#alpha
%matplotlib qt
for l1,u1,y,l2,u2 in zip(df_unlim1['alpha_5_p'],df_unlim1['alpha_95_p'],range(len(df_unlim1)),df_unlim05['alpha_5_p'],df_unlim05['alpha_95_p']):
    plt.plot((l2,u2),(y,y),'ro-',color='orange')
    plt.plot((l1,u1),(y,y),'r|-',color='seagreen')
    #plt.plot((l2,u2),(y,y),'ro',color='orange')
plt.yticks(range(len(df_unlim1)),df_unlim1.index)
#plt.axvline(x=0, color='black',linewidth=0.5)

#eta
%matplotlib qt
for l1,u1,y,l2,u2 in zip(df_unlim1['eta_5_p'],df_unlim1['eta_95_p'],range(len(df_unlim1)),df_unlim05['eta_5_p'],df_unlim05['eta_95_p']):
    plt.plot((l2,u2),(y,y),'ro-',color='orange')
    plt.plot((l1,u1),(y,y),'r|-',color='seagreen')
    #plt.plot((l2,u2),(y,y),'ro',color='orange')
plt.yticks(range(len(df_unlim1)),df_unlim1.index)

%matplotlib qt
for l1,u1,y,l2,u2 in zip(df_unlim1['eta_5_p'],df_unlim1['eta_95_p'],range(len(df_unlim1)),df_unlim05['eta_5_p'],df_unlim05['eta_95_p']):
    plt.plot((l2,u2),(y,y),'ro-',color='orange')
    plt.plot((l1,u1),(y,y),'r|-',color='seagreen')
    #plt.plot((l2,u2),(y,y),'ro-',color='orange')
plt.yticks(range(len(df_unlim1)),df_unlim1.index)


#beta
%matplotlib qt
for l1,u1,y,l2,u2 in zip(df_unlim1['beta_5_p'],df_unlim1['beta_95_p'],range(len(df_unlim1)),df_unlim05['beta_5_p'],df_unlim05['beta_95_p']):
    plt.plot((l2,u2),(y,y),'ro-',color='orange')
    plt.plot((l1,u1),(y,y),'r|-',color='seagreen')
    #plt.plot((l2,u2),(y,y),'ro',color='orange')
plt.yticks(range(len(df_unlim1)),df_unlim1.index)












