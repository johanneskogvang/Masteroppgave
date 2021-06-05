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
%matplotlib inline
fig, ax = plt.subplots()
x1 = df_unlim1['alpha']
y1 = df_unlim1['eta']
x2 = df_unlim05['alpha']
y2 = df_unlim05['eta']
AB = ax.scatter(x1, y1, c = 'mediumseagreen', marker = 'o', s = 10, zorder = 3)
CD = ax.scatter(x2, y2, c = 'orange', marker = 'o', s = 10, zorder = 2)
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.003)
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
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.003)
ax.set_xlim([-0.01, 0.22])
ax.set_ylim([-10, 400])
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
ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, width=0.003)
ax.set_xlim([-0.005, 0.07])
ax.set_ylim([0, 70])
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











