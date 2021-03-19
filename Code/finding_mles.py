# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:31:53 2021

@author: Johan
"""

import numpy as np
import pandas as pd

#finding the mle og eta for the first participant:
#prøver først å lese inn dataene

data = "C:\\Users\\Johan\\OneDrive\\Documents\\Masteroppgave\\Data\\data1.csv"
df = pd.read_csv(data,sep=",")
df.head()
df['BoxNormExtDtD2'][0]
df['BoxNormExtChoice2'][0]

trial2 = ['r','r','r','b','r','r','r','r','b','r','b','r']

#for the first participant in the first real tiral = trial2 in the data:
dtd = df['BoxNormExtDtD2'][0]
choices = np.ones(dtd)*2
final_choice = df['BoxNormExtChoice2'][0]
print(final_choice)
print(choices)
choices[0]
if final_choice == 'Blue':
    choices = np.append(choices,0)
else:
    choices = np.append(choices,1)

print(choices)
#you find the losses of each of the choices
