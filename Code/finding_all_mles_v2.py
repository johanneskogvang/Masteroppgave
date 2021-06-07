# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 08:39:57 2021

@author: Johan
"""



import numpy as np
import scipy as sp 
import pandas as pd
from math import inf
from scipy.optimize import minimize, brute
from ryddet_prosjektoppgave import make_matrix_unlim, make_matrix_lim
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import matplotlib.pyplot as plt
import random
import time
#import multiprocessing as mp


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                putting data from participants into a dataframe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

#start by loading the data:
data = pd.read_csv(r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\data1.csv',sep=',')
data = data.rename(columns={'Unnamed: 0':'ID'}) #putting ID as the name of the first column. 






 """
Starting to find the expected losses for trial 2,3 and 4 which are the unlimited trials. Making a function for it.
"""

def all_exp_losses_unlim(alpha,gamma,kappa):

    #finding the matrix with the expected loss for all possible combinations of red and blue boxes.
    exp_loss_mat = make_matrix_unlim(12,alpha,gamma,kappa,True)
    
    
    #the first unlimited trial is the second trial. 
    trial2 = [1,1,1,0,1,1,1,1,0,1,0,1] #blue=0 and red=1. 
    trial3 = [0,1,0,1,0,0,1,0,1,0,1,0] #Yellow=0, Green=1
    trial4 = [0,0,1,0,1,0,1,0,0,1,0,0] #Purple=0, White=1
    
    #need to find the expected losses for the case we have here:
    #first initializing
    expL_2 = np.zeros(13,dtype=dict) #these are the exp loss for trial 2.
    #print('expL_2',expL_2)
    expL_3 = np.zeros(13,dtype=dict) #these are the exp loss for trial 3. 
    expL_4 = np.zeros(13,dtype=dict) #these are the exp loss for trial 4. 
    
    #The second entry for the matrix for trial 2:
    j2=0    
    #the second entry for the matrix for trial 3:
    j3=0    
    #The second entry for the matrix for trial 4:
    j4=0
    
    k=0 #the elements in expL_2, expL_3 and expL4 and the first entry in the matrix
    for a in range(len(trial2)):
        expL_2[k] = exp_loss_mat[k][j2] #finding the expected losses for trial 2 in this situation in the matrix with all the exp losses.
        expL_3[k] = exp_loss_mat[k][j3] 
        expL_4[k] = exp_loss_mat[k][j4]
        k+=1 #for finding the next element in expL2,3,4 and going down one row in the matrix
        #i+=1 #going down one row in the loss matrix
        if trial2[a] == 1: #if the box that is opened is red, we also go one element to the right in the matrix
            j2+=1
        if trial3[a] == 1: #if the box that is opened is green, we also go one element to the right in the matrix
            j3+=1
        if trial4[a] == 1: #if the box that is opened is white, we also go one element to the right in the matrix
            j4+=1
     #dette ser riktig ut når jeg sjekker om riktig element er kommet med. 
    #all_expL = [expL_2]
            
    #the last elemnt of each of the exp losses is not included in the matrix. 
    #here, we already know what the majority colour is, and then waht the probability and losses are. Loss2 is not existing as it is not possible to open another box.
    #Thus, we are putting Loss2 as 1 if the participant did not choose majority colour, as this counts as a failed trial
    expL_2[-1] = {'prob':1,'Loss0':1,'Loss1':0,'Loss2':1}
    expL_3[-1] = {'prob':0,'Loss0':0,'Loss1':1,'Loss2':1}
    expL_4[-1] = {'prob':0,'Loss0':0,'Loss1':1,'Loss2':1}
        
    all_expL = [expL_2,expL_3,expL_4]
        #print('expL_2',expL_2)
    return all_expL

#expL = all_exp_losses_unlim(0.01,1,1)
#print(expL)


'''
%%%%%%%%%%%% finding the choices that each particiant make %%%%%%%%%
'''
def participant_choices_unlim(df,person):
    dtd2 = df['BoxNormExtDtD2'][person] #dette er riktig
    choice2 = df['BoxNormExtChoice2'][person] #dette er riktig
    all_choices2 = 2*np.ones(dtd2,dtype=int) #making an array of the decisions that the participant make (2 untli last choice)
    if choice2=='Blue': #we denothe the choice of blue as majority colour as '0'
        all_choices2 = np.append(all_choices2,0)
    if choice2=='Red': #choosing red as majority colour is denoted '1'
        all_choices2 = np.append(all_choices2,1)
    if choice2 == -1:
        all_choices2 = np.append(all_choices2,-1)
    #dette ser også riktig ut for trial2 for første participant. 
        
        
     #finding the decisions for the third trial:
    dtd3 = df['BoxNormExtDtD3'][person]
    all_choices3 = 2*np.ones(dtd3,dtype=int)
    if df['BoxNormExtChoice3'][person] == "Yellow":
        all_choices3 = np.append(all_choices3,0)
    if df['BoxNormExtChoice3'][person]=='Green':
        all_choices3 = np.append(all_choices3,1)
    if df['BoxNormExtChoice3'][person]==-1:
        all_choices3 = np.append(all_choices3,-1)
    
    #decisions for the  fourth trial
    dtd4 = df['BoxNormExtDtD4'][person]
    all_choices4 = 2*np.ones(dtd4,dtype=int)
    if df['BoxNormExtChoice4'][person] == 'Purple':
        all_choices4 = np.append(all_choices4,0)
    if df['BoxNormExtChoice4'][person]=='White':
        all_choices4 = np.append(all_choices4,1)
    if df['BoxNormExtChoice4'][person]==-1:
        all_choices4 = np.append(all_choices4,-1)
        
        
    #all_choices = [all_choices2]
    all_choices = [all_choices2,all_choices3,all_choices4]
    return all_choices

#print(participant_choices_unlim(data,0))




"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            firstly finding the negative loss function in the unlimited case. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def neg_log_likel_unlim(x, person,gamma,kappa,all_choices):
#def neg_log_likel_unlim(x, df, person,gamma,kappa,all_choices):
    eta = x[0]
    alpha = x[1]

    #starting to find the expected losses for teh three trials
    all_expL = all_exp_losses_unlim(alpha,gamma,kappa)
    
    #finding the choices that the participant make.
    #all_choices =  participant_choices_unlim(df,person) #these choices should be as an input in the function to make it easier when you dp this for simulated data
    
    
    #now that we have the exp losses and the choices that the participant make, we can start finding the negative log likelihood
    neg_l = 0
    for trial in range(3):
        choices = all_choices[trial]
        expL = all_expL[trial]
        #print('expL ',expL)
        i=0
        for choice in choices:
            #her må du får med at choice noen ganger er -1, og da er loss lik 1, altså e_delta er 1 
            #nå sier jeg at det regnes som et faield trial hvos man ikke velger farge. altså loss er 1 for å ikke velge farge. 
            if choice == -1:
                e_delta = 1 #men skal jeg egntlig har med noe her??? man har jo ikke noe exp loss når testen terminerer??   
            else:
                loss = 'Loss'+str(choice)
                e_delta = expL[i][loss]
            e0=expL[i]['Loss0']
            e1=expL[i]['Loss1']
            e2=expL[i]['Loss2']
            const = max(-eta*e0,-eta*e1,-eta*e2) #blir dette riktig????
            neg_l += eta*e_delta + const + np.log(np.exp(-eta*e0-const)
                                               +np.exp(-eta*e1-const)
                                              +np.exp(-eta*e2-const))
            i+=1
            
    return neg_l
    #return neg_l, all_expL








'''
%%%%%%%%%%%% mle for only one participant with different values for x0 %%%%%%%%%%%%%%%%%%%%
'''

#finding the mle estimators of eta and alpha.
#that is, we are minimizing the negative log likelihood function
#this is only for one person
#trying to do this for many different values of x0, and then choosing hte values that give the lowest function value.
tic = time.perf_counter()
person=58
result_unlim = {'eta':0,'alpha':0,'f':inf}    
bnds = ((-inf,inf),(0,inf))
for k in range(200):
    #alpha0 should be a random number between 0 and 0.1:
    alpha_0 = 0.1*random.random()
    #eta_0 schould be a random number between 3 and 20:
    eta_0 = 3 + 17*random.random()
    x0_unlim = [eta_0,alpha_0]
    mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(person,1,1,participant_choices_unlim(data,person)),bounds=bnds,method='L-BFGS-B')
    if mles.fun < result_unlim['f']:
        print('old:',result_unlim['f'])
        print('new:',mles.fun)
        result_unlim['eta'] = mles.x[0]
        result_unlim['alpha'] = mles.x[1]
        result_unlim['f'] = mles.fun

print(result_unlim)
toc = time.perf_counter()
print(f'Code ran in {toc - tic:0.4f} seconds')





'''
%%%%%%%%%%%%%%%%% finding mles for all participants %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

#saving the mles in a dataframe
df_mles = pd.DataFrame(columns = ['ID','eta','alpha','fun','eta_0','alpha_0'], index = np.arange(0,76))
df_mles['fun'] = inf
#df_mles.head()

bnds = ((-inf,inf),(0,inf)) #bounds for eta and alpha when minimizing
#x0_unlim = [0,0.01] #start values of eta and alpha
#opt = {'ftol':1e-100,'gtol':1e-100,'maxiter':150000,'maxfun':150000}

for person in range(len(data)):
#for person in range(4,len(data)):
    ID = data['ID'][person]
    print(person, ID)    
    df_mles['ID'][person] = ID
    for k in range(200):
    #alpha0 should be a random number between 0 and 0.1:
        alpha_0 = 0.1*random.random()
        #eta_0 schould be a random number between 3 and 20:
        eta_0 = 3 + 17*random.random()
        x0_unlim = [eta_0,alpha_0]
        mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(person,0.5,0.5,participant_choices_unlim(data,person)),bounds=bnds,method='L-BFGS-B')
        if mles.fun < df_mles['fun'][person]:
            print('old:',df_mles['fun'][person])
            print('new:',mles.fun)
            df_mles['eta'][person] = mles.x[0]
            df_mles['alpha'][person] = mles.x[1]
            df_mles['fun'][person] = mles.fun
            df_mles['eta_0'][person] = eta_0
            df_mles['alpha_0'][person] = alpha_0
            print('new',df_mles['fun'][person])

    
    #mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(data,person,0.5,0.5),bounds=bnds,options=opt) #these two are the same, so its prob l-bfgs-b that is used. 
    #mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(data,person,0.5,0.5),bounds=bnds,method='L-BFGS-B')
    #eta = mles.x[0]
    #alpha = mles.x[1]
    #print(eta,alpha)
    #df_mles.loc[person] = [ID,eta,alpha]
    #df_mles.append({'ID':ID,'eta':eta, 'alpha':alpha},ignore_index=True)



df_mles.head()
#df_mles['fun'][0]

#try saving this datafram in csv file:
csv_file_name_and_loc = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\mles_unlimited_x0_200_times.csv'
df_mles.to_csv(csv_file_name_and_loc)





'''
%%%%%%%%%%%%%%%% Bootstrapping %%%%%%%%%%%%%%%
'''
#starting w the first unlimited trial
#first finding the probabilities for one person in the first step, then simulating what that person would do in that step
#then do the same for the next step



def simulating_decisions_unlim(person, alpha_hat,eta_hat,gamma,kappa):
    #starting to find the expected losses for the three trials
    all_expL = all_exp_losses_unlim(alpha_hat,gamma,kappa) #the expected losses based on what colours that the boxes will be in. 
    
    expL_2 = all_expL[0]
    decisions2 = [] #array that holds all decisions that are simulated.
    i=0 #how many boxes that are opened
    dec=2 #decising to opn a box
    while dec==2:
        if i == 13: #if they havent made a decision yet, and there are no more boxes to open. 
            break
    #for the first step, find p0, p1 and p2, which are the probs that choice 0, 1 and 2 are made, respectively.
        e0 = expL_2[i]['Loss0'] 
        e1 = expL_2[i]['Loss1']
        e2 = expL_2[i]['Loss2']
        #print('e0:',e0)
        #print('e1:',e1)
        #print('e2:',e2)
        const = min(eta_hat*e0,eta_hat*e1,eta_hat*e2)
        denom = np.exp(-eta_hat*e0+const) + np.exp(-eta_hat*e1+const) + np.exp(-eta_hat*e2+const)
        #print('denom:',denom)
        p0 = np.exp(-eta_hat*e0+const)/denom
        p1 = np.exp(-eta_hat*e1+const)/denom
        p2 = np.exp(-eta_hat*e2+const)/denom
        
        '''
        denom = np.exp(-eta_hat*e0) + np.exp(-eta_hat*e1) + np.exp(-eta_hat*e2)
        print('denom:',denom)
        p0 = np.exp(-eta_hat*e0)/denom
        p1 = np.exp(-eta_hat*e1)/denom
        p2 = np.exp(-eta_hat*e2)/denom
        '''
        #print('probabilities_2:',p0,p1,p2)
        dec = np.random.choice([0,1,2],p=[p0,p1,p2]) #making a decision based on those probabilities. 
        #print('probs:',p0,p1,p2)
        #print(dec)
        decisions2.append(dec)
        i+=1
    #print(decisions2)
    
    expL_3 = all_expL[1]
    decisions3 = [] #array that holds all decisions that are simulated.
    i=0 #how many boxes that are opened
    dec=2 #decising to opn a box
    while dec==2:
        if i == 13: #if they havent made a decision yet, and there are no more boxes to open. 
            break
    #for the first step, find p0, p1 and p2, which are the probs that choice 0, 1 and 2 are made, respectively.
        e0 = expL_3[i]['Loss0'] 
        e1 = expL_3[i]['Loss1']
        e2 = expL_3[i]['Loss2']
        #print('e0:',e0)
        #print('e1:',e1)
        #print('e2:',e2)
        const = min(eta_hat*e0,eta_hat*e1,eta_hat*e2)
        denom = np.exp(-eta_hat*e0+const) + np.exp(-eta_hat*e1+const) + np.exp(-eta_hat*e2+const)
        #print('denom:',denom)
        p0 = np.exp(-eta_hat*e0+const)/denom
        p1 = np.exp(-eta_hat*e1+const)/denom
        p2 = np.exp(-eta_hat*e2+const)/denom
        '''
        denom = np.exp(-eta_hat*e0) + np.exp(-eta_hat*e1) + np.exp(-eta_hat*e2)
        p0 = np.exp(-eta_hat*e0)/denom
        p1 = np.exp(-eta_hat*e1)/denom
        p2 = np.exp(-eta_hat*e2)/denom
        '''
        #print('probabilities_3:',p0,p1,p2)
        dec = np.random.choice([0,1,2],p=[p0,p1,p2]) #making a decision based on those probabilities. 
        #print('probs:',p0,p1,p2)
        #print(dec)
        decisions3.append(dec)
        i+=1
    #print(decisions3)
    
    expL_4 = all_expL[1]
    decisions4 = [] #array that holds all decisions that are simulated.
    i=0 #how many boxes that are opened
    dec=2 #decising to opn a box
    while dec==2:
    #for the first step, find p0, p1 and p2, which are the probs that choice 0, 1 and 2 are made, respectively.
        if i == 13: #if they havent made a decision yet, and there are no more boxes to open. 
            break
        e0 = expL_4[i]['Loss0'] 
        e1 = expL_4[i]['Loss1']
        e2 = expL_4[i]['Loss2']
        #print('e0:',e0)
        #print('e1:',e1)
        #print('e2:',e2)
        const = min(eta_hat*e0,eta_hat*e1,eta_hat*e2)
        denom = np.exp(-eta_hat*e0+const) + np.exp(-eta_hat*e1+const) + np.exp(-eta_hat*e2+const)
        #print('denom:',denom)
        p0 = np.exp(-eta_hat*e0+const)/denom
        p1 = np.exp(-eta_hat*e1+const)/denom
        p2 = np.exp(-eta_hat*e2+const)/denom
        '''        
        denom = np.exp(-eta_hat*e0) + np.exp(-eta_hat*e1) + np.exp(-eta_hat*e2)
        p0 = np.exp(-eta_hat*e0)/denom #prob that the particiapnt chooses blue as majority colour
        p1 = np.exp(-eta_hat*e1)/denom #prob that the particiapnt chooses red as majority colour
        p2 = np.exp(-eta_hat*e2)/denom #prob that part chooses to open another box. 
        '''
        #print('probabilities_4:',p0,p1,p2)
        dec = np.random.choice([0,1,2],p=[p0,p1,p2]) #making a decision based on those probabilities.  dec=decison. 
        #print('probs:',p0,p1,p2)
        #print(dec)
        decisions4.append(dec)
        i+=1
    #print(decisions4)
    decisions = [decisions2,decisions3,decisions4]
    #print(decisions)
    return decisions

 
def simulating_200(person,df,gamma,kappa):
    num_sim = 1000
    alpha_200 = np.zeros(num_sim)
    eta_200 = np.zeros(num_sim)
    alpha_hat = df['alpha'][person]
    eta_hat = df['eta'][person]
    alpha_0 = df['alpha_0'][person]
    eta_0 = df['eta_0'][person]
    for k in range(num_sim):
        simulated_decisions = simulating_decisions_unlim(person,alpha_hat,eta_hat,gamma,kappa)
        bnds = ((-inf,inf),(0,inf)) #bounds for eta and alpha when minimizing
        '''
        f = inf
        for m in range(20):
            alpha_0 = 0.1*random.random()
            #eta_0 should be a random number between 3 and 20:
            eta_0 = 3 + 17*random.random()
            x0_unlim = [eta_0,alpha_0]
            mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(person,gamma,kappa,simulated_decisions),bounds=bnds,method='L-BFGS-B')
            if mles.fun < f:
                eta_200[k] = mles.x[0]
                alpha_200[k] = mles.x[1]
                f = mles.fun
                print(f)
    return eta_200,alpha_200

'''            
        mles_simulations_1 = minimize(neg_log_likel_unlim,[eta_hat,alpha_hat],args=(person,gamma,kappa,simulated_decisions),bounds=bnds,method='L-BFGS-B')
        mles_simulations_2 = minimize(neg_log_likel_unlim,[eta_0,alpha_0],args=(person,gamma,kappa,simulated_decisions),bounds=bnds,method='L-BFGS-B')
        #Choosing the mles that have the smallest negative log likelihood:
        if mles_simulations_1.fun < mles_simulations_2.fun:
            eta_200[k] = mles_simulations_1.x[0]
            alpha_200[k] = mles_simulations_1.x[1]
        else:
            eta_200[k] = mles_simulations_2.x[0]
            alpha_200[k] = mles_simulations_2.x[1]
    return eta_200, alpha_200
    
    
 #   simulated_decisions = simulating_decisions_unlim(person,alpha_hat,eta_hat,gamma,kappa)
 #   bnds = ((-inf,inf),(0,inf)) #bounds for eta and alpha when minimizing
 #   x0_unlim = [eta_hat,alpha_hat]
 #   mles_simulations = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(person,gamma,kappa,simulated_decisions),bounds=bnds,method='L-BFGS-B')
 #   print(mles_simulations)
    
  
'''
%%%%%%%%%%%%%%%%%%% plotting the simulated eta and alpha for one person %%%%%%%%%%%%%%%%%%%%%%
'''
path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\mles_unlimited_x0_200_times.csv'
df_unlim = pd.read_csv(path,sep=',',usecols=['ID','eta','alpha','fun','eta_0','alpha_0'])
df_unlim.head()
person=0

e, a = simulating_200(person,df_unlim,1,1)

plt.scatter(a,e,s=10)
plt.yscale('log') #for log scale of eta.
plt.title("MLE of eta and alpha based on 200 simulated trials, unlimited")
plt.xlabel('alpha')
plt.ylabel('eta')
plt.show()


a_5 = np.percentile(a,5)
a_95 = np.percentile(a,95)
#print('90% CI alpha: ',a_5,a_95)
e_5 = np.percentile(e,5)
e_95 = np.percentile(e,95)

print('CI alpha:', a_5,a_95)
print('CI eta',e_5,e_95)



'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% finding CI's for all participants %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''    
#path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\mles_unlimited_x0_200_times.csv'
#df_unlim = pd.read_csv(path,sep=',',usecols=['ID','eta','alpha','fun','eta_0','alpha_0'])
#df_unlim.head()
#csv_file_name_and_loc = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\CI_2x0_200B.csv'
#df_unlim.to_csv(csv_file_name_and_loc)
#mabye save the samples of eta and alpha as well and plot them. need to find a way to do that


#path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\mles_unlimited_x0_200_times.csv'
path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\mles_unlimited_x0_200_times.csv'
df_unlim = pd.read_csv(path,sep=',',usecols=['ID','eta','alpha','fun','eta_0','alpha_0'])
df_unlim.head()    
df_unlim['alpha_5_p'] = np.nan
df_unlim['alpha_95_p'] = np.nan
df_unlim['eta_5_p'] = np.nan
df_unlim['eta_95_p'] = np.nan
df_unlim['eta200'] = np.zeros(76,dtype='<U2') #to save the array containing the 200 eta values
df_unlim['alpha200'] = np.zeros(76,dtype='<U2') 

def ci(person,df):
    #adding columns that will contain the confidene intervals
    e, a = simulating_200(person,df_unlim,1,1)
    df['eta200'][person] = e
    df['alpha200'][person] = a
    a_5 = np.percentile(a,5)
    a_95 = np.percentile(a,95)
    #print('90% CI alpha: ',a_5,a_95)
    e_5 = np.percentile(e,5)
    e_95 = np.percentile(e,95)
    #print('90% CI eta: ',e_5,e_95)
    df['alpha_5_p'][person] = a_5
    df['alpha_95_p'][person] = a_95
    df['eta_5_p'][person] = e_5
    df['eta_95_p'][person] = e_95
    return df


#df_unlim = [ci(p,df_unlim) for p in range(76)]

#df_test = ci(0,df_unlim)
#df_test.head()

for p in range(0,76):
    tic = time.perf_counter()
    print(p)
    df_unlim = ci(p,df_unlim) #this is what you can make as a parallell process. 
    toc = time.perf_counter()
    print(f'Code ran in {toc - tic:0.4f} seconds')

csv_file_name_and_loc = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\bootstrap_1000_unlim.csv'
df_unlim.to_csv(csv_file_name_and_loc)
#mabye save the samples of eta and alpha as well and plot them. need to find a way to do that











'''
#trying to use parallell processes:
# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())
#Step 2: pool.apply
df_unlim = [pool.apply(ci,args=(p,df_unlim)) for p in range(3)]
pool.close()

df_ci.head()
'''


    
df_test = ci(0,df_unlim)    
df_test.head()
df_unlim.head()

    '''
    print(person)
    e, a = simulating_200(person,df_unlim,1,1)
    a_5 = np.percentile(a,5)
    a_95 = np.percentile(a,95)
    print('90% CI alpha: ',a_5,a_95)
    e_5 = np.percentile(e,5)
    e_95 = np.percentile(e,95)
    print('90% CI eta: ',e_5,e_95)
    df_unlim['alpha_5_percentile'][person] = a_5
    df_unlim['alpha_95_percentile'][person] = a_95
    df_unlim['eta_5_percentile'][person] = e_5
    df_unlim['eta_95_percentile'][person] = e_95
    '''
#df_unlim.head()
df_unlim[50:65]


tic = time.perf_counter()    
e,a = simulating_200(0,df_unlim,1,1)    
print(a)
print(e)
toc = time.perf_counter()
print(f'Code ran in {toc - tic:0.4f} seconds')


'''
%%%%%%%%%%%% finding percentiles from this %%%%%%%%%%%%%
'''

a_5 = np.percentile(a,5)
a_95 = np.percentile(a,95)
print('90% CI alpha: ',a_5,a_95)

e_5 = np.percentile(e,5)
e_95 = np.percentile(e,95)
print('90% CI eta: ',e_5,e_95)
print(np.sort(e))

plt.scatter(a,e,s=10)
plt.yscale('log') #for log scale of eta.
plt.title("Mle's of simulated alpha and eta, unlimited")
plt.xlabel('alpha')
plt.ylabel('eta')
plt.show()



'''
alpha-eta plot

'''


alpha = df_mles['alpha']
eta = df_mles['eta']
plt.scatter(alpha,eta,s=10)
plt.yscale('log') #for log scale of eta.
plt.title("Mle's of alpha and eta, unlimited")
plt.xlabel('alpha')
plt.ylabel('eta')
plt.show()



'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                plotting the log likelihood for different people
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
'''
#line 11, id=DS4JE30, dtd = 3,5,4, so alpha should not be zero
person = 11
eta = 8.10955857591525
alpha_11 = np.arange(0,0.000000000001,0.00000000000001)
alpha_11 = np.arange(0,0.0000000000001,0.000000000000001)
alpha_11 = np.arange(0,0.0000000000000000001,0.000000000000000000001)
neg_log_l_11 = np.zeros_like(alpha_11)
for i in range(len(alpha_11)):
    neg_log_l_11[i] = neg_log_likel_unlim([eta,alpha_11[i]], data, person,1,1)

plt.plot(alpha_11,neg_log_l_11)
plt.title('Person 11')
plt.show()
'''













'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                                LIMITED TRIALS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

#making code that displayes all the expected losses in each step for each participant
def all_exp_losses_lim(alpha,beta,gamma,kappa):

    #finding the matrix with the expected loss for all possible combinations of red and blue boxes.
    exp_loss_mat = make_matrix_lim(12,alpha,beta,gamma,kappa,True)
    
    
    #the first unlimited trial is the second trial. 
    trial5 = [0,0,0,1,0,0,0,0,1]    #Light blue=0 and Red=1. 
    trial6 = [1,0,0,1,0,0]          #Yellow=0, Light Green=1
    trial7 = [1,0,0,0,1,0,1,0,0]    #Light purple=0, Yellowish white=1
    trial8 = [0,1,0,1,0,1,0,1,0]    #Red = 0, Green = 1
    trial9 = [0,0,1,0,1,0]          #Pink = 0, Green = 1
    trial10 = [0,0,0,1,0,0]         #Blue = 0, Light Yellow = 1

    
    #need to find the expected losses for the case we have here:
    #first initializing
    expL_5 = np.zeros(13,dtype=dict) #these are the exp loss for trial 5.
    #print('expL_5',expL_5)
    expL_6 = np.zeros(13,dtype=dict) #these are the exp loss for trial 6. 
    expL_7 = np.zeros(13,dtype=dict) #these are the exp loss for trial 7.
    expL_8 = np.zeros(13,dtype=dict)
    expL_9 = np.zeros(13,dtype=dict)
    expL_10 = np.zeros(13,dtype=dict)
    
    
    j5=0 #The second entry for the matrix for trial 2:
    j6=0 #the second entry for the matrix for trial 6
    j7=0 #The second entry for the matrix for trial 7
    j8=0
    j9=0
    j10=0
    
    #need to make different for loops for the one with 6 boxes and the ones with 9 boxes. 
    k=0 #the elements in expL_5, expL_6 ... and the first entry in the matrix
    for a in range(10):
        expL_5[k] = exp_loss_mat[k][j5] #finding the expected losses for trial 5 in this situation in the matrix with all the exp losses.
        expL_7[k] = exp_loss_mat[k][j7]
        expL_8[k] = exp_loss_mat[k][j8]
        k+=1 #for finding the next element in expL2,3,4 and going down one row in the matrix
        #i+=1 #going down one row in the loss matrix
        if a!=9:
            if trial5[a] == 1: #if the box that is opened is red, we also go one element to the right in the matrix
                j5+=1
            if trial7[a] == 1: 
                j7+=1
            if trial8[a] == 1: 
                j8+=1

    k=0 #the elements in expL_5, expL_6 ... and the first entry in the matrix
    for a in range(7):
        expL_6[k] = exp_loss_mat[k][j6] #finding the expected losses for trial 6 in this situation in the matrix with all the exp losses.
        expL_9[k] = exp_loss_mat[k][j9] 
        expL_10[k] = exp_loss_mat[k][j10]
        k+=1 #for finding the next element in expL2,3,4 and going down one row in the matrix
        #i+=1 #going down one row in the loss matrix
        if a!=6:
            if trial6[a] == 1: #if the box that is opened is green, we also go one element to the right in the matrix
                j6+=1
            if trial9[a] == 1: 
                j9+=1
            if trial10[a] == 1: 
                j10+=1
        
    all_expL = [expL_5,expL_6,expL_7,expL_8,expL_9,expL_10]
        #print('expL_2',expL_2)
    return all_expL
#dette ser bra ut!
#print(all_exp_losses_lim(0.01,0.8,1,1))
#print(make_matrix_lim(12,0.01,0.8,1,1,True))
    #denne har jeg sjekket mange ganger at er riktig. 



def participant_choices_lim(df,person): #denne er riktig
    all_choices = [[],[],[],[],[],[]]
    neg_l = 0
    for a in range(6): #six trials
        dtd_string = 'BoxNormExtDtD'+str(a+5)
        choice_string = 'BoxNormExtChoice' + str(a+5)
        dtd = df[dtd_string][person]
        choice = df[choice_string][person]
        all_choices[a] = 2*np.ones(dtd,dtype=int)
        
        #find choice as 0,1 or -1, depending on trial
        choice01 = 1
        if (a==0 and choice=='Light blue') or (a==1 and choice=='Yellow') or (a==2 and choice=='Light purple') or (a==3 and choice=='Red') or (a==4 and choice=='Pink') or (a==5 and choice=='Blue'):
            choice01=0
        elif choice == '-1':
            choice01=-1   
        all_choices[a] = np.append(all_choices[a],choice01)
    
    return all_choices


#print(participant_choices_lim(data,2)) #ser bra ut




def neg_log_likel_lim(x,person,gamma,kappa,all_choices):
    eta = x[0]
    alpha = x[1]
    beta = x[2]
    all_expL = all_exp_losses_lim(alpha,beta,gamma,kappa) #array of arrays containing all the expected losses for the 6 trials.
    neg_l = 0
    for trial in range(6): #6 trials
        choices = all_choices[trial] #finding the choices for this particular trial
        expL = all_expL[trial] #finding the exp losses for this trial
        i=0 #the number of boxes that are opened
        for choice in choices:
            if choice != -1: 
                #if choice ==-1: e_delta = beta #if test terminates, the loss is beta
                loss = 'Loss'+str(choice)
                #print(loss)
                e_delta = expL[i][loss] # the expected loss of the choice that is made
                #print(loss, e_delta)
                e0 = expL[i]['Loss0']
                e1 = expL[i]['Loss1']
                e2 = expL[i]['Loss2']
                const = max(-eta*e0,-eta*e1,-eta*e2)
                neg_l += eta*e_delta + const + np.log(np.exp(-eta*e0-const)+np.exp(-eta*e1-const)+np.exp(-eta*e2-const))
                #print(neg_l)
            i+=1
    return neg_l

#neg_log_likel_lim([10,0.01,0.8],0,1,1,participant_choices_lim(data,0)) 
# this give the same value as the old neg log likel

#check if this gives the same for person 75
choices_75 = participant_choices_lim(data,75)
mles_75 = minimize(neg_log_likel_lim,args=(75,1,1,choices_75),x0=[12.99,0.067,0.98],bounds=bnds)
print(mles_75.x)        


'''
sjekke om log likel er riktig:
'''

#this is supposed to be ish 22.1:
person = 15
choices_75 = participant_choices_lim(data,person)
print(neg_log_likel_lim([1,0,0],person,1,1,choices_75))
print(neg_log_likel_lim_old([1,0,0],data,person,1,1))




        
#old_version
#neg_log_likel_lim([10,0.01,0.8],data,0,1,1)
mles_old = minimize(neg_log_likel_lim_old,args=(data,75,1,1),x0=[12.99,0.067,0.98],bounds=bnds)
print(mles_old.x)


def neg_log_likel_lim_old(x,df,person,gamma,kappa): #old, long version
    eta = x[0]
    alpha = x[1]
    beta = x[2]
    #gamma = 1
    #kappa = 1
    
    exp_loss_mat = make_matrix_lim(12,alpha,beta,gamma,kappa,True)
    
    trial5 = [0,0,0,1,0,0,0,0,1] #Light blue = 0, Red = 1
    trial6 = [1,0,0,1,0,0] #Yellow, Light Green
    trial7 = [1,0,0,0,1,0,1,0,0] #Light purple, Yellowish white
    trial8 = [0,1,0,1,0,1,0,1,0] #Red, Green
    trial9 = [0,0,1,0,1,0] #Pink, Green
    trial10 = [0,0,0,1,0,0] #Blue, Light Yellow
    all_trials = [trial5,trial6,trial7,trial8,trial9,trial10]
    
    
    
    #need to find the expected losses for the person here. (maybe do all together in one for loop?)
    #do this for trial 5 first
    
    #exp_losses = [[]] #this is only when there is one trial included
    #for when all trials are included:
    exp_losses = [[],[],[],[],[],[]]
    neg_l = 0
    for a in range(6): #six trials
        trial = all_trials[a]
        dtd_string = 'BoxNormExtDtD'+str(a+5)
        choice_string = 'BoxNormExtChoice' + str(a+5)
        dtd = df[dtd_string][person]
        #print(dtd)
        choice = df[choice_string][person]
        print(choice)
        print(type(choice))
        #find choice as 0,1 or -1, depending on trial
        choice01 = 1
        if (a==0 and choice=='Light blue') or (a==1 and choice=='Yellow') or (a==2 and choice=='Light purple') or (a==3 and choice=='Red') or (a==4 and choice=='Pink') or (a==5 and choice=='Blue'):
            choice01=0
        elif choice == '-1': #dette går ikke, går alrdi gjennom dennne løkka!!
            print(-1)
            choice01=-1
        
        i = 0 #first index in the matrix with losses.
        j = 0 #second index in the matrix with losses.
        
        for l in range(dtd):
            losses = exp_loss_mat[i][j]
            exp_losses[a].append(losses)
            #adding to the neg_l what needs to be added: for when the choice is to open another box.
            e0 = losses['Loss0']
            e1 = losses['Loss1']
            e2 = losses['Loss2']
            const = max(-eta*e0,-eta*e1,-eta*e2)
            print('Loss2',e2)
            neg_l += eta*e2 + const + np.log(np.exp(-eta*e0-const)+np.exp(-eta*e1-const)+np.exp(-eta*e2-const))
            #print(neg_l)
            i += 1 #going one step down in the matrix
            if trial[l]==1: #if red box we go one elemtn to the right in the matrix
                j+=1
  
        #now we are at the last choice: 
        losses = exp_loss_mat[i][j]
        exp_losses[a].append(losses)
        e0 = losses['Loss0']
        e1 = losses['Loss1']
        e2 = losses['Loss2']
        const = max(-eta*e0,-eta*e1,-eta*e2)
        if choice01==0: #If the first colour is chosen as majority colour
            print('Loss0',e0)
            neg_l += eta*e0 + const + np.log(np.exp(-eta*e0-const)+np.exp(-eta*e1-const)+np.exp(-eta*e2-const))
            #print(neg_l)
            #neg_l += eta*losses['Loss0'] + np.log(np.exp(-eta*losses['Loss0'])+np.exp(-eta*losses['Loss1'])+np.exp(-eta*losses['Loss2']))
        elif choice01==1:
            print('Loss1',e1)
            neg_l += eta*e1 + const + np.log(np.exp(-eta*e0-const)+np.exp(-eta*e1-const)+np.exp(-eta*e2-const))
            #print(neg_l)
            #neg_l += eta*losses['Loss1'] + np.log(np.exp(-eta*losses['Loss0'])+np.exp(-eta*losses['Loss1'])+np.exp(-eta*losses['Loss2']))
        #if choice and choice01 is -1, we dont add something to the neg_l, as the particioant was not able to choose then, the test just terminated. 

    return neg_l
    #return neg_l*100
            
#print(make_matrix_lim(12,0.01,0.1,1,1,True))
       
    
#print(neg_log_likel_lim([1,0.01,0.1],data,0))
#neg_log_likel_lim([1,0.01,0.1],data,44,1,1)




#finding mles:
bnds = ((-inf,inf),(0,inf),(0,inf)) #eta, alpha, beta
#x0_lim = [1,0.1,1]
x0_lim = [1,0.5,1]
x0_lim = [mles_lim.x[0],mles_lim.x[1],mles_lim.x[2]]
x0_lim = [5,0,0]
person = 17
#opt = {'ftol':1e-100,'gtol':1e-100,'maxiter':150000,'maxfun':150000}
mles_lim = minimize(neg_log_likel_lim, args=(data,person,0.5,0.5),x0=x0_lim,bounds=bnds)
print(mles_lim)


#sjekk negative likelihood funk for gitte verdier at eta eller beta, da finner du kanskje ut om alpha er null eller ikke
#eta = 1643.980
beta = 1.84007
person = 17
alpha = np.arange(0.07,0.075,0.0005)
eta = np.arange(70,80,0.5)
l = np.zeros((len(alpha),len(eta)))
for i in range(len(alpha)):
    for j in range(len(eta)):
        l[i][j] = neg_log_likel_lim([eta[j],alpha[i],beta],data,person)
    
    #l[i] = neg_log_likel_lim([eta,a,beta],data,person)


plt.imshow(l,extent=(70,80,0.07,0.075),aspect='auto') #alpha on y-axis.
plt.show()


#plotting only one value, only eta
eta = np.arange(50,2000,10)
likel = np.zeros_like(eta)
person=75
alpha=0.0719685
beta=1.84007
for i in range(len(eta)):
    likel[i] = neg_log_likel_lim([eta[i],alpha,beta],data,person)


plt.plot(eta,likel)
plt.show()




'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MLE's for all participants:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
#finding mles for all participants:
df_mles_lim = pd.DataFrame(columns = ['ID','eta','alpha','beta','fun','eta_0','alpha_0','beta_0'], index = np.arange(0,76))
df_mles_lim['fun']=inf
bnds = ((-inf,inf),(0,inf),(0,inf)) #bounds for eta, alpha and beta when minimizing
#x0_lim = [5,0.1,0.5] #start values of eta, alpha and beta
#x0_lim = [0,0.01,0.5] #start values of eta, alpha and beta
#x0_lim = [10,0.01,0.5]
#x0_lim = [0,0,0]
#opt = {'ftol':1e-100,'gtol':1e-100,'maxiter':150000,'maxfun':150000}

path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\mles_limited_test.csv"
df_mles_lim = pd.read_csv(path, usecols = ['ID','eta','alpha','beta','fun','eta_0','alpha_0','beta_0'])
df_mles_lim.head()


'''
%%%%%%%%start here%%%%%%%%%%%%%%%
'''

bnds = ((-inf,inf),(0,inf),(0,inf))

gamma=0.5
kappa=0.5
#for person in range(len(data)):
for person in range(9,76): #start w 377
    ID = data['ID'][person]
    print(person, ID)  
    df_mles_lim['ID'][person] = ID
    choices = participant_choices_lim(data,person)
    for k in range(80):
        eta_0 = 3 + 17*random.random() #random number between 3 and 20
        alpha_0 = 0.1*random.random() #random number between 0 and 0.1
        beta_0 = random.random() #radnom number between 0 and 1
        x0_lim = [eta_0,alpha_0,beta_0]
        mles = minimize(neg_log_likel_lim,x0=x0_lim,args=(person,gamma,kappa,choices),bounds=bnds,method='L-BFGS-B')
        #mles = minimize(neg_log_likel_lim,x0=x0_lim,args=(data,person,1,1),bounds=bnds)
        if mles.fun < df_mles_lim['fun'][person]:
             print('old:',df_mles_lim['fun'][person])
             print('new:',mles.fun)
             df_mles_lim['eta'][person] = mles.x[0]
             df_mles_lim['alpha'][person] = mles.x[1]
             df_mles_lim['beta'][person] = mles.x[2]
             df_mles_lim['fun'][person] = mles.fun
             df_mles_lim['eta_0'][person] = eta_0
             df_mles_lim['alpha_0'][person] = alpha_0
             df_mles_lim['beta_0'][person] = beta_0
             print('new',df_mles_lim['fun'][person])
            
#df_mles_lim.head()
#df_mles_lim[:][5:25]
#df_mles_lim.tail()

path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\mles_limited_fixed_1000.csv"
#path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\mles_limited_test3.csv"
df_mles_lim.to_csv(path)

        
'''
old version, only on x0_lim        
    eta = mles.x[0]
    alpha = mles.x[1]
    beta = mles.x[2]
    print(eta,alpha,beta)
    df_mles_lim.loc[person] = [ID,eta,alpha,beta]
    #df_mles.append({'ID':ID,'eta':eta, 'alpha':alpha},ignore_index=True)
'''



'''
df_mles_lim.head()
df_mles_lim.tail()
df_mles_lim[0:20]
df_mles_lim[20:40]
df_mles_lim[40:60]
df_mles_lim[60:77]




#plotting the paramters 

row = df_mles_lim.loc[df_mles_lim['eta']<0]
row.index
#df_lim_copy=df_mles_lim.drop(df_lim_copy['eta']<0) #dropping the values where
df_lim_copy=df_mles_lim.drop(row.index) #dropping the values where eta is less than zero to be able to take the log 
#but, this is the only one that has alpha different from zero.





alpha = df_lim_copy['alpha']
beta = df_lim_copy['beta']
eta = df_lim_copy['eta']
#one of the elements in eta is negative,hence you cannot take the log of it. 
eta[0:30]



#alpga and eta
plt.scatter(alpha,eta,s=10)
#plt.hist(alpha,np.log(eta))
plt.yscale('log') #for log scale of eta.
plt.title("Mle's of alpha and eta, limited")
plt.xlabel('alpha')
plt.ylabel('eta')
plt.show()


#alpha and beta
plt.scatter(alpha[0:-1],beta[0:-1],s=10) #last participant has beta equal 80, easier to drop this and then compare the other participants
#plt.yscale('log') #for log scale of beta. det går jo ikke siden mange av de er null
plt.title("Mle's of alpha and beta, limited")
plt.xlabel('alpha')
plt.ylabel('Beta')
plt.show()


#eta and beta
plt.scatter(beta[0:-1],eta[0:-1],s=10) #th elast participant has a very high beta, easier to compare the other vlaues if we drop that one
plt.yscale('log') #for log scale of eta.
plt.title("Mle's of beta and eta, limited")
plt.xlabel('beta')
plt.ylabel('eta')
plt.show()
'''



'''
%%%%%%%%%%%%%%%%%%%%%%%%% Bootstrapping %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''



def simulating_decisions_lim(person,eta_hat,alpha_hat,beta_hat,gamma,kappa):
    all_expL = all_exp_losses_lim(alpha_hat,beta_hat,gamma,kappa)
    decisions = [[],[],[],[],[],[]]
    
    for trial in range(6):
        expL = all_expL[trial]
        i=0
        dec=2 #first decision in each trial is to open a box
        while dec==2: 
            if i==6 and (trial == 1 or trial==4 or trial==5):
                decisions[trial].append(-1)
                break
            elif i == 9 and (trial == 0 or trial == 2 or trial == 3):
                decisions[trial].append(-1)
                break
            else: 
                e0 = expL[i]['Loss0']
                e1 = expL[i]['Loss1']
                e2 = expL[i]['Loss2']
                const = min(eta_hat*e0,eta_hat*e1,eta_hat*e2)
                denom = np.exp(-eta_hat*e0+const) + np.exp(-eta_hat*e1+const) + np.exp(-eta_hat*e2+const)
                p0 = np.exp(-eta_hat*e0+const)/denom
                p1 = np.exp(-eta_hat*e1+const)/denom
                p2 = np.exp(-eta_hat*e2+const)/denom
                dec = np.random.choice([0,1,2],p=[p0,p1,p2])
                decisions[trial].append(dec)
            i+=1
    return decisions

#print(simulating_decisions_lim(0,13.16,0,0.26,1,1))        
#print(simulating_decisions_lim(1,26.466,0,0.467,1,1))


#write code that imulates 200 times per person, then find all 200*3 mles, and plot them and find CI's


def simulating_200_lim(person,df,gamma,kappa):
    num_sim=1000
    alpha_200 = np.zeros(num_sim)
    beta_200 = np.zeros(num_sim)
    eta_200 = np.zeros(num_sim)
    
    alpha_hat = df['alpha'][person]
    beta_hat = df['beta'][person]
    eta_hat = df['eta'][person]
    
    alpha_0 = df['alpha_0'][person]
    beta_0 = df['beta_0'][person]
    eta_0 = df['eta_0'][person]
    
    for k in range(num_sim):
        #print('Iteration: ',k)
        simulated_decisions = simulating_decisions_lim(person,eta_hat,alpha_hat,beta_hat,gamma,kappa)
        #print('simulations')
        bnds = ((-inf,inf),(0,inf),(0,inf)) #bounds for eta and alpha when minimizing
        
        mles_simulations_1 = minimize(neg_log_likel_lim,[eta_hat,alpha_hat,beta_hat],args=(person,gamma,kappa,simulated_decisions),bounds=bnds,method='L-BFGS-B')
        #print('mles1')
        mles_simulations_2 = minimize(neg_log_likel_lim,[eta_0,alpha_0,beta_0],args=(person,gamma,kappa,simulated_decisions),bounds=bnds,method='L-BFGS-B')
        
        #Choosing the mles that have the smallest negative log likelihood:
        if mles_simulations_1.fun < mles_simulations_2.fun:
            eta_200[k] = mles_simulations_1.x[0]
            alpha_200[k] = mles_simulations_1.x[1]
            beta_200[k] = mles_simulations_1.x[2]
        else:
            eta_200[k] = mles_simulations_2.x[0]
            alpha_200[k] = mles_simulations_2.x[1]
            beta_200[k] = mles_simulations_2.x[2]
    return eta_200, alpha_200, beta_200



#path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\mles_limited_x0_1000_times.csv'
#df_lim = pd.read_csv(path,sep=',',usecols = ['ID','eta','alpha','beta'])
#df_lim['eta_0'] = 0
#df_lim['alpha_0'] = 0
#df_lim['beta_0'] = 0



'''
%%%%%%%%%%%%% bootstrap for all participants %%%%%%%%%%%%%
'''


path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\mles_limited_fixed_1000.csv"
df_lim = pd.read_csv(path,sep=',',usecols=['ID','eta','alpha','beta','eta_0','alpha_0','beta_0'])
df_lim.head()    
df_lim['eta_5_p'] = np.nan
df_lim['eta_95_p'] = np.nan
df_lim['alpha_5_p'] = np.nan
df_lim['alpha_95_p'] = np.nan
df_lim['beta_5_p'] = np.nan
df_lim['beta_95_p'] = np.nan
df_lim['eta200'] = np.zeros(76,dtype='<U2') #to save the array containing the 200 eta values
df_lim['alpha200'] = np.zeros(76,dtype='<U2') 
df_lim['beta200'] = np.zeros(76,dtype='<U2')



#e,a,b = simulating_200_lim(75,df_lim,1,1)
#print(e)
#print(b)
#print(a)
#path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\bootstrap_limited.csv"
#path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\mles_limited_x0_200_times.csv"
#df_lim.to_csv(path)

'''
df_mles_lim['eta_5_p'] = np.nan
df_mles_lim['eta_95_p'] = np.nan
df_mles_lim['alpha_5_p'] = np.nan
df_mles_lim['alpha_95_p'] = np.nan
df_mles_lim['beta_5_p'] = np.nan
df_mles_lim['beta_95_p'] = np.nan
df_mles_lim['eta200'] = np.zeros(76,dtype='<U2') #to save the array containing the 200 eta values
df_mles_lim['alpha200'] = np.zeros(76,dtype='<U2') 
df_mles_lim['beta200'] = np.zeros(76,dtype='<U2')
df_mles_lim.tail()

for person in range(75,76): #start with 15 next time
    tic = time.perf_counter()
    print(person)
    e,a,b = simulating_200_lim(person,df_mles_lim,1,1)
    df_mles_lim['eta200'][person] = e
    df_mles_lim['alpha200'][person] = a
    df_mles_lim['beta200'][person] = b

    df_mles_lim['eta_5_p'][person] = np.percentile(e,5)
    df_mles_lim['eta_95_p'][person] = np.percentile(e,95)
    df_mles_lim['alpha_5_p'][person] = np.percentile(a,5)
    df_mles_lim['alpha_95_p'][person] = np.percentile(a,95)
    df_mles_lim['beta_5_p'][person] = np.percentile(b,5)
    df_mles_lim['beta_95_p'][person] = np.percentile(b,95)
    toc = time.perf_counter()
    print(f'Code ran in {toc - tic:0.4f} seconds')
'''

'''
%%%%%% hvis du allerede har lagret noen verider %%%%%%%%%%%%%%
'''
#path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\bootstrap_limited_fixed1.csv'
#df_lim = pd.read_csv(path,sep=',',usecols=['ID','eta','alpha','beta','eta_0','alpha_0','beta_0','eta_5_p',
#                                           'eta_95_p','alpha_5_p','alpha_95_p','beta_5_p','beta_95_p','eta200',
#                                           'alpha200','beta200'],dtype={'eta200':'object','alpha200':'object','beta200':'object'})


#bootstrap for alle 75 deltakere: 
for person in range(40,42): 
    tic = time.perf_counter()
    print(person)
    e,a,b = simulating_200_lim(person,df_lim,1,1)
    df_lim['eta200'][person] = e
    df_lim['alpha200'][person] = a
    df_lim['beta200'][person] = b

    df_lim['eta_5_p'][person] = np.percentile(e,5)
    df_lim['eta_95_p'][person] = np.percentile(e,95)
    df_lim['alpha_5_p'][person] = np.percentile(a,5)
    df_lim['alpha_95_p'][person] = np.percentile(a,95)
    df_lim['beta_5_p'][person] = np.percentile(b,5)
    df_lim['beta_95_p'][person] = np.percentile(b,95)
    toc = time.perf_counter()
    print(f'Code ran in {toc - tic:0.4f} seconds')


path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\bootstrap_limited_fixed1000.csv"
df_lim.to_csv(path)



path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\bootstrap_limited_fixed1000.csv'
df_lim = pd.read_csv(path,sep=',',usecols=['ID','eta','alpha','beta','eta_0','alpha_0','beta_0','eta_5_p',
                                           'eta_95_p','alpha_5_p','alpha_95_p','beta_5_p','beta_95_p','eta200',
                                           'alpha200','beta200'],dtype={'eta200':'object','alpha200':'object','beta200':'object'})

for person in range(50,55): 
    tic = time.perf_counter()
    print(person)
    e,a,b = simulating_200_lim(person,df_lim,1,1)
    df_lim['eta200'][person] = e
    df_lim['alpha200'][person] = a
    df_lim['beta200'][person] = b

    df_lim['eta_5_p'][person] = np.percentile(e,5)
    df_lim['eta_95_p'][person] = np.percentile(e,95)
    df_lim['alpha_5_p'][person] = np.percentile(a,5)
    df_lim['alpha_95_p'][person] = np.percentile(a,95)
    df_lim['beta_5_p'][person] = np.percentile(b,5)
    df_lim['beta_95_p'][person] = np.percentile(b,95)
    toc = time.perf_counter()
    print(f'Code ran in {toc - tic:0.4f} seconds')


path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\bootstrap_limited_fixed1000.csv"
df_lim.to_csv(path)



path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\bootstrap_limited_fixed1000.csv'
df_lim = pd.read_csv(path,sep=',',usecols=['ID','eta','alpha','beta','eta_0','alpha_0','beta_0','eta_5_p',
                                           'eta_95_p','alpha_5_p','alpha_95_p','beta_5_p','beta_95_p','eta200',
                                           'alpha200','beta200'],dtype={'eta200':'object','alpha200':'object','beta200':'object'})

for person in range(55,76): 
    tic = time.perf_counter()
    print(person)
    e,a,b = simulating_200_lim(person,df_lim,1,1)
    df_lim['eta200'][person] = e
    df_lim['alpha200'][person] = a
    df_lim['beta200'][person] = b

    df_lim['eta_5_p'][person] = np.percentile(e,5)
    df_lim['eta_95_p'][person] = np.percentile(e,95)
    df_lim['alpha_5_p'][person] = np.percentile(a,5)
    df_lim['alpha_95_p'][person] = np.percentile(a,95)
    df_lim['beta_5_p'][person] = np.percentile(b,5)
    df_lim['beta_95_p'][person] = np.percentile(b,95)
    toc = time.perf_counter()
    print(f'Code ran in {toc - tic:0.4f} seconds')


path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\bootstrap_limited_fixed1000.csv"
df_lim.to_csv(path)



path = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\bootstrap_limited_fixed1000.csv'
df_lim = pd.read_csv(path,sep=',',usecols=['ID','eta','alpha','beta','eta_0','alpha_0','beta_0','eta_5_p',
                                           'eta_95_p','alpha_5_p','alpha_95_p','beta_5_p','beta_95_p','eta200',
                                           'alpha200','beta200'],dtype={'eta200':'object','alpha200':'object','beta200':'object'})

for person in range(60,76): 
    tic = time.perf_counter()
    print(person)
    e,a,b = simulating_200_lim(person,df_lim,1,1)
    df_lim['eta200'][person] = e
    df_lim['alpha200'][person] = a
    df_lim['beta200'][person] = b

    df_lim['eta_5_p'][person] = np.percentile(e,5)
    df_lim['eta_95_p'][person] = np.percentile(e,95)
    df_lim['alpha_5_p'][person] = np.percentile(a,5)
    df_lim['alpha_95_p'][person] = np.percentile(a,95)
    df_lim['beta_5_p'][person] = np.percentile(b,5)
    df_lim['beta_95_p'][person] = np.percentile(b,95)
    toc = time.perf_counter()
    print(f'Code ran in {toc - tic:0.4f} seconds')

path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=0.5\bootstrap_limited_fixed1000.csv"
df_lim.to_csv(path)











'''
#skriv en kode som skriver e tallene du vil ha til en tekstfil.
filename = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Masteroppgave\Code\optimizing_results.txt'
file = open(filename,'a')
string = 'Person: 1, x0='+str(x0_unlim) + ', method: BFGS, eta: '+str(mles.x[0])+', alpha: '+str(mles.x[1])+',  function value: '+str(mles.fun)+'\n'
#print(string)
file.writelines(string)
file.close()




#plot neg log likelihood for person 0 med alpha=0 og forskjellige verdier av eta.
eta_vals = np.arange(32.56563,32.5657,0.000001)
neg_log_l = np.ones_like(eta_vals)
for i in range(len(eta_vals)):
    neg_log_l[i] = neg_log_likel_unlim([eta_vals[i],0])
plt.plot(eta_vals,neg_log_l)
plt.show()

min(neg_log_l)
























for å sjekke hvordan du kan gjøre den beste optimeringen/minimeringen, sjekk ut denne siden:
    https://scipy-lectures.org/advanced/mathematical_optimization/


#for å sjekke hvordan funcsjonen ser ut, bruk scipy.optimize.brute()
#brute(neg_log_likel_unlim,ranges=((-10,10),(0,100)))
#dette ser jo helt løk ut. får alpha mindre en null selv om jeg satt den til å være over null




#plotting the negative log likelihood for different values of eta and alpha
#e = np.arange(-10,10,1)
#a = np.arange(0,2500,250)
e = np.arange(32.56,32.57,0.0001)
#a = np.arange(0,2000,50)
a = np.arange(-0.005,0.005,0.0001)
l = np.zeros((len(e),len(a)))

for i in range(len(e)):
    for j in range(len(a)):
        l[i][j] = neg_log_likel_unlim([e[i],a[j]])
        
plt.imshow(l,extent=[32.56,32.57,-0.005,0.005])
plt.colorbar()
plt.show()


 
plt.pcolormesh(l)
plt.colorbar()
plt.show()
''' 






