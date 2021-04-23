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


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                putting data from participants into a dataframe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
#start by loading the data:
data = pd.read_csv(r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\data1.csv',sep=',')
data = data.rename(columns={'Unnamed: 0':'ID'}) #putting ID as the name of the first column. 





"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            firstly finding the negative loss function in the unlimited case. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def neg_log_likel_unlim(x, df, person,gamma,kappa):
    eta = x[0]
    alpha = x[1]
#    gamma = 1
#    kappa = 1
    
    """
    Starting to find the expected losses for trial 2 which is the first unlimited trial.
    """
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
    
    
    """
    finding the choices that the participant make. firslty only for the second participant in the first unlimited trial (=trial2)
    """
    
    #person=0
    #only finding the data for the first participant now., that is row 0.
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
    #print('all_choices: ',all_choices)
    
    """
    now that we have the exp losses and the choices that the participant make, we can start finding the negative log likelihood
    """
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
                e_delta = 1
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
            
        
    """    
    trial = 0 #dettte må bli en for-løkke når alle tre trials er med
    choices = all_choices[trial]
    expL = all_expL[trial]
    i=0
    for choice in choices:
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
    """

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
result_unlim = {'eta':0,'alpha':0,'f':inf}    
bnds = ((-inf,inf),(0,inf))
for k in range(1000):
    #alpha0 should be a random number between 0 and 0.1:
    alpha_0 = 0.1*random.random()
    #eta_0 schould be a random number between 3 and 20:
    eta_0 = 3 + 17*random.random()
    x0_unlim = [eta_0,alpha_0]
    mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(data,13,1,1),bounds=bnds,method='L-BFGS-B')
    if mles.fun < result_unlim['f']:
        print('old:',result_unlim['f'])
        print('new:',mles.fun)
        result_unlim['eta'] = mles.x[0]
        result_unlim['alpha'] = mles.x[1]
        result_unlim['f'] = mles.fun

print(result_unlim)
toc = time.perf_counter()
print(f'Code ran in {toc - tic:0.4f} seconds')
#mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(data,person),bounds=bnds)
#mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(data,person),bounds=bnds,method='L-BFGS-B')
#opt = {'ftol':1e-100,'gtol':1e-100,'maxiter':1500000,'maxfun':15000000}
#opt = {'ftol':1e-100,'gtol':1e-100,'maxiter':150000,'maxfun':150000}
#mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(data,13,1,1),bounds=bnds,method='L-BFGS-B',options=opt)
#mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(data,person),bounds=bnds,method='BFGS',options={'g-tol':1e-40})
#mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(data,person),bounds=bnds,method='Nelder-Mead')
#print(mles)


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                        finding mles for all participants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

#saving the mles in a dataframe
df_mles = pd.DataFrame(columns = ['ID','eta','alpha','fun'], index = np.arange(0,76))
df_mles['fun'] = inf
#df_mles.head()

bnds = ((-inf,inf),(0,inf)) #bounds for eta and alpha when minimizing
#x0_unlim = [0,0.01] #start values of eta and alpha
#opt = {'ftol':1e-100,'gtol':1e-100,'maxiter':150000,'maxfun':150000}

#for person in range(len(data)):
for person in range(4,len(data)):
    ID = data['ID'][person]
    print(person, ID)    
    df_mles['ID'][person] = ID
    for k in range(1000):
    #alpha0 should be a random number between 0 and 0.1:
        alpha_0 = 0.1*random.random()
        #eta_0 schould be a random number between 3 and 20:
        eta_0 = 3 + 17*random.random()
        x0_unlim = [eta_0,alpha_0]
        mles = minimize(neg_log_likel_unlim,x0=x0_unlim,args=(data,person,1,1),bounds=bnds,method='L-BFGS-B')
        if mles.fun < df_mles['fun'][person]:
            print('old:',df_mles['fun'][person])
            print('new:',mles.fun)
            df_mles['eta'][person] = mles.x[0]
            df_mles['alpha'][person] = mles.x[1]
            df_mles['fun'][person] = mles.fun
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

#try saving this datafram in excel file:
excel_file_name_and_loc = r'C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\mles_unlimited_x0_1000_times.xlsx'
df_mles.to_excel(excel_file_name_and_loc)





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




person= 10
eta = 6.03855720281903
#plottet ble bare en rett linje
alpha = np.arange(0,0.0000000000001,0.000000000000001)
alpha = np.arange(0,0.00000000000001,0.0000000000000001)
neg_log_l = np.zeros_like(alpha)
for i in range(len(alpha)):
    neg_log_l[i] = neg_log_likel_unlim([eta,alpha[i]], data, person,1,1)

plt.plot(alpha,neg_log_l)
plt.title('Person 10')
plt.show()


person= 13
eta = 2.51175344860878
likel, expL = neg_log_likel_unlim([eta,0.01],data,person)
print(expL)
#plottet ble bare en rett linje
alpha = np.arange(0,0.00000001,0.000000000001)

#alpha = np.arange(0,0.00000000000001,0.0000000000000001)
neg_log_l = np.zeros_like(alpha)
for i in range(len(alpha)):
    neg_log_l[i] = neg_log_likel_unlim([eta,alpha[i]], data, person,1,1)

plt.plot(alpha,neg_log_l)
plt.title('Person 13')
plt.show()



person= 5
eta = 14.257237716594
#plottet ble bare en rett linje
alpha = np.arange(0,1,0.001)
alpha = np.arange(0,0.00001,0.0000001)
neg_log_l = np.zeros_like(alpha)
for i in range(len(alpha)):
    neg_log_l[i] = neg_log_likel_unlim([eta,alpha[i]], data, person,1,1)

plt.plot(alpha,neg_log_l)
plt.title('Person 5')
plt.show()



person = 14
eta = 6.52001129020757
alpha = np.arange(0,1,0.001)
alpha = np.arange(0,0.0000001,0.000000001)
neg_log_l = np.zeros_like(alpha)
for i in range(len(alpha)):
    neg_log_l[i] = neg_log_likel_unlim([eta,alpha[i]], data, person,1,1)

plt.plot(alpha,neg_log_l)
plt.title('Person 14')
plt.show()


person = 44
eta = 143.842177675026
alpha = np.arange(0.14,0.18,0.001)
alpha = np.arange(0,0.0000001,0.000000001)
neg_log_l = np.zeros_like(alpha)
for i in range(len(alpha)):
    neg_log_l[i] = neg_log_likel_unlim([eta,alpha[i]], data, person,1,1)

plt.plot(alpha,neg_log_l)
plt.title('Person 44')
plt.show()
'''











'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                                LIMITED TRIALS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''


def neg_log_likel_lim(x,df,person,gamma,kappa):
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
    for a in range(6):
        trial = all_trials[a]
        dtd_string = 'BoxNormExtDtD'+str(a+5)
        choice_string = 'BoxNormExtChoice' + str(a+5)
        dtd = df[dtd_string][person]
        #print(dtd)
        choice = df[choice_string][person]
        #print(choice)
        #find choice as 0,1 or -1, depending on trial
        choice01 = 1
        if (a==0 and choice=='Light blue') or (a==1 and choice=='Yellow') or (a==2 and choice=='Light purple') or (a==3 and choice=='Red') or (a==4 and choice=='Pink') or (a==5 and choice=='Blue'):
            choice01=0
        elif choice == -1:
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
            neg_l += eta*e2 + const + np.log(np.exp(-eta*e0-const)+np.exp(-eta*e1-const)+np.exp(-eta*e2-const))
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
            neg_l += eta*e0 + const + np.log(np.exp(-eta*e0-const)+np.exp(-eta*e1-const)+np.exp(-eta*e2-const))
            #neg_l += eta*losses['Loss0'] + np.log(np.exp(-eta*losses['Loss0'])+np.exp(-eta*losses['Loss1'])+np.exp(-eta*losses['Loss2']))
        elif choice01==1:
            neg_l += eta*e1 + const + np.log(np.exp(-eta*e0-const)+np.exp(-eta*e1-const)+np.exp(-eta*e2-const))
            #neg_l += eta*losses['Loss1'] + np.log(np.exp(-eta*losses['Loss0'])+np.exp(-eta*losses['Loss1'])+np.exp(-eta*losses['Loss2']))
        #if choice and choice01 is -1, we dont add something to the neg_l, as the particioant was not able to choose then, the test just terminated. 

    return neg_l
    #return neg_l*100
            
#print(make_matrix_lim(12,0.01,0.1,1,1,True))
       
    
#print(neg_log_likel_lim([1,0.01,0.1],data,0))
neg_log_likel_lim([1,0.01,0.1],data,44,1,1)




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
df_mles_lim = pd.DataFrame(columns = ['ID','eta','alpha','beta','fun'], index = np.arange(0,76))
df_mles_lim['fun']=inf
bnds = ((-inf,inf),(0,inf),(0,inf)) #bounds for eta, alpha and beta when minimizing
#x0_lim = [5,0.1,0.5] #start values of eta, alpha and beta
#x0_lim = [0,0.01,0.5] #start values of eta, alpha and beta
#x0_lim = [10,0.01,0.5]
#x0_lim = [0,0,0]
#opt = {'ftol':1e-100,'gtol':1e-100,'maxiter':150000,'maxfun':150000}

for person in range(3,len(data)):
#for person in range(len(data)):
    ID = data['ID'][person]
    print(person, ID)  
    df_mles_lim['ID'][person] = ID
    for k in range(1000):
        eta_0 = 3 + 17*random.random() #random number between 3 and 20
        alpha_0 = 0.1*random.random() #random number between 0 and 0.1
        beta_0 = random.random() #radnom number between 0 and 1
        x0_lim = [eta_0,alpha_0,beta_0]
        mles = minimize(neg_log_likel_lim,x0=x0_lim,args=(data,person,1,1),bounds=bnds,method='L-BFGS-B')
        #mles = minimize(neg_log_likel_lim,x0=x0_lim,args=(data,person,1,1),bounds=bnds)
        if mles.fun < df_mles_lim['fun'][person]:
             print('old:',df_mles_lim['fun'][person])
             print('new:',mles.fun)
             df_mles_lim['eta'][person] = mles.x[0]
             df_mles_lim['alpha'][person] = mles.x[1]
             df_mles_lim['beta'][person] = mles.x[2]
             df_mles_lim['fun'][person] = mles.fun
             print('new',df_mles_lim['fun'][person])
            


path = r"C:\Users\Johan\OneDrive\Documents\Masteroppgave\Data\Gamma=kappa=1\mles_limited_x0_1000_times.csv"
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






