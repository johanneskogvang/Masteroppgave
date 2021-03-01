# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:30:45 2021

@author: Johan
"""



import numpy as np
import scipy.special as sc
from scipy.stats import hypergeom
from itertools import chain



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The following code is for finding the probabilities needed when assuming uniform

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""

#find all the probabilities and losses fro the uniform solution
def zeta(u_i,i,n):
    #zeta(i,u_i,n)=P(U_i=u_i)
    concat_range=chain(range(int(n/2)),range(int(n/2)+1,n+1))
    prob=0
    for j in concat_range:
        prob+=hypergeom.pmf(u_i,n,j,i)
    return prob/n
    
def unif_rho(i,u_i,n):
    #=number of boxes
    #Rho(i,x,n) = P(u_n >= (n/2)+1 | U_i=u_i) = P(red majority given U_i=u_i)
    #If n=12: Rho(i,x,12)=P(u_12 >=7 | U_i=u_i)
    
    #numerator=P(u_n >= (n/2)+1 and u_i=x) = P(majority red and u_i=x)
    nume=0
    for j in range(int((n/2))+1,n+1):
        #P(u_i=x|u_n=j)=hypergeom.pmf(x,n,j,i)
        nume+=hypergeom.pmf(u_i,n,j,i)
    nume=nume/n    
    #denumerator=P(U_i=u_i)
    denume=zeta(u_i,i,n)
    return nume/denume


#making functions to find loss2:
def gamma(k,u_i,i,n):
    #gamma(k,u_i,n)=P(X_i+1 | U_i=u_i)
    #correction: gamma(k,u_i,i,n)=P(X_{i+1}=1|U_n=k,U_i=u_i)
    if u_i>k or k==int(n/2) or (k-u_i)>(n-i):
        return 0
    else :
        return (k-u_i)/(n-i)


def unif_epsilon(u_i,i,n):
    #epsilon(u_i,i,n)=P(X_i+1 = 1 | U_i=u_i)
    prob=0
    b = zeta(u_i,i,n)
    concat_range=chain(range(int(n/2)),range(int(n/2)+1,n+1))
    for k in concat_range:
        prob+=gamma(k,u_i,i,n)*hypergeom.pmf(u_i,n,k,i)
    prob=prob/(n*b)        
    return prob




"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The following code is for finding the probabilities needed when assuming binomial

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""

#def binom_rho(i,u_i,n,gamma,kappa):
#    return 1

#def binom_epsilon(u_i,i,n,gamma,kappa):
#    return 1
def u12equalj(j,i,ui,gamma,kappa):
#P(U_i+V_i = j | U_i=u_i)
    #ui <= j <= 12
    #ui <= i <= 12   
    #j <= 12- (i-ui) 
    if j > ui + 12-i: #cant have more red boxes than ui + the numer of boxes that are left to open
        return 0
    if ui>j:
        return 0
    if j>12:
        raise ValueError('j has to be smaller than or equal to 12')
    if i>12:
        raise ValueError('i has to be smaller than or equal to 12')
    if ui>i:
        raise ValueError('i has to be bigger than or equal to ui')
    if i>12:
        raise ValueError('i has to be smaller than or equal to 12')
    if ui>12:
        raise ValueError('ui has to be smaller than or equal to 12')
    
    
    nume = sc.beta(j+gamma,12-j+kappa) #runtime warning her
    nume = nume*sc.binom(12-i,j-ui)
    denume = sc.beta(ui+gamma,i-ui+kappa)
    return nume/denume

    
def binom_rho(i,ui,n,gamma,kappa):
#def redmajority(i,ui,gamma,kappa):
    #P(U_i+V_i >= 7 | U_i=u_i, U_i+V_i neq 6)
    #ui <= i <= 12 
    if ui>6:
        return 1
    elif ui+12-i<6:
        return 0
       
    nume = 0
    for j in [7,8,9,10,11,12]:
        nume += u12equalj(j,i,ui,gamma,kappa)
    
    denume= 1 - u12equalj(6,i,ui,gamma,kappa)
    
    return nume/denume


    




def nextisred(i,ui,gamma,kappa):
    #P(X_i+1 = 1 | U_i=u_i) = P(X_i+1 = 1)
    return (gamma+ui)/(gamma+kappa+i)

def majority_givennextisred(i,ui,gamma,kappa):
    #P(U_i+V_i neq 6 | U_i=u_i, X_i+1 = 1)
    if ui>i:
        raise ValueError('i has to be bigger than or equal to ui')
    a = sc.binom(11-i,5-ui)
    b = sc.beta(6+gamma,6+kappa)/sc.beta(gamma+ui+1,kappa+i-ui)
    return 1 - (a*b)
   
    
def majority_given_ui(i,ui,gamma,kappa):
    a = sc.binom(12-i,6-ui)
    b = sc.beta(6+gamma,6+kappa)/sc.beta(ui+gamma,i-ui+kappa)
    return 1 - (a*b)

def binom_epsilon(ui,i,n,gamma,kappa):
#def nextisred_givenmajority(i,ui,gamma,kappa):
    #P(X_i+1 = 1 | U_i=u_i, U_i+V_i neq 6)
    return nextisred(i,ui,gamma,kappa)*majority_givennextisred(i,ui,gamma,kappa)/majority_given_ui(i,ui,gamma,kappa)






""""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The following code is for finding the losses and visualising them. This code works for both the uniform and binomial cases

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""

#finding the propabilities that red and blue are majority colours in addition to loss0 and loss1. 
def prob_loss_0_1(n,alpha,gamma,kappa,binom):
    if n%2 != 0 or n==0:
            raise ValueError('Number of boxes must be an even number above zero.')
    
    m = np.zeros((n,n),dtype=dict)
    for i in range(0,n):
            for u_i in range(0,i+1):
                if binom==False:
                    r=unif_rho(i,u_i,n) #Rho(i,x,n) = P(u_n >= (n/2)+1 | U_i=u_i)
                else: #binom==True:
                    r=binom_rho(i,u_i,n,gamma,kappa) #Rho(i,x,n) = P(u_n >= (n/2)+1 | U_i=u_i)
                #r=rho(i,u_i,n) #Rho(i,x,n) = P(u_n >= (n/2)+1 | U_i=u_i)
                L0=r #Expected loss when choosing blue
                L1=1-r #Expected loss when chosing red 
                
                #Loss2:
                if i==(n-1): #if we are at the last row of the tree. 
                    #making the last "Loss2"=alpha
                    m[i,u_i]={"prob":r,"Loss0":L0,"Loss1":L1,"Loss2":alpha}
                else: #If we are anywhere but the kast row of the tree, use the function find_loss2.
                    #don't know the rest of the "loss2" yet, just putting it tp a high value. 
                    m[i,u_i]={"prob":r,"Loss0":L0,"Loss1":L1,"Loss2":1000}
    return m

def loss2_unlim(matrix,n,alpha,gamma,kappa,binom): 
    for i in range(n-2,-1,-1): 
        #looping from the second to last row of the matrix to the first one
        for u_i in range(0,i+1):
            #expected loss if the next one is blue:
            EL_0 = min(matrix[i+1,u_i]["Loss0"],matrix[i+1,u_i]["Loss1"],matrix[i+1,u_i]["Loss2"])
            #expected loss if the next one is red. 
            EL_1 = min(matrix[i+1,u_i+1]["Loss0"],matrix[i+1,u_i+1]["Loss1"],matrix[i+1,u_i+1]["Loss2"])
            if binom == False:
                eps = unif_epsilon(u_i,i,n)
            else: #binom==True
                eps = binom_epsilon(u_i,i,n,gamma,kappa)
                
            matrix[i,u_i]["Loss2"]=alpha + (1-eps)*EL_0 + eps*EL_1    
    return matrix

def loss2_lim(matrix,n,alpha,beta,gamma,kappa,binom):
#def find_loss2(matrix,n,alpha): 
    for i in range(n-2,-1,-1): 
        #looping from the second to last row of the matrix to the first one
        #P(T=i+1|T>i) = 1/(12-i) =  P(get stopped in next | not stopped until now) = PT
        PT = 1/(n-i)
        if i==0: #P(T=1|T>0)=0. means that you are able to open the first box without getting stopped
            PT = 0
        #print(PT)
        for u_i in range(0,i+1):
            #expected loss if the next one is blue:
            EL_0 = min(matrix[i+1,u_i]["Loss0"],matrix[i+1,u_i]["Loss1"],matrix[i+1,u_i]["Loss2"])
            #expected loss if the next one is red. 
            EL_1 = min(matrix[i+1,u_i+1]["Loss0"],matrix[i+1,u_i+1]["Loss1"],matrix[i+1,u_i+1]["Loss2"])
            if binom == False:
                eps = unif_epsilon(u_i,i,n)
            else: #binom==True
                eps = binom_epsilon(u_i,i,n,gamma,kappa)
            matrix[i,u_i]["Loss2"]=alpha + (1-PT)*((1-eps)*EL_0 + eps*EL_1) + PT*beta    
    return matrix
 

def make_matrix_unlim(n,alpha,gamma,kappa,binom):
    matrix=prob_loss_0_1(n,alpha,gamma,kappa,binom)
    matrix=loss2_unlim(matrix,n,alpha,gamma,kappa,binom)
    return matrix

def make_matrix_lim(n,alpha,beta,gamma,kappa,binom):
    matrix=prob_loss_0_1(n,alpha,gamma,kappa,binom)
    matrix=loss2_lim(matrix,n,alpha,beta,gamma,kappa,binom)
    return matrix









#making these matrixes into tikz-trees:
#We have the matrix with the expected losses, the next step is then to find the optimal solution:
#making decision trees using tikz.
def make_node_matrix(matrix):
    node_mat=np.zeros_like(matrix)
    n = len(matrix)
    
    for i in range(n):
        for j in range(i+1):
            l0 = matrix[i][j]["Loss0"]
            l1 = matrix[i][j]["Loss1"]
            l2 = matrix[i][j]["Loss2"]
            e0 = round(l1+l2,14) #=sum of losses - l0 = l1+l2
            e1 = round(l0+l2,14) #= l0+l2
            e2 = round(l0+l1,14) #=l0+l1
            
            col1 = "green!70!black"
            col2 = "green!70!black"
            
            if l0<l2 and round(l0,5)<round(l1,5): #if blue has the smallest loss
                col1 = "blue"
                col2 = "blue"
            elif l1<l2 and round(l1,5)<round(l0,5): #red has the smallest loss
                col1 = "red"
                col2 = "red"
            elif l0<l2 and round(l0,5) == round(l1,5): #red and blue has the smallest loss, but they are equal
                col1 = "blue"
                col2 = "red"
            
        
            name = "N" + str(i) + "-" + str(j)
            #making each entry in the matrix a dict. col1 and col2 is the colour that haas the least expected loss.
            #if there are two colours that has the same loss and that is the least expected loss, then col1 is one of them and col2 is the other colour.
            #e0 is the "inverse" expected loss for choosing blue as the majority colour, e1 for choosing red and
            #e2 is the inverse expected loss for choosing to open another box.
            node_mat[i][j] = {"name":name, "col1":col1, "col2":col2, "e0":e0, "e1":e1, "e2":e2}
    return node_mat

#making tikz code to visualise the IO solution
def visualise_optimal(mat,file_location_and_name, radius):
    file = open(file_location_and_name,"w")

    start_of_doc=r"""
\begin{tikzpicture}[
    treenodeT/.style={
      circle, align=center},
    node distance=1cm,
    ]
    """
    file.write(start_of_doc)
    
    #the first node:
    string = "\DoNode{N0-0}{" + str(mat[0][0]["e0"]) + "}{" + str(mat[0][0]["e1"]) + "}{1}{" + str(mat[0][0]["col1"]) + "}{" + str(mat[0][0]["col2"]) + "}{" + str(radius) + "};\n    "
    file.write(string)
    
    n = len(mat)
    for i in range(1,n):
        
        #to check if we have to break the loop (we have reached a decision in all of the nodes above)
        g=0
        if i>1:
            for j in range(i):
                if str(mat[i-1][j]["col1"]) == "green!70!black": #checing if any of the nodes in the row above are green
                    g=1
            if g == 0: #if none of the nodes on the row above are green, break out of the for loop
                    #break out of the for loop
                print("breaking loop at row", i)
                break
            
        for j in range(i+1):
            
            if j==0: #we are at the left side of the tree. the only possible parent i at (i-1,j)
                if mat[i-1][j]["col1"] == "green!70!black": #if we continue to open boxes in the last node
                    string = "\DoNode[below of=" + mat[i-1][j]["name"] + ", left of= " + mat[i-1][j]["name"] + "]{"+ mat[i][j]["name"] +"}{" + str(mat[i][j]["e0"]) + "}{" + str(mat[i][j]["e1"]) + "}{1}{" + str(mat[i][j]["col1"]) + "}{" + str(mat[i][j]["col2"]) + "}{" + str(radius) + "};\n    "
                    file.write(string)
                    string2 = "\draw[->] (" + str(mat[i-1][j]["name"]) + ") -- (" + mat[i][j]["name"] + ");\n    "
                    file.write(string2)
            elif j==i: #we are at the right side of the tree. the only possible parent is at (i-1,j-1)
                if mat[i-1][j-1]["col1"] == "green!70!black": #if we continue to open boxes in the last node.
                    string = "\DoNode[below of=" + mat[i-1][j-1]["name"] + ", right of= " + mat[i-1][j-1]["name"] + "]{"+ mat[i][j]["name"] +"}{" + str(mat[i][j]["e0"]) + "}{" + str(mat[i][j]["e1"]) + "}{1}{" + str(mat[i][j]["col1"]) + "}{" + str(mat[i][j]["col2"]) + "}{" + str(radius) + "};\n    "
                    file.write(string)
                    string2 = "\draw[->] (" + str(mat[i-1][j-1]["name"]) + ") -- (" + mat[i][j]["name"] + ");\n    "
                    file.write(string2)
            else: #we are not on either side of the tree
                if mat[i-1][j-1]["col1"]=="green!70!black": #if the left top node is a parent
                    string = "\DoNode[below of=" + mat[i-1][j-1]["name"] + ", right of= " + mat[i-1][j-1]["name"] + "]{"+ mat[i][j]["name"] +"}{" + str(mat[i][j]["e0"]) + "}{" + str(mat[i][j]["e1"]) + "}{1}{" + str(mat[i][j]["col1"]) + "}{" + str(mat[i][j]["col2"]) + "}{" + str(radius) + "};\n    "
                    file.write(string)
                    string2 = "\draw[->] (" + str(mat[i-1][j-1]["name"]) + ") -- (" + mat[i][j]["name"] + ");\n    "
                    file.write(string2)
                    if mat[i-1][j]["col1"] == "green!70!black": #if the top right node also is a parent
                        string3 = "\draw[->] (" + str(mat[i-1][j]["name"]) + ") -- (" + mat[i][j]["name"] + ");\n    "
                        file.write(string3)
                elif mat[i-1][j-1]["col1"] != "green!70!black" and mat[i-1][j]["col1"]=="green!70!black": #left is not a parent, but the right is
                    string = "\DoNode[below of=" + mat[i-1][j]["name"] + ", left of= " + mat[i-1][j]["name"] + "]{"+ mat[i][j]["name"] +"}{" + str(mat[i][j]["e0"]) + "}{" + str(mat[i][j]["e1"]) + "}{1}{" + str(mat[i][j]["col1"]) + "}{" + str(mat[i][j]["col2"]) + "}{" + str(radius) + "};\n    "
                    file.write(string)
                    string2 = "\draw[->] (" + str(mat[i-1][j]["name"]) + ") -- (" + mat[i][j]["name"] + ");\n    "
                    file.write(string2)
                    
    end_of_file= r"""
\end{tikzpicture}
"""
    file.write(end_of_file)
    
    file.close()












def main(alpha,beta,unlim=True,binom=True,gamma=1,kappa=1):
    n=12
    node_radius=0.4
    file_loc= "C:\\Users\\Johan\\OneDrive\\Documents\\Masteroppgave\\Masteroppgave\\tikz-trees"
    
    #uniform unlimited
    if unlim==True and binom==False:
        file = file_loc + "\\unif_unlim_a"+str(alpha)+".tex"
        mat_losses = make_matrix_unlim(n,alpha,gamma,kappa,binom=False)
        nodes = make_node_matrix(mat_losses)
        visualise_optimal(nodes,file,node_radius)
        print("Uniform unlimited")
    
    #limted uniform
    if unlim==False and binom==False:
        file = file_loc + "\\unif_lim_a"+str(alpha)+"_b"+str(beta)+".tex"
        mat_losses = make_matrix_lim(n,alpha,beta,gamma,kappa,binom=False)
        nodes = make_node_matrix(mat_losses)
        visualise_optimal(nodes,file,node_radius)
        print("Uniform limited")
    
    #unlimited binomial
    if unlim==True and binom==True:
        file = file_loc + "\\binom_unlim_a"+str(alpha)+"_g"+str(gamma)+ "_k"+str(kappa)+".tex"
        mat_losses = make_matrix_unlim(n,alpha,gamma,kappa,binom=True)
        nodes = make_node_matrix(mat_losses)
        visualise_optimal(nodes,file,node_radius)
        print("Binomial unlimited")
    
    #limited binomial
    if unlim==False and binom==True:
        file = file_loc + "\\binom_lim_a"+str(alpha)+"_b"+str(beta)+"_g"+str(gamma)+ "_k"+str(kappa)+".tex"
        mat_losses = make_matrix_lim(n,alpha,beta,gamma,kappa,binom=True)
        nodes = make_node_matrix(mat_losses)
        visualise_optimal(nodes,file,node_radius)
        print("Binomial limited")
    
    
    
#uniform unlimited
#main(0.01,1,True,False)

#uniform limited
#main(0.01,0.4,False,False)


#unlimited binomial:
main(0.01,0.6,True,True,2,0.5)


#limited binomial:
#main(0.01,0.6,False,True,1,1)