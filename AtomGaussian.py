# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:21:17 2021

@author: THINKPAD
"""
import numpy as np
from sympy.physics.wigner import gaunt
import scipy as sp
from sympy import *
from scipy import special
from sympy.physics.wigner import clebsch_gordan

import matplotlib.pyplot as plt


class AtomGaussian(): # give a system which incoulde the initial condition
    
    def __init__(self):
        self.centre= np.array([0.0, 0.0, 0.0])
        self.alpha = 0.0  #Gaussion paprameter
        self.volume = 0.0
        self.weight = 2.828427150
        self.n = 0 # number of parents gaussians for this guassian function, 
                        # used for recording overlap gaussian information
        self.sh_overlap = 0
        #!!! set all initial orbital as p_z orbital
        self.l = 1 # orbital angular momentum
        self.m = 0 # magnetic quantum number
        self.axis = np.zeros(3)
        self.n_qm = 1 #!!! principle quantum number = 1 for now
        
        
def atomIntersection(a = AtomGaussian(),b = AtomGaussian()):
    
    '''the centre of the overlap is still the same as before
        the volume has been changed to the SHoverlap
        the weight of the overlap is not changed
        the alpha value have changed accroding to the overlap volume'''
    c = AtomGaussian()
    
    #trade the overlap volume as a guassian
    c.l = 0
    c.m = 0
    
    #centre 
    c.centre = (a.alpha * a.centre + b.alpha * b.centre)/(a.alpha + b.alpha);  
    
    #gradient = SHoverlap(a,b)[1]
    
    #intersection_volume
    d = a.centre - b.centre
    d_sqr = d.dot(d)  #The distance squared between two gaussians
     
    c.weight = a.weight * b.weight #* np.exp(- a.alpha * b.alpha/c.alpha * d_sqr)  
                                    #!!! ingore this part for now to ez the calculation of alpha
    #c.volume = c.weight * (np.pi/c.alpha) ** 1.5
    c.volume = SHoverlap(a,b)[0]
    
    if abs(c.volume) == 0: 
        return c
    #c.alpha = a.alpha + b.alpha
    #!!! here we use the overlap volume to set the alpha value
    c.alpha = np.pi * (c.weight/c.volume)**(2/3)
         
    # Set the numer of atom of the overlap gaussian
    c.n = a.n + b.n
    
    return  c

def Normalize(alpha,l):
    Norm = np.sqrt(2*(2*alpha)**(l + 3/2)/special.gamma(l+ 3/2))
    return Norm

'''def SHoverlap(a = AtomGaussian(), b = AtomGaussian()):
   
    R = b.centre - a.centre
    radius2 = R.dot(R)
    radius = np.sqrt(radius2)
    xi = a.alpha * b.alpha /(a.alpha + b.alpha)
    lague_x = xi*radius2
    
    

    
    l1 = a.l
    l2 = b.l
    
    m1 = a.m
    m2 = b.m

    I = 0
    F = 0
                         
    m = m2 - m1

    # for one centre overlap integrals
    if radius == 0:
        if l1 == l2 and  m1 == m2: 
            I = (-1)**l2 * special.gamma(l2+3/2)* (4*xi)**(l2+3/2) /(2*(2*np.pi)**(3/2))
    else:
    # for two centre overlap integrals
        
        theta   =  np.arccos(R[2]/radius)
        phi     =  np.arctan2(R[1],R[0])
        
        # set the range of theta and phi for 
        if theta < 0:
            theta = theta + 2*np.pi
        if phi < 0:
            phi = phi + 2*np.pi
            
            # use the selection rule to 
        lset = []
        for value in range(abs(l1-l2),l1+l2+1):
            if (l1+l2+ value) %2 == 0:
                lset.append(value)
 
        # Sum I for each L
        for l in lset:    
            if abs(m) > l: continue
        
            # Calculate the overlap
            n             = (l1+l2-l)/2
            C_A_nl        = 2**n * np.math.factorial(n) * (2*xi)**(n+l+3/2)
            Laguerre      = special.assoc_laguerre(lague_x, n, l+1/2)
            SolidHarmonic = radius**l * special.sph_harm(m, l, phi, theta)
            Psi_xi_R      = np.exp(-lague_x)*Laguerre* SolidHarmonic   
            gaunt_value   = float((-1.0)**m2 *  gaunt(l2,l1,l,-m2,m1,m))
            
            I             += (-1)**n * gaunt_value * C_A_nl * Psi_xi_R
            
            #calculate the gradient 
            overGrad = Gradient(n,l,m,xi,R)
            moment = np.cross(b.centre,overGrad)
            overGrad= np.hstack((overGrad,moment))

            F += (-1)**n * gaunt_value * C_A_nl * overGrad
                    
            
            #calculate the Hessian
            C_A_l_np      = 2**(n+1) * np.math.factorial(n+1) * (2*xi)**(n+1+l+3/2)
            Laguerre_np   = special.assoc_laguerre(lague_x, n+1, l+1/2)
            Psi_xi_R_np   = np.exp(-lague_x)*Laguerre_np* SolidHarmonic
            
            K             += (-1)**n * gaunt_value * C_A_l_np * Psi_xi_R_np
            
            
           
    # Normalized version               
    #S = (-1.0)**l2 * (2*np.pi)**(3/2)* Normalize(1/(4*a.alpha),l1)* Normalize(1/(4*b.alpha),l2)*I
    S = (-1.0)**l2 * (2*np.pi)**(3/2)* I
    #S = float((-1.0)**l2 * (2*np.pi)**(3/2)* I)
    Grad_S = (-1.0)**l2 * (2*np.pi)**(3/2)* F
    #Hess_S = (-1.0)**l2 * (2*np.pi)**(3/2)* K
    return S, Grad_S'''

def SHoverlap(a = AtomGaussian(), b = AtomGaussian()):
   
    R = b.centre - a.centre
    radius2 = R.dot(R)
    radius = np.sqrt(radius2)
    xi = a.alpha * b.alpha /(a.alpha + b.alpha)
    lague_x = xi*radius2
    
    Rot = np.zeros([3,3]).astype(complex) 
    Rot[0,0] = -1/np.sqrt(2)
    Rot[1,0] = 1/np.sqrt(2)
    Rot[0,1] = Rot[1,1] = complex(0,1/np.sqrt(2))                                      
    Rot[2,2] = 1
    
    Ax = np.matmul(Rot,a.axis)
    Bx = np.matmul(Rot,b.axis)

    axis_mat = np.outer(Ax,Bx)

    l1 = a.l
    l2 = b.l
    
    m1 = a.m
    m2 = b.m
        
    I_map = np.zeros([3,3]).astype(complex)
    mset = [+1,-1,0]
    for i in range(3):
        for j in range(3):
            m1 = mset[i]
            m2 = mset[j]

            I = 0
            F = 0
                                 
            m = m2 - m1
        
            # for one centre overlap integrals
            if radius == 0:
                if l1 == l2 and  m1 == m2: 
                    I = (-1)**l2 * special.gamma(l2+3/2)* (4*xi)**(l2+3/2) /(2*(2*np.pi)**(3/2))
            else:
            # for two centre overlap integrals
                
                theta   =  np.arccos(R[2]/radius)
                phi     =  np.arctan2(R[1],R[0])
                
                # set the range of theta and phi for 
                if theta < 0:
                    theta = theta + 2*np.pi
                if phi < 0:
                    phi = phi + 2*np.pi
                    
                    # use the selection rule to 
                lset = []
                for value in range(abs(l1-l2),l1+l2+1):
                    if (l1+l2+ value) %2 == 0:
                        lset.append(value)
         
                # Sum I for each L
                for l in lset:    
                    if abs(m) > l: continue
                
                    # Calculate the overlap
                    n             = (l1+l2-l)/2
                    C_A_nl        = 2**n * np.math.factorial(n) * (2*xi)**(n+l+3/2)
                    Laguerre      = special.assoc_laguerre(lague_x, n, l+1/2)
                    SolidHarmonic = radius**l * special.sph_harm(m, l, phi, theta)
                    Psi_xi_R      = np.exp(-lague_x)*Laguerre* SolidHarmonic   
                    gaunt_value   = float((-1.0)**m2 *  gaunt(l2,l1,l,-m2,m1,m))
                    
                    I             += (-1)**n * gaunt_value * C_A_nl * Psi_xi_R
                    
                    #calculate the gradient 
                    overGrad = Gradient(n,l,m,xi,R)
                    moment = np.cross(b.centre,overGrad)
                    overGrad= np.hstack((overGrad,moment))
        
                    F += (-1)**n * gaunt_value * C_A_nl * overGrad
                            
                    '''
                    #calculate the Hessian
                    C_A_l_np      = 2**(n+1) * np.math.factorial(n+1) * (2*xi)**(n+1+l+3/2)
                    Laguerre_np   = special.assoc_laguerre(lague_x, n+1, l+1/2)
                    Psi_xi_R_np   = np.exp(-lague_x)*Laguerre_np* SolidHarmonic
                    
                    K             += (-1)**n * gaunt_value * C_A_l_np * Psi_xi_R_np
                    '''            
            I_map[i,j] = I
    
    result = axis_mat * I_map
    resum = result.sum()

    # Normalized version               
    #S = (-1.0)**l2 * (2*np.pi)**(3/2)* Normalize(1/(4*a.alpha),l1)* Normalize(1/(4*b.alpha),l2)*I
    S = (-1.0)**l2 * (2*np.pi)**(3/2)* resum
    Grad_S = (-1.0)**l2 * (2*np.pi)**(3/2)* F
    #Hess_S = (-1.0)**l2 * (2*np.pi)**(3/2)* K
    return np.real(S), Grad_S

def Gradient(n, l, m, alpha, r_vec):
    '''The function takes input n, l, m from the resulted overlap function
        AKA Phi^b_nlm'''
    #transform to spherical polar coordinate
    r2 = r_vec.dot(r_vec)
    r = np.sqrt(r2)
    theta   =  np.arccos(r_vec[2]/r)
    phi     =  np.arctan2(r_vec[1],r_vec[0])
    
    if theta < 0:
        theta = theta + 2*np.pi
    if phi < 0:
        phi = phi + 2*np.pi
    
    #calculate functions that only related to the radius
    exp = np.exp(-alpha*r2)
    f   = r**l *exp* special.assoc_laguerre(alpha*r2, n, l+1/2)
    df  = (l/r - 2*alpha*r) * f \
          - 2*alpha* exp *r**(l+1) * special.assoc_laguerre(alpha*r2, n-1, l+3/2)
   
    
    #This part is same for all direction, so evalute outside the loop
    F_plus = df - l*f/r
    F_minus = df + (l+1)*f/r
          
    #Define the transformation matrix from spherical basis to cartizian basis
    U = np.zeros([3,3]).astype(complex)   
    U[0,0] = -1/np.sqrt(2)
    U[0,1] = 1/np.sqrt(2)
    U[1,0] = U[1,1] = complex(0,1/np.sqrt(2)) #!!!The sign should be minus
                                            #But plus give the right answer?
    U[2,2] = 1
    
    G_spherical = np.zeros(3).astype(complex)
    G_plus = np.zeros(3).astype(complex)
    G_minus = np.zeros(3).astype(complex)
    G_coi = np.empty([3,2])
    it = 0
    
    #mu = np.array([1,-1,0])
    for mu in [+1,-1,0]:  
        G_coi[it,0] = C_p(l,m,mu)
        G_coi[it,1] = C_m(l,m,mu)
        
        G_plus[it] = G_coi[it,0]* np.nan_to_num(special.sph_harm(m+mu,l+1, phi, theta))
                
        G_minus[it] = G_coi[it,1]* np.nan_to_num(special.sph_harm(m+mu,l-1, phi, theta)) 
        it+=1
        
    G_spherical = G_plus* F_plus + G_minus* F_minus
                              
    G = np.real(np.matmul(U,G_spherical))
    return G #G_spherical #, G_coi, f, df


def C_p(l,m,mu):
    return np.sqrt((l+1)/(2*l+3)) * clebsch_gordan(l,1,l+1,m,mu,m+mu)

def C_m(l,m,mu):
    return - np.sqrt(l/(2*l-1)) * clebsch_gordan(l,1,l-1,m,mu,m+mu)



