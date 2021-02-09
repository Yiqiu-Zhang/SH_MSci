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

class AtomGaussian(): # give a system which incoulde the initial condition
    
    def __init__(self):
        self.centre= np.array([0.0, 0.0, 0.0])
        self.alpha = 0.0  #Gaussion paprameter
        self.volume = 0.0
        self.weight = 2.828427125
        self.n = 0 # number of parents gaussians for this guassian function, 
                        # used for recording overlap gaussian information
        self.sh_overlap = 0
        self.l = 1 # orbital angular momentum
        self.m = 0 # magnetic quantum number
        self.n_qm = 1 #!!! principle quantum number = 0 for now
        
        
def atomIntersection(a = AtomGaussian(),b = AtomGaussian()):
    
    c = AtomGaussian()
    c.alpha = a.alpha + b.alpha

    #centre 
    c.centre = (a.alpha * a.centre + b.alpha * b.centre)/c.alpha; 
    
    #intersection_volume
    d = a.centre - b.centre
    d_sqr = d.dot(d)  #The distance squared between two gaussians
     
    c.weight = a.weight * b.weight * np.exp(- a.alpha * b.alpha/c.alpha * d_sqr)  
    
    c.volume = c.weight * (np.pi/c.alpha) ** 1.5
    #if a.n and b.n ==1:
        
    #    c.sh_overlap = SHoverlap(a,b)
    
    # Set the numer of atom of the overlap gaussian
    c.n = a.n + b.n
    
    return  c

def Normalize(alpha,l):
    n = np.sqrt(2*(2*alpha)**(l + 3/2)/special.gamma(l+ 3/2))
    return n

def SHoverlap(a = AtomGaussian(), b = AtomGaussian()):
   
    l1 = a.l
    l2 = b.l
    
    m1 = a.m
    m2 = b.m
    

    
    # use the selection rule to 
    lset = []
    l = range(abs(l1-l2),l1+l2+1)
    for value in l:
        if (l1+l2+ value) %2 == 0:
            lset.append(value)
    m = m2 -m1
    
    R = b.centre - a.centre
    radius2 = R.dot(R)
    radius = np.sqrt(radius2)
    xi = a.alpha * b.alpha /(a.alpha + b.alpha)
    I = 0
    
    # for one centre overlap integrals
    if radius == 0:
        if l1 == l2: 
            if m1 == m2: 
                    I = (-1)**l2 * special.gamma(5/2)* (4*xi)**(5/3) /(2*(2*np.pi)**(3/2))

    else:
    # for two centre overlap integrals
        
        theta   =  np.arccos(R[2]/radius)*180/ np.pi #to degrees
        phi     =  np.arctan2(R[1],R[0])*180/ np.pi    
    
        lague_x = xi*radius2
        
        
        for i in range(2):
            
            l = lset[i]
            n = (l1+l2-l)/2
            
            C_A_nl = 2**n * np.math.factorial(n) / (2/(4*xi))**(n+l+3/2)
            Laguerre = special.assoc_laguerre(lague_x, n, l+1/2)
            SolidHarmonic = radius**l * special.sph_harm(m, n, theta, phi)
            Psi_xi_R = np.exp(-lague_x)*Laguerre* SolidHarmonic
            
            gaunt_value =  gaunt(l2,l1,l,-m2,m1,m, prec=15)
            
            I += (-1)**n * (-1.0)**m2 * gaunt_value * C_A_nl * Psi_xi_R 
    S = (-1.0)**l2 * (2*np.pi)**(3/2)* Normalize(1/4*a.alpha,l1)* Normalize(1/4*b.alpha,l2)*I
    
    return S

# 0.8199044671160419, 1 , 2.5
#%%
m1 = -1
shstore = []
for x in np.linspace(-5,5,100):
    testatom1 = AtomGaussian()
    testatom1.alpha = 0.836674025
    testatom1.m = m1
    testatom1.centre = np.array([x,    0.0,    0.0])
    
    testatom2 = AtomGaussian()
    testatom2.alpha = 0.836674025
    testatom2.m = m1
    testatom2.centre = np.array([0.0,    0.0000,    0.0000])
    
    
    sh = SHoverlap(testatom1,testatom2)
    shstore.append(sh)
    #print([m1,m2])
    #print(sh)

gaussian = atomIntersection(testatom1,testatom2)
print(gaussian.volume)