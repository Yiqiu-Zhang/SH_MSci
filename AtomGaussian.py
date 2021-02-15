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

import matplotlib.pyplot as plt


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
    ratio = c.volume/ ((np.pi/a.alpha)**1.5 * a.weight)
    
    return  ratio

def Normalize(alpha,l):
    Norm = np.sqrt(2*(2*alpha)**(l + 3/2)/special.gamma(l+ 3/2))
    return Norm

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
    m = m2 - m1
    
    R = b.centre - a.centre
    radius2 = R.dot(R)
    radius = np.sqrt(radius2)
    xi = a.alpha * b.alpha /(a.alpha + b.alpha)
    I = 0
    
    # for one centre overlap integrals
    if radius == 0:
        if l1 == l2: 
            if m1 == m2: 
                    I = (-1)**l2 * special.gamma(l2+3/2)* (4*xi)**(l2+3/2) /(2*(2*np.pi)**(3/2))

    else:
    # for two centre overlap integrals
        
        theta   =  np.arccos(R[2]/radius)*180/ np.pi#0# #to degrees
        phi     =  np.arctan2(R[1],R[0])*180/ np.pi#2*np.pi#    
    
        lague_x = xi*radius2
        
        
        for l in lset:           
            
            n = (l1+l2-l)/2
            
            C_A_nl = 2**n * np.math.factorial(n) * (2*xi)**(n+l+3/2)
            Laguerre = special.assoc_laguerre(lague_x, n, l+1/2)
            SolidHarmonic = radius**l * special.sph_harm(m, l, phi, theta)
            Psi_xi_R = np.exp(-lague_x)*Laguerre* SolidHarmonic    
            gaunt_value = (-1.0)**m2 *  gaunt(l2,l1,l,-m2,m1,m)
            
            I += (-1)**n * gaunt_value * C_A_nl * Psi_xi_R 
            
    S = (-1.0)**l2 * (2*np.pi)**(3/2)* Normalize(1/(4*a.alpha),l1)* Normalize(1/(4*b.alpha),l2)*I
    
    return S

# 0.8199044671160419, 1 , 2.5
#%%

shstore_pm = []
gaussian_volume = []
#for psi in np.linspace(-np.pi,np.pi,50):
psi = 0.5 * np.pi
#for theta in np.linspace(0, np.pi,1000):
for m1 in [-1,0,1]:
    for x in np.linspace(0,4,100):
    
        atomA = AtomGaussian()
        atomA.alpha = 0.836674025
        atomA.m = m1
        atomA.l = 1
        atomA.centre = np.array([0.0,   0.0000,   0.0000])
        
        atomB = AtomGaussian()
        atomB.alpha = 0.836674025
        atomB.m = m1
        atomB.l = 1
        atomB.centre = np.array([0.0,   x,   0.0000])
        
        sh = SHoverlap(atomA,atomB)
        shstore_pm.append(sh)
    
    
        gaussian = atomIntersection(atomA,atomB)
        gaussian_volume.append(gaussian)
    x = np.linspace(0,4,100)
    plt.plot(x, np.array(shstore_pm),'*',label=m1)
    
plt.legend()

#%%
m1 = 0
shstore = []
gaussian_volume = []
psi = 0.5 * np.pi
for x in np.linspace(0,4,100):

    atomA = AtomGaussian()
    atomA.alpha = 0.836674025
    atomA.m = m1
    atomA.l = 1
    atomA.centre = np.array([0.0,   0.0000,   0.0000])
    
    atomB = AtomGaussian()
    atomB.alpha = 0.836674025
    atomB.m = m1
    atomB.l = 1
    atomB.centre = np.array([x,  0.0 ,  0.0000 ])
    
    sh = SHoverlap(atomA,atomB)
    shstore.append(sh)

    gaussian = atomIntersection(atomA,atomB)
    gaussian_volume.append(gaussian)
    
x = np.linspace(0,4,100)
plt.plot(x, np.array(shstore),'x',label='m_a = m_b = 0 pp_sigma')# + shstore_neg)/2
#%%
m1 = 0
shstore_angle = []
#for psi in np.linspace(-np.pi,np.pi,50):

psi = 0.25 * np.pi
for theta in np.linspace(0, np.pi,1000):
    
    atomA = AtomGaussian()
    atomA.alpha = 0.836674025
    atomA.m = m1
    atomA.l = 1
    atomA.centre = np.array([0.0,   0.0000,   0.0000])
    
    atomB = AtomGaussian()
    atomB.alpha = 0.836674025
    atomB.m = m1
    atomB.l = 1
    atomB.centre = np.array([1.5 * np.sin(psi)*np.sin(theta),  1.5*np.cos(psi)*np.sin(theta),  1.5 * np.cos(theta)])
    
    sh = SHoverlap(atomA,atomB)
    shstore_angle.append(sh)

    #gaussian = atomIntersection(atomA,atomB)
plt.plot(np.linspace(0, np.pi,1000), np.array(shstore_angle),label='psi_0.5, m1_2 = +1')
#plt.plot(np.linspace(0, np.pi,1000), np.array(gaussian_volume),'.',label='gaussian')
plt.xlabel("theta")
plt.ylabel("overlap integral")
plt.legend()
#%%
x = np.linspace(0,4,100)
plt.plot(x, np.array(gaussian_volume),'.',label='gaussian')
plt.xlabel("distance")
plt.ylabel("overlap integral")
plt.legend()