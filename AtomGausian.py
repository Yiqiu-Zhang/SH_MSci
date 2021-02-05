# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:21:17 2021

@author: THINKPAD
"""
import numpy as np
from sympy.physics.wigner import gaunt
import scipy as sp
from sympy import *


class AtomGaussian(): # give a system which incoulde the initial condition
    
    def __init__(self):
        self.centre= np.array([0.0, 0.0, 0.0])
        self.alpha = 0.0  #Gaussion paprameter
        self.volume = 0.0
        self.weight = 2.828427125
        self.l = 1 # orbital angular momentum
        self.m = 0 # magnetic quantum number
        self.n_qm = 0 #!!! principle quantum number = 0 for now 
        self.n = 0 # number of parents gaussians for this guassian function, 
                        # used for recording overlap gaussian information

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
    
    # Set the numer of atom of the overlap gaussian
    c.n = a.n + b.n
    
    return  c

def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  sqrt(x*x + y*y + z*z)
    theta   =  acos(z/r)*180/ pi #to degrees
    phi     =  atan2(y,x)*180/ pi
    return r,theta,phi

def SHoverlap(a = AtomGaussian(), b = AtomGaussian()):
   
    l1 = a.l
    l2 = b.l
    
    m1 = a.m
    m2 = b.m
    
    l = l2+l1
    m = m2 -m1
    n = a.n_qm  #!!! k = 0 for now 
    
    R = b.centre - a.centre
    r,theta, phi = asSpherical(R)
    r2 = r*r
    
    xi = a.alpha * b.alpha /(a.alpha + b.alpha)
    C_A_nl = 2**n * np.factorial(n) / (2/(4*xi))**(n+l+3/2)
    Laguerre = sp.special.assoc_laguerre(xi*r2, n, k= l+0.5)
    SolidHarmonic = r**l * sp.special.sph_harm(m, n, theta, phi)
    
    Psi_xi_R = np.exp(-xi*r2)*Laguerre* SolidHarmonic
    I = gaunt(l1,l2,l,m1,m2,m)*C_A_nl * Psi_xi_R #!!! ignored the sum over l, for now l is constant
    S = (-1)**l2 * (2*np.pi)**(3/2)* I
    
    return S
