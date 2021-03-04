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
        self.l = 1 # orbital angular momentum
        self.m = 0 # magnetic quantum number
        self.n_qm = 1 #!!! principle quantum number = 0 for now
        
        
def atomIntersection(a = AtomGaussian(),b = AtomGaussian()):
    
    c = AtomGaussian()
    c.alpha = a.alpha + b.alpha

    #centre 
    c.centre = (a.alpha * a.centre + b.alpha * b.centre)/c.alpha; 
    
    c.sh_overlap = SHoverlap(a,b)
    
    #intersection_volume
    d = a.centre - b.centre
    d_sqr = d.dot(d)  #The distance squared between two gaussians
     
    c.weight = a.weight * b.weight * np.exp(- a.alpha * b.alpha/c.alpha * d_sqr)  
    
    c.volume = c.weight * (np.pi/c.alpha) ** 1.5
    #if a.n and b.n ==1:
         
    # Set the numer of atom of the overlap gaussian
    c.n = a.n + b.n
    ratio = c.volume/ ((np.pi/a.alpha)**1.5 * a.weight)
    
    return  ratio

def Normalize(alpha,l):
    Norm = np.sqrt(2*(2*alpha)**(l + 3/2)/special.gamma(l+ 3/2))
    return Norm


def SHoverlap(a = AtomGaussian(), b = AtomGaussian()):
   
    R = b.centre - a.centre
    radius2 = R.dot(R)
    radius = np.sqrt(radius2)
    xi = a.alpha * b.alpha /(a.alpha + b.alpha)
    lague_x = xi*radius2

    
    l1 = a.l
    l2 = b.l
    
    m1set = np.arange(-l1,l1+1,1)
    m2set = np.arange(-l2,l2+1,1)
    #m1 = a.m
    #m2 = b.m


     
    I = 0
    F = 0
    K = 0
    
    for m1 in m1set:
        m1 = m1.item()
        for m2 in m2set:
            
            m2 = m2.item()
            if m1!=0 or m2 !=0: continue
            #if m1 != m2: continue # When sum the P orbital, overlap with 
                                  # different orbital cancled, need this condition
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
                    overGrad      = Gradient(n,l,m,xi,R)
                    
                    F             += (-1)**n * gaunt_value * C_A_nl * overGrad
                    
                    #calculate the Hessian
                    C_A_l_np      = 2**(n+1) * np.math.factorial(n+1) * (2*xi)**(n+1+l+3/2)
                    Laguerre_np   = special.assoc_laguerre(lague_x, n+1, l+1/2)
                    Psi_xi_R_np   = np.exp(-lague_x)*Laguerre_np* SolidHarmonic
                    
                    K             += (-1)**n * gaunt_value * C_A_l_np * Psi_xi_R_np
                    
                    
                    ''' for use of sp overlap
                    
                    if m2 ==-1:
                        z = complex(1,-1)*np.sqrt(1/2)
                        I += z*(-1)**n * gaunt_value * C_A_nl * Psi_xi_R   
                    if m2 == 1:
                        z = complex(1,1)*np.sqrt(1/2)
                        I -= z*(-1)**n * gaunt_value * C_A_nl * Psi_xi_R 
                        
                    if m2 == 0:
                        I += (-1)**n * gaunt_value * C_A_nl * Psi_xi_R
                     '''  
    # Normalized version               
    #S = (-1.0)**l2 * (2*np.pi)**(3/2)* Normalize(1/(4*a.alpha),l1)* Normalize(1/(4*b.alpha),l2)*I
    
    S = (-1.0)**l2 * (2*np.pi)**(3/2)* I
    Grad_S = (-1.0)**l2 * (2*np.pi)**(3/2)* F
    Hess_S = (-1.0)**l2 * (2*np.pi)**(3/2)* K

    return S

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
    f = r**l *exp* special.assoc_laguerre(alpha*r2, n, l+1/2)
    df = (l/r - 2*alpha*r) * f \
        - 2*alpha* exp *r**(l+1) * special.assoc_laguerre(alpha*r2, n-1, l+3/2)
    #df = l* r**(l-1) * special.assoc_laguerre(alpha*r**2, n, l+1/2)\
        #- 2*alpha*r**(l+1)*special.assoc_laguerre(alpha*r**2, n-1, l+3/2)
    
    #This part is same for all direction, so evalute outside the loop
    F_plus = np.sqrt((l+1)/(2*l+3)) * (df - l*f/r)
    F_minus = np.sqrt(l/(2*l-1)) * (df + (l+1)*f/r)
            
    #Define the transformation matrix from spherical basis to cartizian basis
    U = np.zeros([3,3]).astype(complex)   
    U[0,0] = -1/np.sqrt(2)
    U[0,1] = 1/np.sqrt(2)
    U[1,0] = U[1,1] = complex(0,1/np.sqrt(2)) #!!!The sign should be minus
                                            #But plus give the right answer?
    U[2,2] = 1
    
    G_spherical = np.zeros(3).astype(complex)
    it = 0
    for mu in [+1,-1,0]:  
        
        G_plus = clebsch_gordan(l,1,l+1,m,mu,m+mu)\
            * np.nan_to_num(special.sph_harm(m+mu,l+1, phi, theta)) * F_plus
            
        G_minus = clebsch_gordan(l,1,l-1,m,mu,m+mu)\
            * np.nan_to_num(special.sph_harm(m+mu,l-1, phi, theta)) * F_minus
            
        G_spherical[it] = G_plus - G_minus
                        
        it+=1
    G= np.matmul(U,G_spherical)
    return G

def test_F(n, l, m, alpha, r_vec):
    
    r2 = r_vec.dot(r_vec)
    r = np.sqrt(r2)
    theta   =  np.arccos(r_vec[2]/r)
    phi     =  np.arctan2(r_vec[1],r_vec[0])
    if theta < 0:
        theta = theta + 2*np.pi
    if phi < 0:
        phi = phi + 2*np.pi
    f = r**l *np.exp(-alpha*r2)* special.assoc_laguerre(alpha*r2, n, l+1/2)
    Y = np.nan_to_num(special.sph_harm(m,l, phi, theta))
    
    return f*Y
    
        
#%%
R = np.array([0.1,0.5,0.8])
G = Gradient(1,1,0,0.782,R)
#%%
eps = 0.001
d = np.array([0,1,0])
test_G = (test_F(1,1,0,0.782,R + eps*d) - test_F(1,1,0,0.782,R - eps*d))/(2*eps)
#%%
atomA = AtomGaussian()
atomA.alpha = 0.836674050
#atomA.m = 0
atomA.l = 0
atomA.centre = np.array([0.0,   0.0000,   0.0000])

atomB = AtomGaussian()
atomB.alpha = 0.836674050
#atomB.m = -1
atomB.l = 2
atomB.centre = R
Grad = SHoverlap(atomA,atomB)
#%%
R = np.array([0.1,0.5,0.8])
eps = 0.00001
d = np.array([0,1,0])

atomD = AtomGaussian()
atomD.alpha = 0.836674050
atomD.l = 2
atomD.centre = R + eps*d

atomC = AtomGaussian()
atomC.alpha = 0.836674050
atomC.l = 2
atomC.centre = R - eps*d

test_G = (SHoverlap(atomA,atomD) - SHoverlap(atomA,atomC))/(2*eps)

#%%
import seaborn as sns

''' 


for phi in np.linspace(0,2*np.pi,50):
    shstore_angle =np.empty(0)

    for theta in np.linspace(0, np.pi,50):
     ''' 
   
shmap_inside =np.zeros(50)
count = 0
for z in np.linspace(-3,3,50):
    shstore_angle =np.empty(0)
    
    for x in np.linspace(-3,3,50): 
    
        #x =     x**0.5       
        atomA = AtomGaussian()
        atomA.alpha = 0.836674050
        #atomA.m = 0
        atomA.l = 0
        atomA.centre = np.array([0.0,   0.0000,   0.0000])
        
        atomB = AtomGaussian()
        atomB.alpha = 0.836674050
        #atomB.m = -1
        atomB.l = 2
        atomB.centre = np.array([x,0,z])
        #atomB.centre = np.array([1.5 * np.cos(phi)*np.sin(theta),  1.5*np.sin(phi)*np.sin(theta),  1.5 * np.cos(theta)])
        
        sh = SHoverlap(atomA,atomB)
        shstore_angle = np.append(shstore_angle,sh)
        
    if count == 0:
        shmap_inside = np.vstack([shstore_angle])
        
    else:     
        shmap_inside = np.vstack([shmap_inside,shstore_angle])
    count += 1
    
#%%
import cmath
x = np.sqrt(shmap_inside.real**2 + shmap_inside.imag**2)
#shmap += shmap_inside
ax = sns.heatmap(np.real(shmap_inside),center=0)

    #gaussian = atomIntersection(atomA,atomB)
#plt.plot(np.linspace(0, np.pi,10), np.array(shstore_angle),label='psi_0.5, m1_2 = +1')
#plt.plot(np.linspace(0, np.pi,1000), np.array(gaussian_volume),'.',label='gaussian')
plt.xlabel("x")
plt.ylabel("z")
#plt.legend()

#%%
gaussian_volume = []
#for psi in np.linspace(-np.pi,np.pi,100):
psi = 0.5 * np.pi
#for theta in np.linspace(0, np.pi,1000):
#for m1 in [-1,0,1]:
shstore_pm = []
for x in np.linspace(-4,4,100):

    atomA = AtomGaussian()
    atomA.alpha = 0.836674050
    atomA.m = -1
    atomA.l = 1
    atomA.centre = np.array([0.0,   0.0000,   0.0000])
    
    atomB = AtomGaussian()
    atomB.alpha = 0.836674050
    atomB.m = -1
    atomB.l = 1
    atomB.centre = np.array([ x, 0.00 ,0.0000])
    
    sh = SHoverlap(atomA,atomB)
    shstore_pm.append(abs(sh))


    gaussian = atomIntersection(atomA,atomB)
    gaussian_volume.append(gaussian)
x = np.linspace(-4,4,100)
plt.figure()
plt.plot(x, np.array(shstore_pm),'.',label=m1)
plt.show()

plt.legend()

#%%
m1 = 0
shstore = []
gaussian_volume = []
psi = 0.5 * np.pi
for x in np.linspace(0,4,100):

    atomA = AtomGaussian()
    atomA.alpha = 0.836674050
    atomA.m = m1
    atomA.l = 1
    atomA.centre = np.array([0.0,   0.0000,   0.0000])
    
    atomB = AtomGaussian()
    atomB.alpha = 0.836674050
    atomB.m = m1
    atomB.l = 1
    atomB.centre = np.array([  0.0 ,  0.0000, x])
    
    sh = SHoverlap(atomA,atomB)
    shstore.append(sh)

    gaussian = atomIntersection(atomA,atomB)
    gaussian_volume.append(gaussian)
    
x = np.linspace(0,4,100)
plt.plot(x, np.array(shstore),'x',label='m_a = m_b = 0 pp_sigma')# + shstore_neg)/2
#%%
x = np.linspace(0,4,100)
plt.plot(x, np.array(gaussian_volume),'.',label='gaussian')
plt.xlabel("distance")
plt.ylabel("overlap integral")
plt.legend()

#%%
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.wigner import wigner_3j
A = clebsch_gordan(1,1,2,1,-1,0)
print(A)
    