# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:55:55 2021

@author: THINKPAD
"""

#%%     

import seaborn as sns
'''Test part for the gradient finite difference method'''
x1 = 1/np.sqrt(2)
z1 = np.sqrt(1- x1**2)#np.sqrt(1- x**2 - y**2)
x2 = 1/np.sqrt(2) *0.5
z2 = np.sqrt(1- x2**2)

axis1 = np.array([x1,0,z1]).astype(complex)
axis2 = np.array([x2,0,z2]).astype(complex)


'''code used to test the SH overlap'''
 
SHmap =np.zeros(50)
count = 0
for z in np.linspace(-3,3,50):
    SHstore =np.empty(0)
    
    for x in np.linspace(-3,3,50): 
       
        atomA = AtomGaussian()
        atomA.alpha = 0.836674050
        atomA.axis = axis1

        atomA.m = 0
        atomA.l = 1
        atomA.centre = np.array([0.0,   0.0000,   0.0000])
        
        atomB = AtomGaussian()
        atomB.alpha = 0.836674050
        atomB.axis = axis2
        
        atomB.m = 0
        atomB.l = 1
        atomB.centre = np.array([x,0,z])
        overlap = SHoverlap(atomA,atomB)[0]
    
        SHstore = np.append(SHstore,overlap)
        
    if count == 0:
        SHmap = np.vstack([SHstore])
        
    else:     
        SHmap = np.vstack([SHmap,SHstore])
    count += 1
 
ax = sns.heatmap(SHmap,center=0)

plt.xlabel("x")
plt.ylabel("z")


#%%

#%%
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

R = np.array([0.1,0.5,0.8])  
eps = 0.001
d = np.array([0,0,1])
test_G = (test_F(1,1,0,0.782,R + eps*d) - test_F(1,1,0,0.782,R - eps*d))/(2*eps)
#%%
'''Test part for the gradient finite difference method'''
R = np.array([0.1,0.5,0.8])

atomA = AtomGaussian()
atomA.alpha = 0.836674050
atomA.m = 0
atomA.l = 0
atomA.centre = np.array([0.0,   0.0000,   0.0000])

atomB = AtomGaussian()
atomB.alpha = 0.836674050
atomB.m = 0
atomB.l = 2
atomB.centre = R
Grad = SHoverlap(atomA,atomB)[1]

eps = 0.00001
d = np.array([0,1,0])

atomD = AtomGaussian()
atomD.alpha = 0.836674050
atomD.l = 2
atomD.m = 0
atomD.centre = R + eps*d

atomC = AtomGaussian()
atomC.alpha = 0.836674050
atomC.l = 2
atomC.m = 0
atomC.centre = R - eps*d

test_G = (SHoverlap(atomA,atomD)[0] - SHoverlap(atomA,atomC)[0])/(2*eps)

#%%
import seaborn as sns

'''code used to test the SH overlap'''

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
'''Test for the gra'''
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
'''
R = np.array([0.1,0.5,0.8])
Hess,df, d2f = Hessian(1,1,0,0.782,R,G_coi)

eps = 0.000001
d = np.array([0,0,1])
R1 = R+ eps*d
R2 = R- eps*d
dr = np.sqrt(R1.dot(R1)) - np.sqrt(R2.dot(R2))
testd2f = (Hessian(1,1,0,0.782,R+ eps*d ,G_coi)[1] - Hessian(1,1,0,0.782,R - eps*d ,G_coi)[1])/ dr
'''
#%%
def Hessian(n, l, m, alpha, r_vec, H_coi):
    
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
    d2f = (-l/r2 - 2*alpha) * f +(l/r - 2*alpha*r) * df\
        - 2*alpha * ((-2*alpha* r**(l+2) +(l+1) * r**l) * exp \
                  *special.assoc_laguerre(alpha*r2, n-1, l+3/2)\
                  -2 * alpha* r * special.assoc_laguerre(alpha*r2, n-2, l+5/2))

    H_pp = d2f - (2*l+1)*df/r + l*(l+2)*f/r2
    H_pm = d2f + 2*df/r - l*(l+1) *f/r2
    H_mp = d2f + 2*df/r - l*(l+1) *f/r2
    H_mm = d2f + (2*l+1)*df/r + (l*l-1)*f/r2
    
    OverHess = np.empty([3,3]).astype(complex)
    out_it = 0
    for mu_out in [+1,-1,0]:   
        in_it = 0
        H_plus = H_coi[out_it,0] # C_p(l,m, mu_out)
        H_minus = H_coi[out_it,1] # C_m(l, m, mu_out)
        for mu_in in [+1,-1,0]: 

            OverHess[in_it,out_it]  = H_plus * (C_p(l+1,m+mu_out,mu_in) * H_pp* np.nan_to_num(special.sph_harm(m+mu_out+mu_in,l+2, phi, theta))\
                                                + C_m(l+1,m+mu_out,mu_in)*H_pm* np.nan_to_num(special.sph_harm(m+mu_out+mu_in,l, phi, theta)))\
                + H_minus * (C_p(l-1,m+mu_out,mu_in) * H_mp * np.nan_to_num(special.sph_harm(m+mu_out+mu_in,l, phi, theta))\
                             + C_m(l-1,m+mu_out,mu_in) * H_mm * np.nan_to_num(special.sph_harm(m+mu_out+mu_in,l-2, phi, theta)))
                    
            in_it += 1
            
        out_it +=1
        
    
    # Add U tomorrow
    #Define the transformation matrix from spherical basis to cartizian basis
    U = np.zeros([3,3]).astype(complex)   
    U[0,0] = -1/np.sqrt(2)
    U[0,1] = 1/np.sqrt(2)
    U[1,0] = U[1,1] = complex(0,1/np.sqrt(2)) #!!!The sign should be minus
                                            #But plus give the right answer?
    U[2,2] = 1
    UT = U.transpose()
    temp = np.matmul(OverHess,UT)
    OverHess = np.matmul(U,temp)
    
    return OverHess, df, d2f
#%%
R = np.array([0.1,0.5,0.6])
G, G_coi, f, df = Gradient(1,1,0,0.782,R)
Hess = Hessian(1,1,0,0.782,R,G_coi)[0]

#%%
eps = 1e-14
d = np.array([0,1,0])
Test1 = np.matmul(Hess,d)
TestHess = (Gradient(1,1,0,0.782,R + eps*d)[0] - Gradient(1,1,0,0.782,R - eps*d)[0])/(2*eps)           