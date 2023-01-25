#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:58:13 2022

@author: cesar
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.special import gamma, factorial
import scipy.integrate as integrate
from scipy import interpolate
import quadpy as quadpy # pip install quadpy
from argparse import Namespace
import warnings


###########################################################
#   JPAC color style
###########################################################

jpac_blue   = "#1F77B4"; jpac_red    = "#D61D28";
jpac_green  = "#2CA02C"; jpac_orange = "#FF7F0E";
jpac_purple = "#9467BD"; jpac_brown  = "#8C564B";
jpac_pink   = "#E377C2"; jpac_gold   = "#BCBD22";
jpac_aqua   = "#17BECF"; jpac_grey   = "#7F7F7F";

jpac_color = [jpac_blue, jpac_red, jpac_green, 
              jpac_orange, jpac_purple, jpac_brown,
              jpac_pink, jpac_gold, jpac_aqua, jpac_grey, 'black' ];

dashes, jpac_axes = 10*'-', jpac_color[10];


''' Libraries '''

def x_and_weights_gausslegendre(domain,deg): 
    su, ad = (domain[1] - domain[0])/2, (domain[1] + domain[0])/2
    x, w = scipy.special.roots_legendre(deg)
    xb = su*x + ad
    return xb, w;


def complex_function(name,arrayin,nuin,nuout,parameter_a,lowernu):
    
    ''' libraries '''    

    ###########################################################################
    #   Interpolation
    #
    def interpolation(x,y): 
        return interpolate.interp1d(x, y,kind='cubic')
    ###########################################################################
    #   Analytic continuation
    #
    def analytic_continuation(f,nu0,lowernu):
        nlen = len(np.real(nu0))
        Integral = np.zeros((nlen,),dtype='complex')
        eps = 0.0005; a, b = np.arctan(lowernu)+eps, np.pi/2.-eps;
        for i in range(nlen):
            part_re = quadpy.quad(f,a,b,args=(nu0[i],0))
            part_im = quadpy.quad(f,a,b,args=(nu0[i],1))
            Integral[i] = (nu0[i]+1.)*(part_re[0]+1j*part_im[0])/np.pi;
        return Integral
    #
    #   alpha
    #
    def alpha_integrand(nu_atan,nu0,ioption):
        nu = np.tan(nu_atan)
        iminter = imalpha_inter(nu)
        integrand = iminter*(nu-1.)*(nu+nu0)/( (nu**2-1)*(nu**2-nu0**2) )/(np.cos(nu_atan)**2)
        if ioption==0:
            output = np.real(integrand)
        else:
            output = np.imag(integrand)
        return output
    
    def alpha_function(nu0,lowernu):
        Integral = analytic_continuation(alpha_integrand,nu0,lowernu)
        return parameter_a + Integral


    ''' alpha calculation
        Input
        f = alpha_function or 
        arrayin: array with the input (same size as nuin)        
        nuin: array with the nu points for the alpha function in the real axis above threshold
        nuout: array with the nu points where you want to compute the alpha function
        parameter_a: parameter a of the model
        lowernu: Minimum value of nu for the Integration range
    '''
    
    if name=='alpha':
        imalpha_inter = interpolation(nuin,np.imag(arrayin))
        arrayout = alpha_function(nuout,lowernu)
    else:
        return None
    
    return arrayout

###############################################################################
#   Iteration
#
def system_solver(nu, alpha, mlambda, m, nlimit_Nhat, free_parameters, lowernu):

    ''' Libraries ''' 
            
    ###########################################################################
    #   Phase space
    #    
    def s_function(nu): return (nu+1.)*m**2;

    def rho_function(nu): return np.sqrt(nu/(nu+1.));

    def rho_integrand(nu_atan,nu0,ioption):
        nu = np.tan(nu_atan)
        integrand = rho_inter(nu)*(nu-1.)*(nu+nu0)/( (nu**2-1)*(nu**2-nu0**2) )/(np.cos(nu_atan)**2)
        if ioption==0:
            output = np.real(integrand)
        else:
            output = np.imag(integrand)
        return output

    ###########################################################################
    #   Interpolation
    #
    def interpolation(x,y): 
        return interpolate.interp1d(x, y,kind='cubic')

    ###########################################################################
    #   Analytic continuation
    #
    def analytic_continuation(f,nu,lowernu):
        Integral = np.zeros((len(nu),),dtype='complex')
        eps = 0.0005; a, b = np.arctan(lowernu)+eps, np.pi/2.-eps;
        for i in range(len(nu)):
            nu0 = nu[i] + iepsilon
            part_re = quadpy.quad(f,a,b,args=(nu0,0))
            part_im = quadpy.quad(f,a,b,args=(nu0,1))
            Integral[i] = (nu0+1.)*(part_re[0]+1j*part_im[0])/np.pi;
        return Integral

    def PV_analytic_continuation(f,nu,lowernu,discontinuity):
        Integral = np.zeros((len(nu),))
        eps = 0.0005; a, b = np.arctan(lowernu)+eps, np.pi/2.-eps;
        for i in range(len(nu)):
            nu0 = nu[i] + iepsilon
            ntop = len(discontinuity) 
            if ntop==0:
                part_re = quadpy.quad(f,a,b,args=(nu0,0))
                Integral[i] =(np.real(nu0)+1.)*part_re[0]/np.pi;
            else:
                arec = np.zeros(ntop+1)
                brec = np.zeros(ntop+1)
                arec[0] = a
                brec[ntop] = b
                for n in range(1,ntop+1):
                    arec[n] = np.arctan(discontinuity[n-1])
                    brec[n-1] = np.arctan(discontinuity[n-1])
                Integral[i] = 0.
                for n in range(ntop+1):
                    part_re = integrate.quad(f,arec[n],brec[n],args=(nu0,0))
                    Integral[i] = Integral[i] + (np.real(nu0)+1.)*part_re[0]/np.pi;
        return Integral

    ###########################################################################
    #   Model
    #
    #   Nhat
    #
    def Nhat_function(nu,alpha,r,nlimit):
        factor0 = (2.**(alpha-1))*np.sqrt(np.pi)*gamma(alpha+2)
        factor1 = ( nu+r )*gamma(alpha+3/2)
        barrier = ( nu/4/(nu+2*r) )**alpha
        suma = factor0/factor1*barrier
        for n in np.arange(1,nlimit+1):
            factor0 = (2.**(alpha-1))*(alpha+1.)*np.sqrt(np.pi)*gamma(alpha+2*n+1)
            factor1 = ( nu+r )*scipy.special.factorial(n)*gamma(alpha+n+3/2)
            barrier = ( nu/4/(nu+2*r) )**(2*n+alpha)
            suma = suma + factor0/factor1*barrier
        return suma;
    #
    #   phi
    #
    def phi_function(cNhat): return np.angle(cNhat);
    
    def discontinuity_phi(nu,cNhat):
        disc = []
        for i in range(1,len(cNhat)):
            phi0 = np.angle(cNhat[i-1])
            phi1 = np.angle(cNhat[i])
            if np.sign(phi0)!=np.sign(phi1) and np.abs(phi0-phi1)>50*np.pi/180:
                disc.append((nu[i-1] + nu[i] )/2.)
        return np.array(disc)
    #
    #   beta
    #
    def beta_integrand(nu_atan,nu0,ioption):
        nu = np.tan(nu_atan)
        cNhat = Nhat_inter_re(nu) + 1j*Nhat_inter_im(nu)
        phi = phi_function(cNhat)
        integrand = phi*(nu-1.)*(nu+nu0)/( (nu**2-1)*(nu**2-nu0**2) )/(np.cos(nu_atan)**2)
        if ioption==0:
            output = np.real(integrand)
        else:
            output = np.imag(integrand)
        return output

    def beta_function(nu,lowernu,discontinuity):
        Integral = PV_analytic_continuation(beta_integrand,nu,lowernu,discontinuity)
        return np.exp(-Integral)*np.absolute(Nhat)
    #
    #   alpha
    #
    def alpha_integrand(nu_atan,nu0,ioption):
        nu = np.tan(nu_atan)
        integrand = imalpha_inter(nu)*(nu-1.)*(nu+nu0)/( (nu**2-1)*(nu**2-nu0**2) )/(np.cos(nu_atan)**2)
        if ioption==0:
            output = np.real(integrand)
        else:
            output = np.imag(integrand)
        return output

    def alpha_function(nu,lowernu):
        Integral = analytic_continuation(alpha_integrand,nu,lowernu)
        return parameter_a + Integral
    
    ''' Solver
    
        Input
        
        nu: array with the nu points
        alpha: array with the alpha trajectory (same size as nu)
        mlambda: mass of the vector meson
        m: mass of the fermion
        nlimit_Nhat: Maximum value of the Nhat sucession
        free_parameters: [a,c] parameters of the model
        lowernu: Minimum value for the integration range
    '''

    #   Numerical parameters
    epsilon = 0.01; iepsilon = 1j*epsilon;
    
    #   Physical parameters
    r = 2.*(mlambda/m)**2
    parameter_a, parameter_c = free_parameters

    rho = rho_function(nu) # Phase space
    rho_inter = interpolation(nu,rho) # interpolation of the phase space

    Nhat = Nhat_function(nu,alpha,r,nlimit_Nhat) # Nhat
    Nhat_inter_re = interpolation(nu,np.real(Nhat)) # interpolation of Nhat
    Nhat_inter_im = interpolation(nu,np.imag(Nhat)) # interpolation of Nhat
#    plt.xscale('log')
#    plt.plot(nu,np.real(Nhat),'-',c=jpac_color[0])
#    plt.plot(nu,np.imag(Nhat),'--',c=jpac_color[1])
#    plt.show()

    disc = discontinuity_phi(nu,Nhat)
    print('discontinuities',disc)
#    beta = parameter_c*beta_function(nu,lowernu,[])
#    plt.ylim((-180,180))
#    plt.xscale('log')
#    plt.plot(nu,np.angle(Nhat)*180/np.pi,'-',c=jpac_color[1])
#    plt.show()
    beta = parameter_c*beta_function(nu,lowernu,disc)
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.plot(nu,beta,'-',c=jpac_color[0])
#    plt.show()

    imalpha = rho*beta
    imalpha_inter = interpolation(nu,imalpha) # interpolation of Im alpha
    alpha = alpha_function(nu,lowernu)

    return alpha, beta, Nhat


###############################################################################
###############################################################################
#
'''             Main Program '''
#
###############################################################################
###############################################################################

#
# timing the code
#
starting_time = time.time() 
#
#   Number of iterations to solve the integral equations
#
iterations = 5
#
#   Output filename
#
filename = 'test'
filenametxt = filename + '.txt'
filenamepdf = filename + '.pdf'
#
#   Kinematics
#
lowernu, deg = 0., 1000;
numin, numax = np.arctan(lowernu), np.pi/2.
nu_atan, w_atan = x_and_weights_gausslegendre([numin,numax],deg)
#
#   Input
#
nu = np.tan(nu_atan)
alphaini = -0.25 # alpha initial value
alpha = alphaini + np.zeros(deg) # alpha initial array
mlambda, m = 1., 1.; # masses in GeV
nlimit_Nhat, free_parameters = 2, [0., 0.5];
#
#   First iteration
#
print('Solving')
print('Iteration: 0')
alpha, beta, Nhat = system_solver(nu, alpha, mlambda, m, nlimit_Nhat, free_parameters, lowernu)
#   Storage
#
alpha_store, beta_store, Nhat_store = np.array([alpha]), np.array([beta]), np.array([Nhat]);
#
#   Iterative resolution
#
for i in range(iterations):
    print('Iteration:', i+1)
    alpha, beta, Nhat = system_solver(nu, alpha, mlambda, m, nlimit_Nhat, free_parameters, lowernu)
    alpha_store, beta_store, Nhat_store  = np.append(alpha_store,[alpha],axis=0), np.append(beta_store,[beta],axis=0), np.append(Nhat_store,[Nhat],axis=0)
#
#   Final solution
#
alphain, betain, Nhatin = alpha, beta, Nhat
#
#   Analytical continuations
#
print('Analytical continuation')
longitud = len(alpha_store[:,0])
parameter_a = free_parameters[0]
epsilon = 0.001; iepsilon = 1j*epsilon;
nuin, nuout = nu, np.linspace(-5,20,100)
nuout_plus, nuout_minus = nuout + iepsilon, nuout - iepsilon

print('iteration: 0')
alphain, betain  = alpha_store[0,:], beta_store[0,:]
alphaout = complex_function('alpha',alphain,nuin,nuout_plus,parameter_a,lowernu)
alpha_store_analytic = np.array([alphaout])

for i in range(1,longitud):
    print('iteration:', i)
    alphain, betain  = alpha_store[i,:], beta_store[i,:]
    alphaout = complex_function('alpha',alphain,nuin,nuout_plus,parameter_a,lowernu)
    alpha_store_analytic = np.append(alpha_store_analytic,[alphaout],axis=0)

alphaout_plus  = alphaout
alphaout_minus = complex_function('alpha',alphain,nuin,nuout_minus,parameter_a,lowernu)
#
#   Plots
#
print('Plotting')
fuente = 15
fig, subfig = plt.subplots(2,3,figsize=(15,10))

for i in range(2):
    for j in range(3):
        subfig[i,j].set_xlim(-5,15)
    
subfig[1,0].set_yscale('log')
subfig[1,1].set_yscale('log')
#subfig[1,2].set_ylim((-180,180))

for i in range(longitud):
    etiqueta = 'it. ' + str(i)
    subfig[0,0].plot(nu,np.real(alpha_store[i,:]),'-',c=jpac_color[i],label=etiqueta)
    subfig[0,0].plot(nuout,np.real(alpha_store_analytic[i,:]),'--',c=jpac_color[i])

    subfig[0,1].plot(nu,np.imag(alpha_store[i,:]),'-',c=jpac_color[i])
    subfig[0,1].plot(nuout,np.imag(alpha_store_analytic[i,:]),'--',c=jpac_color[i])

    subfig[1,0].plot(nu,beta_store[i,:],'-',c=jpac_color[i])

    subfig[1,1].plot(nu,np.real(Nhat_store[i,:]),'-',c=jpac_color[i])
    subfig[1,1].plot(nu,np.imag(Nhat_store[i,:]),'--',c=jpac_color[i])

    subfig[1,2].plot(nu,np.angle(Nhat_store[i,:])*180/np.pi,'-',c=jpac_color[i])


subfig[0,2].plot(nuout,np.real(alphaout_plus),'-',c=jpac_color[0],label=r'Re $\alpha(+)$')
subfig[0,2].plot(nuout,np.imag(alphaout_plus),'-',c=jpac_color[1],label=r'Im $\alpha(+)$')
subfig[0,2].plot(nuout,np.real(alphaout_minus),'--',c=jpac_color[0],label=r'Re $\alpha(-)$')
subfig[0,2].plot(nuout,np.imag(alphaout_minus),'--',c=jpac_color[1],label=r'Im $\alpha(-)$')
subfig[0,2].plot(nuin,np.imag(alphain),'.',c=jpac_color[2])

subfig[0,0].set_xlabel(r'$\nu$',fontsize=fuente)
subfig[0,0].set_title(r'Re $\alpha(\nu)$',fontsize=fuente)
subfig[0,0].tick_params(direction='in',labelsize=fuente)
subfig[0,0].legend(loc='lower left',ncol=2,frameon=False,fontsize=fuente-2)  

subfig[0,1].set_xlabel(r'$\nu$',fontsize=fuente)
subfig[0,1].set_title(r'Im $\alpha(\nu)$',fontsize=fuente)
subfig[0,1].tick_params(direction='in',labelsize=fuente)

subfig[0,2].set_title(r'$\alpha$',fontsize=fuente)
subfig[0,2].set_xlabel(r'$\nu$',fontsize=fuente)
subfig[0,2].legend(loc='lower left',ncol=1,frameon=False,fontsize=fuente-2)   
subfig[0,2].tick_params(direction='in',labelsize=fuente)
     
subfig[1,0].set_xlabel(r'$\nu$',fontsize=fuente)
subfig[1,0].set_title(r'$\beta(\nu)$',fontsize=fuente)
subfig[1,0].tick_params(direction='in',labelsize=fuente)

subfig[1,1].set_xlabel(r'$\nu$',fontsize=fuente)
subfig[1,1].set_title(r'$\hat{N}(\nu)$',fontsize=fuente)
subfig[1,1].tick_params(direction='in',labelsize=fuente)

subfig[1,2].set_xlabel(r'$\nu$',fontsize=fuente)
subfig[1,2].set_title(r'$\phi(\nu)$',fontsize=fuente)
subfig[1,2].tick_params(direction='in',labelsize=fuente)

plt.show()
#
#   Storage
#
print('Storing')
alpha_to_file = np.append(alpha_store_analytic,[nuout],axis=0)
fig.savefig(filenamepdf, bbox_inches='tight')
np.savetxt(filenametxt, alpha_to_file, delimiter=',')  
#
'''
gl = np.loadtxt('gauss-legendre.txt', dtype=np.complex_)

plt.plot(np.real(gl[1,:]),np.real(gl[0,:]),'--',c=jpac_color[0])
plt.plot(nuout,np.real(alphaout_plus),'-',c=jpac_color[10],label=r'Re $\alpha(+)$')
plt.legend(loc='upper right',ncol=2,frameon=False,fontsize=11)  
plt.show()

plt.plot(np.real(gl[1,:]),np.imag(gl[0,:]),'--',c=jpac_color[0])
plt.plot(nuout,np.imag(alphaout_plus),'-',c=jpac_color[10],label=r'Im $\alpha(+)$')
plt.legend(loc='upper right',ncol=2,frameon=False,fontsize=11)
plt.show()
'''

'''
kevin = np.loadtxt('cesar_compare.txt', dtype=np.complex_)
for i in range(1,5):
    etiqueta = 'it. ' + str(i)
    plt.plot(np.real(kevin[:,0]),np.real(kevin[:,i]),'--',c=jpac_color[i],label=etiqueta)
plt.plot(nuout,np.real(alphaout_plus),'-',c=jpac_color[10],label=r'Re $\alpha(+)$')
plt.legend(loc='upper right',ncol=2,frameon=False,fontsize=11)  
plt.show()

for i in range(1,5):
    etiqueta = 'it. ' + str(i)
    plt.plot(np.real(kevin[:,0]),np.imag(kevin[:,i]),'--',c=jpac_color[i],label=etiqueta)
plt.plot(nuout,np.imag(alphaout_plus),'-',c=jpac_color[10],label=r'Im $\alpha(+)$')
plt.legend(loc='upper right',ncol=2,frameon=False,fontsize=11)
plt.show()
'''
#
#   Timing
#
end_time = time.time()
elapsed_time = end_time - starting_time
print(dashes)
print('Computing time:', elapsed_time, 'seconds')
elapsed_time = end_time - starting_time
print('Total time:', elapsed_time, 'seconds')

