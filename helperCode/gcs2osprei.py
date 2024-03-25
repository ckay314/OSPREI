import numpy as np
from scipy.optimize import fsolve

# Take alpha and kappa from a GCS fit and convert to AW and AWp

dtor = np.pi / 180
radeg = 1. / dtor


def calcAWs(alpha, kappa):
    rFront = 21.5
    alpha2 = alpha*dtor
    h = (rFront*(1-kappa)) * (np.cos(alpha2)/(1+np.sin(alpha2)))
    b = h / np.cos(alpha2)
    rho = h * np.tan(alpha2)
    def f(x):
        A = np.sqrt(((rho+(b*kappa**2*np.sin(x)))/(1-kappa**2))**2 + (((b**2*kappa**2)-rho**2)/(1-kappa**2)))
        part1 = ((b*kappa**2)/(1-kappa**2)) * (np.cos(x)**2 - np.sin(x)**2)
        part2 = ((b*kappa**2)/(A*((1-kappa**2)**2))) * ((b*kappa**2*np.sin(x)*np.cos(x)**2) + (rho*np.cos(x)**2))
        part3 = (A+(rho/(1-kappa**2))) * np.sin(x)
        return part1 + part2 - part3
    beta = fsolve(f,0)
    X0 = (rho+b*kappa**2*np.sin(beta[0]))/(1-kappa**2)
    R = np.sqrt(X0**2 + ((b**2*kappa**2-rho**2)/(1-kappa**2)))
    BP = X0+R
    OBP = np.pi/2 + beta[0]
    OP = np.sqrt(b**2 + BP**2 - 2*b*BP*np.cos(OBP))
    AW = np.arcsin((np.sin(OBP)/OP)*BP) * radeg
    AWp = np.arctan(kappa)*radeg
    return AW, AWp
    
print (calcAWs(88.20, .9137))