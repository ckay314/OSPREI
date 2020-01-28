from scipy.special import lpmn
from scipy.misc import factorial
import pyfits as pf
import numpy as np
import math
import sys

# Script that takes in a magnetogram and produces the harmonic
# coefficients needed for the PFSS model.
# The algorithm was originally largely based on the IDL script
# harmonics.pro that produces coefficients to run the Michigan
# SWMF code.  It has been vectorized and changed but the same
# normalization is used here and in our PFSS code.  
# This normalization for the Legendre polynomials differs from
# that of the built in lpmn function so we adjust for that.
# This doesn't include the source surface height, that part of the 
# normalization is done in the PFSS script.
# More details on normalization can be found in Xudong Sun's
# notes on the PFSS model at wso.stanford.edu/words/pfss.pdf,
# which was used extensively as a reference for this script

# The syntax for calling it is
# python harmonics.py ####
# where #### is a date used to identify the magnetogram


# Number of harmonics
nHarmonics = 90  # default is 90


# read in the magnetogram using the date provided
date = str(sys.argv[1])
# location of the magnetogram, make this match your system
myfits = pf.open('/Users/ckay/PickleJar/HMI'+date+'.fits')  #MTMYS
orig_data = myfits[0].data


# The data can be uniformly scaled up if desired.  There's no 
# reason to do this for normal solar cases, but can be useful
# for extrasolar work
#orig_data = orig_data * 3


# Determine the size of the magnetogram
ny = orig_data.shape[0]
nx = orig_data.shape[1]
print 'Magnetogram size = ', ny, '(y) x', nx, ' (x)'
print ''


# Check for various bad data flags, either NaNs/infs.  The flags change 
# from observatory to observatory but this takes care of most issues
# clean out NaN/infs
good_points = np.isfinite(orig_data)
# assign bad points to zero
data = np.zeros([ny,nx])
data[good_points] = orig_data[good_points]


# Set up constants used in the process
nPhi = nx
nTheta = ny
dPhi =2. * math.pi /nPhi
dTheta = math.pi /nTheta
dSinTheta = 2.0/nTheta

# Arrays that will hold harmonic coefficients
g_ml = np.zeros([nHarmonics+1, nHarmonics+1])
h_ml = np.zeros([nHarmonics+1, nHarmonics+1])

# More arrays to hold useful values 
# Thetas = colatitude
Thetas = np.array([math.pi * 0.5 - np.arcsin((float(iTheta)+0.5) * dSinTheta - 1.0) for iTheta in range(nTheta)])
CosThetas = np.cos(Thetas)
# Phis = longitude
Phis = np.array([dPhi*i for i in range(nPhi)])
# vertical arrays of Ms to calculate mPhi
phiMs = np.array([[i]*nPhi for i in range(nHarmonics+1)])
phiMs = np.transpose(phiMs)
SinMphi = np.sin(phiMs * Phis.reshape([nPhi,-1]))
CosMphi = np.cos(phiMs * Phis.reshape([nPhi,-1]))
# Array holding the values of m and l in a 2d grid to make matrix mult easier
mls = np.empty([nHarmonics+1,nHarmonics+1,2])
for i in range(nHarmonics+1):
    mls[:,i,0] = range(nHarmonics+1) #ms
    mls[i,:,1] = range(nHarmonics+1) #ls
    
    
# Need to be able to convert from scipy poly normalization to 
# PFSS version, which depends on some large factorials
# Save the log of convPml without the (-1)^m factor otherwise will round to
# zero for m+l above 170, will take exponent later    
convPml = np.empty([nHarmonics+1,nHarmonics+1], dtype=np.longdouble)
convPml = np.sqrt(2. * factorial(mls[:,:,1]-mls[:,:,0])/factorial(mls[:,:,1]+mls[:,:,0]))
convPml[0,:] = 1.
logconvPml = np.log(convPml)
# Replace the values for l+m above 170 where the other factorial breaks
brokenFac = np.where(mls[:,:,1]+mls[:,:,0] > 170)
badms = brokenFac[0]
badls = brokenFac[1]
for i in range(len(badms)):
    if badms[i] <= badls[i]:
        logconvPml[badms[i],badls[i]] = 0.5*(math.log(2. * math.factorial(badls[i]-badms[i])) -math.log(math.factorial(badls[i]+badms[i])))


# Array for Legendre polys
PmlTheta = np.empty([nTheta, nHarmonics+1, nHarmonics+1])
# Calculate Pml for each theta
for i in range(nTheta):
    PmlTheta[i,:,:] = lpmn(nHarmonics,nHarmonics,CosThetas[i])[0]*np.power(-1,mls[:,:,0])*np.exp(logconvPml)


# Open a file to save the output    
f1 = open('/Users/ckay/PickleJar/HMI'+str(date)+'coeffs.dat', 'w') #MTMYS


# Calculate the harmonic coefficients using the arrays we have and the
# magnetogram data and save to file
for l in range(nHarmonics+1):
    print l, '/'+str(nHarmonics)
    for m in range(l+1):
        BrPlm = data * PmlTheta[:,m,l].reshape([nTheta,-1])
        sumforG = BrPlm*CosMphi[:,m] 
        sumforH = BrPlm*SinMphi[:,m] 
        #print l,m, np.sum(sumforG) * (2*l+1.)/(nTheta*nPhi), np.sum(sumforH) * (2*l+1.)/(nTheta*nPhi)
        f1.write('%4i %4i %15.8f %15.8f' % (l, m, np.sum(sumforG) * (2*l+1.)/(nTheta*nPhi), np.sum(sumforH) * (2*l+1.)/(nTheta*nPhi))+ '\n') 
f1.close()          