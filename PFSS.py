import numpy as np
import math
import pickle
import sys
from scipy.special import lpmn
from scipy.misc import factorial
import matplotlib.pyplot as plt

# Script that takes the harmonics coefficient from harmonics.py
# and produces pickles of the magnetic field strength for the full
# solar surface between 1 and the source surface.  We assume half
# degree resolution in latitude and longitude and 151 radial points
# so for Rss = 2.5 this means 0.1 Rsun radial resolution.  The script
# also contains a function to find the location of the HCS and 
# calculated the distance from it, which is used in the density model
# of ForeCAT.

global dtor
dtor = math.pi/360.  # degrees to rads

# Main PFSS programs ----------------------------------------------------|
def initcoeffs(date, ncoeffs, Rss):
    # function to open up the coefficients file and stuff values
    # into the gml, hml arrays
    # location of the coeff file, make this match your system
    fname = '/Users/ckay/PickleJar/HMI'+str(date)+'coeffs.dat'  #MTMYS
    inarr = np.genfromtxt(fname, dtype=None)
    # empty coefficient arrays
    gml = np.zeros([ncoeffs+1,ncoeffs+1])
    hml = np.zeros([ncoeffs+1,ncoeffs+1])
    # loop through the text file and put each value where it goes
    for i in range(len(inarr)):
        temp = inarr[i]
        LoScor = temp[0] + 1. + temp[0] * Rss** -(2*temp[0]+1)
        if temp[0] < ncoeffs+1:
            gml[temp[1], temp[0]] = temp[2]/LoScor
            hml[temp[1], temp[0]] = temp[3]/LoScor
    # set the g00 term = 0 -> no monopoles
    gml[0,0] = 0
    return gml, hml
    
    

def setupMLarrs(nHarmonics):
    # We make use of the built in Legendre polynomials from scipy but
    # need to convert from it to the PFSS normalization.  Calculate a 
    # conversion matrix to be used repeatedly later.  Also set up a matrix
    # with the m and l values in a 2D array matching the Legendre poly form
    global mls, logconvPml
    # ml matrix
    mls = np.empty([nHarmonics+1,nHarmonics+1,2])
    for i in range(nHarmonics+1):
        mls[:,i,0] = range(nHarmonics+1) #ms
        mls[i,:,1] = range(nHarmonics+1) #ls
        
    # Conversion matrix doesn't include (-1)^m since have to use log because
    # of higher orders (l+m > 170)
    convPml = np.empty([nHarmonics+1,nHarmonics+1], dtype=np.longdouble)
    convPml = np.sqrt(2. * factorial(mls[:,:,1] - mls[:,:,0]) / factorial(mls[:,:,1] + mls[:,:,0]))
    convPml[0,:] = 1.
    logconvPml = np.log(convPml)
    
    # Replace the higher orders individually
    brokenFac = np.where(mls[:,:,1]+mls[:,:,0] > 170)
    badms = brokenFac[0]
    badls = brokenFac[1]
    for i in range(len(badms)):
        if badms[i] <= badls[i]:
            logconvPml[badms[i],badls[i]] = 0.5*(math.log(2. * math.factorial(badls[i]-badms[i])) -math.log(math.factorial(badls[i]+badms[i])))
            
            
    
def calcB(theta,phi,rs,nHarmonics,gml,hml, Rss):  
    # Calculate the Br, Btheta, and Bphi components of the PFSS magnetic field
    # for a single theta and phi but vector of r values
    
    # Legendre polys and their derivative in format mxl
    plmres = lpmn(nHarmonics,nHarmonics,np.cos(theta))
    Pml = plmres[0] * np.power(-1,mls[:,:,0]) * np.exp(logconvPml)
    dPml = plmres[1] * np.power(-1,mls[:,:,0]) * np.exp(logconvPml) * np.sin(theta)
   
    # trig functions of m phi in convenient mxl arrays
    CosMPhi = np.cos(phi*mls[:,:,0])
    SinMPhi = np.sin(phi*mls[:,:,0])
   
    # the gh term in either Br or Bp format
    Br_gh = gml*CosMPhi + hml*SinMPhi    
    Bp_gh = gml*SinMPhi - hml*CosMPhi
    
    # ls in format of rs x l
    ls = np.array([range(nHarmonics+1)]*len(rs))
    
    # Calc Br    
    sum1r = np.sum(Br_gh*Pml,axis=0)   
    ltermr = (ls+1.)*np.power(rs,-ls-2) - ls * -np.power(Rss,-ls-2) * np.power(rs/Rss, ls-1) 
    Br = np.sum(sum1r*ltermr, axis=1)  
    
    # Calc Bphi
    sum1p = np.sum(Bp_gh*Pml*mls[:,:,0], axis=0) / np.sin(theta)
    ltermp = np.power(rs,-ls-2) + -np.power(Rss,-ls-2) * np.power(rs/Rss,ls-1)
    Bp = np.sum(sum1p*ltermp, axis=1)
    
    # Calc Btheta
    sum1t = np.sum(Br_gh*dPml, axis=0) 
    Bt = np.sum(sum1t*ltermp, axis=1)
    
    # return array with a vector for each component containing the values
    # for the given rs
    return [Br, Bt, Bp]
    
    
    
def SPH2CARTvec(colat, lon, vr, vt, vp):
    # Convert from Cartesian to Spherical coordinates
    # Works for vectors, assuming input in rads
    vx = np.sin(colat) * np.cos(lon) * vr + np.cos(colat) * np.cos(lon) * vt - np.sin(lon)* vp
    vy = np.sin(colat) * np.sin(lon) * vr + np.cos(colat) * np.sin(lon) * vt + np.cos(lon)* vp
    vz = np.cos(colat) * vr - np.sin(colat) * vt
    return [vx,vy,vz]
    
    

def makedapickle(date,nHarmonics, Rss):
    # Main function for calculating the magnetic field for the full volume
    # Set up coefficients and conversion array
    setupMLarrs(nHarmonics)
    gml,hml = initcoeffs(date, nHarmonics, Rss)

    # Parameters defining the pickle shape
    nR = 151     # default 151 -> 0.1 Rs resolution for Rss = 2.5 Rs
    nTheta = 361 # default 361 -> half deg resolution
    nPhi = 720  # default 720 -> half deg resolution
    
    # Things calculated from the above n-values
    rs = np.linspace(1.0,Rss,nR)
    dPhi =2. * math.pi /nPhi
    dTheta = math.pi /nTheta
    dSinTheta = 2.0/nTheta
    Thetas = np.linspace(math.pi,0,nTheta)
    # shift end points slightly to avoid div by sintheta=0
    Thetas[0] -= 0.0001
    Thetas[-1] += 0.0001
    # Non-inclusive endpoint so don't have 0 and 2pi -> half deg spacing
    Phis = np.linspace(0,2*math.pi,nPhi, endpoint=False)
    Thetas2D = np.zeros([nTheta, nPhi])
    Phis2D = np.zeros([nTheta, nPhi])
    for i in range(nTheta):
        Phis2D[i,:] = Phis
    for i in range(nPhi):
        Thetas2D[:,i] = Thetas
    
    # Set up output arrays
    dataa = np.empty([int(nR/2+1), nTheta, nPhi, 4])
    datab = np.empty([int(nR/2+1), nTheta, nPhi, 4])
    Brmap = np.zeros([nR,nTheta,nPhi,3])

    # Loop over theta and phi and simulataneously calculate the B
    # vector for all R values
    for i in range(nTheta):
        print( i, ' of ', nTheta)
        for j in range(nPhi):
            outs = calcB(Thetas[i], Phis[j], np.reshape(rs, [-1,1]), nHarmonics, gml, hml,Rss)
            Brmap[:,i,j,0] = outs[0]
            Brmap[:,i,j,1] = outs[1]
            Brmap[:,i,j,2] = outs[2]
                
    # Convert from Spherical coords to Cartesian and stuff in the
    # output arrays          
    for iR in range(nR):
        Bxyz = np.array(SPH2CARTvec(Thetas2D,Phis2D,Brmap[iR,:,:,0],Brmap[iR,:,:,1],Brmap[iR,:,:,2]))
        # Lower half
        if iR <= nR/2:
            dataa[iR,:,:,0] = Bxyz[0]
            dataa[iR,:,:,1] = Bxyz[1]
            dataa[iR,:,:,2] = Bxyz[2]
            dataa[iR,:,:,3] = np.sqrt(Bxyz[0]**2 + Bxyz[1]**2 + Bxyz[2]**2)
        # Higher half
        if iR >= nR/2-1:
            newiR = int(iR - nR/2)+1
            datab[newiR,:,:,0] = Bxyz[0]
            datab[newiR,:,:,1] = Bxyz[1]
            datab[newiR,:,:,2] = Bxyz[2]
            datab[newiR,:,:,3] = np.sqrt(Bxyz[0]**2 + Bxyz[1]**2 + Bxyz[2]**2)
    #fig = plt.figure()
    #plt.imshow(dataa[0,:,:,0], origin='lower')
    #plt.show()
    
    
    # Open up files for output and dump the pickles #MTMYS
    pickle_path = '/Users/ckay/PickleJar/'
    # Lower half pickle
    fa = open(pickle_path+'PFSS'+str(date)+'a3.pkl', 'wb')
    pickle.dump(dataa,fa,-1)
    fa.close()
    # Upper half pickle
    fb = open(pickle_path+'PFSS'+str(date)+'b3.pkl', 'wb')
    pickle.dump(datab,fb,-1)
    fb.close()
    # Br at source surface pickle (useful for calcHCSdist)
    fc = open(pickle_path+'PFSS'+str(date)+'SS3.pkl', 'wb')
    pickle.dump(Brmap[-1,:,:,0],fc,-1)
    fc.close()
    
    
    
    
# HCS distance function----------------------------------------------------|
def calcHCSdist(date):
    # Function that first determines the location of the Heliospheric Current
    # Sheet then find the minumum distance from it for all lats/lons.  Use
    # only one deg resolution, should be sufficient since only used for ForeCAT
    # density function
    
    print ('Finding HCS location...')
    pickle_path = '/Users/ckay/PickleJar/' #MTMYS
    fa = open(pickle_path+'PFSS'+str(date)+'SS3.pkl', 'rb')
    Bss = pickle.load(fa)
    fa.close()
    
    # Set up new array with one deg resolution
    bounds = np.zeros([181,360])
    
    # Part 1 - Find the HCS
    # Loop through each longitude and find the latitude(s) where Br swiches signs
    for i in range(bounds.shape[1]):
        Btemp = np.sign(Bss[:,2*i])
        signshift = Btemp[1:]-Btemp[:-1]
        shiftidx = np.where(signshift != 0)[0]
        for j in shiftidx:
            j2 = int(j/2)
            bounds[j2,i] = -10
        
    # Loop through each latitude and find the longitude(s) where Br switches signs   
    for j in range(bounds.shape[0]):
        Btemp = np.sign(Bss[2*j,:])
        signshift = Btemp[1:]-Btemp[:-1]
        shiftidx = np.where(signshift != 0)[0]
        for i in shiftidx:
            PorM = np.sign(signshift[i])
            i2 = int(i/2)
            bounds[j,i2] = -10
            
    # Part 2 - Calculate distances from the HCS
    print( 'Calculating distances from HCS')
    dists = np.zeros([181,360])
    HCSlats = []
    HCSlons = []
    # Pull out the HCS points from bounds
    idxs = np.where(bounds==-10)
    for i in range(len(idxs[0])):
        HCSlats.append(dtor * (idxs[0][i]-90))
        HCSlons.append(dtor*idxs[1][i])
    HCSlats = np.array(HCSlats)
    HCSlons = np.array(HCSlons)
    
    # Calculate the distance from all HCS points and take the min
    for i in range(dists.shape[0]):
        #print i, 'of ', dists.shape[0]
        mylat = dtor * (i - 90.)
        for j in range(dists.shape[1]): 
            mylon = dtor * j 
            HCSdists =  np.abs(np.arccos(np.sin(HCSlats) * np.sin(mylat) + np.cos(HCSlats) * np.cos(mylat) * np.cos(HCSlons - mylon)))
            HCSdists = np.nan_to_num(HCSdists)
            minHCSdist = np.min(HCSdists[np.where(HCSdists >0)]) / dtor
            dists[i,j] = minHCSdist  
            
    
    # Save the HCS distance to a file
    f1 = open(pickle_path+'PFSS'+str(date)+'dists3.pkl', 'wb') #MTMYS
    pickle.dump(dists,f1,-1)
    f1.close()
    
    # Useful option for quick plots to check things
    #fig = plt.figure()
    #plt.imshow(dists, origin='lower')
    #plt.show()

# Read in the date ID from command line
date = str(sys.argv[1])    
# Make the PFSS pickles
makedapickle(date,90, 2.5)
# Calculate the distance from the HCS
calcHCSdist(date)