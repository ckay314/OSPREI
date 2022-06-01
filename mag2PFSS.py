from scipy.special import lpmn, factorial
from astropy.io import fits
import numpy as np
import math
import sys
import numpy as np
import pickle

global mainpath, codepath, magpath
mainpath = '/Users/ckay/OSPREI/' 
codepath = mainpath + 'codes/'
magpath  = '/Users/ckay/PickleJar/' 

global nHarmonics, rSS
nHarmonics = 90
rSS = 2.5

global nR, nTheta, nPhi
# Parameters defining the pickle shape
nR = 151     # default 151 -> 0.01 Rs resolution for Rss = 2.5 Rs
nTheta = 361 # default 361 -> half deg resolution
nPhi = 720  # default 720 -> half deg resolution


def sync2carr(input_file):
        new_file = input_file[:-5]+'CL'+'.fits'
        myfits = fits.open(magpath+input_file)  
        
        # determine Carrington lon of Earth at time of observation
        Elon = myfits[0].header['CRLN_OBS']

        # calculate shifted lons
        mylons = np.linspace(0, 359.9, 3600)
        newlons = (mylons + Elon - 60.)
        # make sure all new lons are greater than 0
        if newlons[0] < 0: newlons += 360

        # find the point where newlons goes past 360
        # -> where the new split should be
        idx = np.min(np.where(newlons > 360))
        idx2 = 3600 - idx

        # make a new array and fill it with the resplit data
        myfits2 = np.zeros([1440, 3600])
        myfits2[:,:idx2] = myfits[0].data[:,idx:]
        myfits2[:,idx2:] = myfits[0].data[:,:idx]

        # put the new array in the old ones place in the fits and save it
        myfits[0].data[:,:] = myfits2
        myfits.writeto(magpath+new_file, output_verify='ignore', overwrite=True) 
        return new_file
        
def harmonics(obs, IDname, nHarmonics, isSinLat=True):
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
    
    # read in the magnetogram using the date provided
    myfits = fits.open(magpath+obs+IDname+'.fits') 
    orig_data = myfits[0].data

    # Determine the size of the magnetogram
    ny = orig_data.shape[0]
    nx = orig_data.shape[1]
    print('Magnetogram size = ', ny, '(y) x', nx, ' (x)')
    
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
    if isSinLat:
        Thetas = np.array([math.pi * 0.5 - np.arcsin((float(iTheta)+0.5) * dSinTheta - 1.0) for iTheta in range(nTheta)])
    else:
        Thetas = np.linspace(math.pi, 0, nTheta)
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
            logconvPml[badms[i],badls[i]] = 0.5*(math.log(2. * factorial(badls[i]-badms[i])) -math.log(factorial(badls[i]+badms[i])))


    # Array for Legendre polys
    PmlTheta = np.empty([nTheta, nHarmonics+1, nHarmonics+1])
    # Calculate Pml for each theta
    for i in range(nTheta):
        PmlTheta[i,:,:] = lpmn(nHarmonics,nHarmonics,CosThetas[i])[0]*np.power(-1,mls[:,:,0])*np.exp(logconvPml)


    # Open a file to save the output    
    f1 = open(magpath+obs+IDname+'coeffs.dat', 'w') #MTMYS

    # Calculate the harmonic coefficients using the arrays we have and the
    # magnetogram data and save to file
    for l in range(nHarmonics+1):
        print (l, '/'+str(nHarmonics))
        for m in range(l+1):
            BrPlm = data * PmlTheta[:,m,l].reshape([nTheta,-1])
            sumforG = BrPlm*CosMphi[:,m] 
            sumforH = BrPlm*SinMphi[:,m] 
            #print l,m, np.sum(sumforG) * (2*l+1.)/(nTheta*nPhi), np.sum(sumforH) * (2*l+1.)/(nTheta*nPhi)
            f1.write('%4i %4i %15.8f %15.8f' % (l, m, np.sum(sumforG) * (2*l+1.)/(nTheta*nPhi), np.sum(sumforH) * (2*l+1.)/(nTheta*nPhi))+ '\n') 
    f1.close()

    return obs+IDname+'coeffs.dat'
    

# ----------------- PFSS stuff-----------------------------------------------
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

def initcoeffs(coeff_file, nHarmonics, rSS):   
    # function to open up the coefficients file and stuff values
    # into the gml, hml arrays
    # location of the coeff file, make this match your system
    fname = magpath + coeff_file
    inarr = np.genfromtxt(fname, dtype=None)
    # empty coefficient arrays
    gml = np.zeros([nHarmonics+1,nHarmonics+1])
    hml = np.zeros([nHarmonics+1,nHarmonics+1])
    # loop through the text file and put each value where it goes
    for i in range(len(inarr)):
        temp = inarr[i]
        LoScor = temp[0] + 1. + temp[0] * rSS** -(2*temp[0]+1)
        if temp[0] < nHarmonics+1:
            gml[temp[1], temp[0]] = temp[2]/LoScor
            hml[temp[1], temp[0]] = temp[3]/LoScor
    # set the g00 term = 0 -> no monopoles
    gml[0,0] = 0
    return gml, hml 

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

def makedapickle(obs, IDname, nHarmonics, rSS):
    # Main function for calculating the magnetic field for the full volume
    # Set up coefficients and conversion array
    setupMLarrs(nHarmonics)
    gml,hml = initcoeffs(coeff_file, nHarmonics, rSS)
    
    # Things calculated from the global n-values
    rs = np.linspace(1.0,rSS,nR)
    dPhi =2. * math.pi /nPhi
    dTheta = math.pi /nTheta
    #dSinTheta = 2.0/nTheta
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
            outs = calcB(Thetas[i], Phis[j], np.reshape(rs, [-1,1]), nHarmonics, gml, hml,rSS)
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
    
    # Open up files for output and dump the pickles 
    # get name from input file
    pickle_file = 'PFSS_'+obs+IDname
    # Lower half pickle
    fa = open(magpath+pickle_file+'a3.pkl', 'wb')
    pickle.dump(dataa,fa,-1)
    fa.close()
    # Upper half pickle
    fb = open(magpath+pickle_file+'b3.pkl', 'wb')
    pickle.dump(datab,fb,-1)
    fb.close()
    # Br at source surface pickle (useful for calcHCSdist)
    fc = open(magpath+pickle_file+'SS3.pkl', 'wb')
    pickle.dump(Brmap[-1,:,:,0],fc,-1)
    fc.close()

    return pickle_file

def calcHCSdist(pickle_file):
    # Function that first determines the location of the Heliospheric Current
    # Sheet then find the minumum distance from it for all lats/lons.  Use
    # only one deg resolution, should be sufficient since only used for ForeCAT
    # density function
    dtor = math.pi/360.
    
    print ('Finding HCS location...')
    fa = open(magpath+pickle_file+'SS3.pkl', 'rb')
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
    f1 = open(magpath+pickle_file+'dists3.pkl', 'wb') #MTMYS
    pickle.dump(dists,f1,-1)
    f1.close()
    
    
if __name__ == '__main__':
    # call from the command line as python magnetogram2harmonics.py magnetogram.fits
    input_file = sys.argv[1]
    
    obs = None
    # check which observatory we are using (assume it's in filename somewhere)
    if ('HMI' in input_file):
        obs = 'HMI'
        tempname = (input_file.replace('HMI',''))
    elif ('GONG' in input_file):
        obs = 'GONG'
        tempname = (input_file.replace('GONG',''))
    # pull out the unique identifier for this case (no obs or .fits)
    IDname = tempname.replace('.fits', '')

    fits4har = input_file
    if 'sync' in IDname:
        # assume if CL is in IDname we have already ran
        # sync2carr so don't do it again
        if 'CL' not in IDname:
            fits4har = sync2carr(input_file)
            IDname = IDname + 'CL'
    
    # make it ignore div by zero in logs, complains about unused part of array
    np.seterr(divide = 'ignore', invalid = 'ignore') 
    
    # turn magnetogram into harmonic coefficients
    coeff_file = harmonics(obs, IDname, nHarmonics)

    # make the PFSS pickles
    pickle_file = makedapickle(obs, IDname, nHarmonics, rSS)
    
    # get distance from the HCS
    calcHCSdist(pickle_file)
        