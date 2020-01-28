import pyfits
import numpy as np

# This script simply shifts the fits file from the synchronic
# frame into Carrington cooridnates at the time of observation
# The fits header provides the Carrington lon and we know the 
# first 120 deg of data correspond to +/-60 deg about this 
# Carrington lon

# pull in the magnetogram, set for my naming convention but can adjust
# as desired and make this match your system
date = 20120712
myfits = pyfits.open('/Users/ckay/PickleJar/HMI' + str(date) + 'sync.fits')  #MTMYS

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
myfits.writeto('/Users/ckay/PickleJar/HMI'+str(date)+'.fits', output_verify='ignore') #MTMYS
