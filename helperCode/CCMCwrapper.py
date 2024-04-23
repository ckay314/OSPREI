import os, sys
import numpy as np

# Import path names from file
myPaths = np.genfromtxt('myPaths.txt', dtype=str)
mainpath = myPaths[0,1]
codepath = myPaths[1,1]
magpath  = myPaths[2,1]

sys.path.append(os.path.abspath(mainpath)) 
sys.path.append(os.path.abspath(codepath)) 
sys.path.append(os.path.join(codepath[:-8],'processCode/')) 
sys.path.append(os.path.join(codepath[:-8],'helperCode/')) 

from OSPREI import runOSPREI
from processOSPREI import runproOSP
from query2osprei import preProcessIt

# Takes the output from the CCMC query form and sets up everything
# needed to run OSPREI
input_file = sys.argv[1]

# ------------------------------------------------
# ---------- Read in the query files -------------
# ------------------------------------------------
# Set to skip top line "Control File"
data = np.genfromtxt(input_file, dtype=str)
# Turn into a dictionary and rm : from names
inputs = {}
for i in range(len(data[:,0])):
    inputs[data[i,0][:-1]] = data[i,1]
    
# check if we have magfile and put it in the mag Folder if it isn't 
# there already
if 'MagFile' in inputs.keys():
    if not os.path.isfile(magpath+inputs['MagFile']):
        os.system('mv '+inputs['MagFile'] +' '+magpath+inputs['MagFile'])
        
    # Specific processing for Gong files from ISWA data
    if inputs['MagFile'][:5] == 'mqrzs':
        imFile = inputs['MagFile']
        myfits = fits.open(magpath+imFile)  
 
        # get lon of L edge
        lLon = int(imFile[-8:-5])

        # make list of lon of each column
        # map has 1 deg resolution, unhardcode for more flexibility?
        lons = np.linspace(0,359,360) + lLon
        lons = lons % 360

        # find split point
        splitIdx = np.where(lons == 0)[0][0]
        idx2 = 360-splitIdx

        # copy and move
        newfits = np.zeros([180, 360])
        newfits[:, :idx2] = myfits[0].data[:, splitIdx:]
        newfits[:, idx2:] = myfits[0].data[:, :splitIdx]

        newlons = np.zeros(360)
        newlons[:idx2] = lons[splitIdx:]
        newlons[idx2:] = lons[:splitIdx]

        # resave
        myfits[0].data[:,:] = newfits
        newName = imFile[:-9]+'_000.fits'
        inputs['MagFile'] = newName
        myfits.writeto(magpath+newName, output_verify='ignore', overwrite=True)

# run the preprocessing 
preProcessIt(inputs)

runName = 'runScript_'+inputs['suffix']+'.txt'
runOSPREI(inputPassed=runName)
runproOSP(inputPassed=runName)


# package the results
os.system('cp '+runName +' '+mainpath+inputs['date']+'/')
os.chdir(mainpath+inputs['date']+'/')
os.system('tar -czf '+inputs['suffix']+'.tgz *'+inputs['suffix']+'*')
os.system('mv '+inputs['suffix']+'.tgz'+' ' +mainpath)
os.chdir(mainpath)