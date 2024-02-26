import os, sys
import numpy as np

# Import path names from file
myPaths = np.genfromtxt('myPaths.txt', dtype=str)
mainpath = myPaths[0,1]
codepath = myPaths[1,1]
magpath  = myPaths[2,1]

sys.path.append(os.path.abspath(mainpath)) 
sys.path.append(os.path.abspath(codepath)) 
sys.path.append(os.path.abspath(mainpath='processCode/')) 

from OSPREI import runOSPREI
from processOSPREI import runproOSP


# Takes the output from the CCMC query form and sets up everything
# needed to run OSPREI
input_file = sys.argv[1]

# ------------------------------------------------
# ---------- Read in the query files -------------
# ------------------------------------------------
# Set to skip top line "Control File"
data = np.genfromtxt(input_file, dtype=str, skip_header=1)
# Turn into a dictionary and rm : from names
inputs = {}
for i in range(len(data[:,0])):
    inputs[data[i,0][:-1]] = data[i,1]

preProcessIt(inputs)

runName = 'runScript_'+inputs['suffix']+'.txt'
runOSPREI(inputPassed=runName)
runproOSP(inputPassed=runName)


# package the results
os.chdir(mainpath+inputs['date']+'/')
os.system('tar -czf '+inputs['suffix']+'.tgz *'+inputs['suffix']+'*')
os.system('mv '+inputs['suffix']+'.tgz'+' ' +mainpath)
os.chdir(mainpath)