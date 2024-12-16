import numpy as np
import sys
import os


mainFile = sys.argv[1]
ensNum   = int(sys.argv[2])

sVs = dict(np.genfromtxt(mainFile, dtype=str)) # seed values

ensFile = sVs['date:']+'/EnsembleParams'+sVs['date:']+sVs['suffix:']+'.dat'
# 20210509/EnsembleParams20210509Christina_Kay_121224_SH_1.dat

ensIns = np.genfromtxt(ensFile, dtype='str')
hdr = ensIns[0]
nParams = len(hdr)-1

#myVals = np.copy(sVs)

for j in range(nParams):
    i = j + 1 
    print (hdr[i], ensIns[ensNum+1][i])
    sVs[hdr[i]+":"] = ensIns[ensNum+1][i]

sVs['nRuns:'] = '1'
outFile = open('run'+sVs['suffix:']+'_EnsMem'+str(ensNum)+'.txt', 'w')

for key in sVs.keys():
    print (key, sVs[key])
    outFile.write(key + ' ' + sVs[key] + '\n')
outFile.close()
    