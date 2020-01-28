import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import matplotlib.cm as cm


# read in inputs from command line
date = str(sys.argv[1]) 
pickle_path = '/Users/ckay/PickleJar/' #MTMYS
fname = pickle_path+'PFSS'+ date + 'a.pkl'


# init figure
fig = plt.figure()
ax = plt.axes([0.1, 0.3, 0.8, 0.4]) 

# load B105 pickle
f1 = open(fname, 'rb')
B = pickle.load(f1)
f1.close()
Bslice = B[10,:,:,:]

# calc Br
f3 = open('/Users/ckay/PickleJar/xyz.pkl', 'rb')
xyz = pickle.load(f3)
f3.close()
Br = np.zeros([361, 720])
for i in range(361):
	for j in range(720):
		Br[i,j] = (Bslice[i,j,0] * xyz[i,j,0] +  Bslice[i,j,1] * xyz[i,j,1] 
				+  Bslice[i,j,2] * xyz[i,j,2]) / 1.05

maxB = 0.5*np.max(np.abs(Br))
plt.imshow(Br, origin='lower', cmap=cm.RdBu, vmin=-maxB, vmax=maxB)
plt.show()