import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import pickle
import matplotlib.cm as cm


# read in inputs from command line
fname = str(sys.argv[1])

# init figure
fig = plt.figure()
ax = plt.axes([0.1, 0.3, 0.8, 0.4]) 

# load B105 pickle
f1 = open(fname, 'rb')
B = pickle.load(f1)
f1.close()
Bslice = B[5,:,:,:]

# calc Br
nTheta = 361
nPhi = 720

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

xyz = np.zeros([nTheta, nPhi, 3])
xyz[:,:,0] = np.sin(Thetas2D)*np.cos(Phis2D)
xyz[:,:,1] = np.sin(Thetas2D)*np.sin(Phis2D)
xyz[:,:,2] = np.cos(Thetas2D)
        
Br = np.zeros([361, 720])
for i in range(361):
	for j in range(720):
		Br[i,j] = (Bslice[i,j,0] * xyz[i,j,0] +  Bslice[i,j,1] * xyz[i,j,1] 
				+  Bslice[i,j,2] * xyz[i,j,2]) / 1.05

maxB = 0.1*np.max(np.abs(Br))
plt.imshow(Br, origin='lower', cmap=cm.RdBu, vmin=-maxB, vmax=maxB)
plt.show()