import numpy as np
import math
import ForeCAT_functions as FC
import ForceFields as FF
from scipy.special import ellipk, ellipe


global dtor, radeg
dtor  = 0.0174532925   # degrees to radians
radeg = 57.29577951    # radians to degrees

# Empirical function to calculate length of axis
global lenFun
lenCoeffs = [0.61618, 0.47539022, 1.95157615]
lenFun = np.poly1d(lenCoeffs)


def cart2cart(x_in, lat, lon, tilt):
    # take in an xyz coordinate in frame where FR parallel to z axis and nose along x axis
    xyz 	  = [x_in[0], x_in[1], x_in[2]]
    newxyz 	  = FC.rotz(FC.roty(FC.rotx(xyz, -(90-tilt)), -lat), lon)
    return newxyz


class CME:
    "Represent a CME using a grid representing the flux rope shape"

    def __init__(self, pos, params, Ntor_in, Npol_in, user_vr, user_exp, user_mass, AWratio, rsun_in):
        # Initialize parameters using numbers fed in from other parts of ForeCAT
        # pos should contain [lat, lon, tilt]
        # params should contain [deltaAx, deltaCS, rstart]

        # Grid Parameters
        global Ntor, Npol, Npoints, idcent
        Ntor = Ntor_in
        Npol = Npol_in
        Npoints = Ntor * Npol # total number of grid points
        idcent = int(Npoints / 2) # ID of the nose point
        # Determine the angular spacing of the grid points
        delta_maj = 120. / (Ntor - 1.) * dtor  # point spacing along toroidal
        delta_min = 120. / (Npol - 1.) * dtor  # point spacing along poloidal
        # toroidal and poloidal angle for each grid point
        self.t_angs = [delta_maj * (int(Ntor / 2) - int(i/Npol)) for i in range(Npoints)]
        self.p_angs = [delta_min * (i - int(Npol / 2)) for i in range(Npol)] * Ntor
		
        # assign the stellar parameters
        global rsun
        rsun = rsun_in
        
        rstart = params[2]
        self.AWratio = AWratio
 
        # Initial CME mass
        self.M  = user_mass(rstart)

        # Set up arrays to hold the info at each grid point and calculate 
        # their positions in Cart and SPH
        # Cartesian and SPH position for each point
        self.points = [[]]*Npoints
        for i in range(Npoints):			# points.[i][0,:] = xyz
	        self.points[i] = np.zeros([2, 3])	# points.[i][1,:] = SPH (R, lat, lon)
        # position of the center of the CME cone (center of toroidal axis) 
        self.cone = np.zeros([2,3])
        # Set up the CME shape parameters
        self.deltaAx = params[0]
        self.deltaCS = params[1]
        self.AW = user_exp(rstart)* dtor # initial angular width
        self.AWp = self.AWratio(rstart) * self.AW 
        # initial nose position
        self.points[idcent][1,:] = [rstart, pos[0], pos[1]]
        self.calc_lens()
        self.deltaCSAx = self.rr / self.Lp
        
        # convenient for serial version to have r/lat/lon/x/y/z in arrays
        self.rs = np.zeros(Npoints)
        self.lats = np.zeros(Npoints)
        self.lons = np.zeros(Npoints)
        self.xs = np.zeros(Npoints)
        self.ys = np.zeros(Npoints)
        self.zs = np.zeros(Npoints)

        # unit vector in the radial direction -- needed by many programs
        self.rhat = np.zeros(3)   # updated in calc points
        # Tilt of the flux rope (measured wrt N)
        self.tilt = pos[2]  
        # calculate initial position of grid points
        self.calc_points()
        
        # Calculate eccentricity for moment of inertia
        self.ecc =0.
        if self.Lp >= self.Lr : # c >= a
            self.ecc = np.sqrt((self.Lp**2 - self.Lr**2) / self.Lp**2 )
        else:
            self.ecc = np.sqrt((self.Lr**2 - self.Lp**2) / self.Lr**2 )

        # Forces at each point
        self.defforces = [[]]*Npoints
        for i in range(Npoints):
            # 0 = tension, 1 = grad B
            self.defforces[i] = np.zeros([2,3]) 
        self.Fpgrad = np.zeros([Npoints,3])
        self.Ftens = np.zeros([Npoints,3])
        # Single drag force for full CME
        self.Fdrag = np.zeros(3)

        # Set up vectors for velocity and momentum
        # have a 3D cartesian velocity for center (both radial velocity, def vel, and drag vel)
        self.vels = np.zeros([3,3])
        # want to express deflection/drag velocity in terms of a lat and lon component
        self.vdefLL = np.array([0.e5, 0.e5]) #can change this to initiate CME with nonzero vdef (in cm/s)
        self.vdragLL = np.array([0., 0.])
        # have a 3D cartesian acceleration for center (keep deflection and drag separate)
        self.acc = np.zeros([2,3])
        # angular momentum (for rotation)
        self.angmom = 0. # starts not rotating
        # angular velocity (which changes the tilt of the CME)
        self.angvel = 0.  # [deg/s]
        
        # initialize density
        self.calc_rho()
        # init radial velocity 
        vrmag, self.vels[0,:]= user_vr(self.points[idcent][1,0], self.rhat)

        # include time parameters for convenience even though not really CME properties
        self.dt = 0.1 # in minutes
        self.t  = 0.	# start at zero, in minutes
		
        # Will compare current force values to previous average, initialize average here
        global prev_avgF
        prev_avgF = 99999.  # set to absurdly large number as place holder

        # add extra variables for OSPREI, set to reasonable defaults
        self.Cd = 1.0
        self.v1AU  = 400. * 1e5
        self.vSW = 400
        self.nSW = 5.
        self.BSW = 6.9
        self.BSWvec = [np.sqrt(self.BSW), -np.sqrt(self.BSW), 0]
        self.TSW = 68000
        self.FRBtor = 0
        self.FRT = 0
        self.tau  = 1.
        self.cnm  = 1.927
        self.vs   = np.array([vrmag, 0, 0, 0, 0, 0, 0]) 
        self.getvs(vrmag)
        self.vTrans = 0.
        self.IVDfs = [0.5, 0.5]
        self.impV = 0.
        self.impVE = 0.
        self.gamma = 1.3333
        # add yaw option for ANT/FIDO but not used in FCAT
        self.yaw = 0.
        
        
        # CME sheath properties
        self.hasSheath = False
        self.vShock = 0
        self.comp = 1
        self.shDur = 0.
        self.shDens = self.nSW
        self.shB = self.BSW
        self.shTheta = 0.
        self.shvt = 0.
        self.shv = self.v1AU / 1e5 
        self.shT = self.TSW
        # MEOW-HiSS properties
        self.MHdist = 0.5
        self.MHarea = 800
        
        
        # redefine dt in secs and as short name variable for convenience
        global dt
        dt = self.dt * 60.
        

	# Programs typically called only within this class
    def calc_lens(self):
        self.rp = np.tan(self.AWp) / (1 + self.deltaCS * np.tan(self.AWp)) * self.points[idcent][1,0]
        self.rr = self.deltaCS * self.rp
        self.Lp = (np.tan(self.AW) * (self.points[idcent][1,0] - self.rr) - self.rr) / (1 + self.deltaAx * np.tan(self.AW)) 
        self.Lr = self.deltaAx * self.Lp
        self.cone[1,:] = self.points[idcent][1,:]
        self.cone[1,0] += -self.Lr - self.rr # new version
        self.cone[0,:] = FC.SPH2CART(self.cone[1,:])

    def calc_rho(self):        
        # Initiate CME density
        vol = math.pi*self.rr*self.rp * lenFun(self.Lr/self.Lp)*self.Lp
        self.rho = self.M / vol / rsun**3
	
    def calc_points(self):
        # example for 15 point grid 5x3
        # points numbering  (top)  0  3  6  9   12   (bot)
        #  toroidal --->	   1  4  7  10  13    --> 7 is nose
        #	axis		   2  5  8  11  14
        FF.calc_posCPU(self)
        # Update the unit vector giving radial direction
        colat = (90. - self.cone[1,1]) * dtor
        lon   = self.cone[1,2] * dtor
        nx = np.sin(colat) * np.cos(lon)
        ny = np.sin(colat) * np.sin(lon)
        nz = np.cos(colat)
        self.rhat = np.array([nx, ny, nz])


    def get_center_acc(self):
    # assume in cgs
    # convert the forces at each point into an acceleration at the center
	
        # Sum up all the forces
        # initialize variables to hold force sums
        fx = 0.; fy = 0.; fz = 0.
        for i in range(Npoints):  # sum over all points 
            fx += np.sum(self.defforces[i][:,0]) # summing over both forces
            fy += np.sum(self.defforces[i][:,1]) 
            fz += np.sum(self.defforces[i][:,2]) # CAN GPU SUM? !!!!!CK
        ftot = np.array([fx, fy, fz])
        
        # determine average magnitude of the forces
        # in GPU_functions we check that individual forces are with 2 orders of mag of 
        # the previous average as they can blow up for various reasons
        global prev_avgF  
        prev_avgF = (np.abs(fx) + np.abs(fy) + np.abs(fz)) / Npoints / 3.
		
        # remove radial component of deflection forces
        Fdotr = np.sum(ftot * self.rhat)
        ftot -= Fdotr * self.rhat  # remove radial component
        self.acc[0,:] = ftot / self.rho / Npoints # avg def force
        self.acc[1,:] = self.Fdrag / self.rho # drag force
        # include rotation if desired
        if FC.rotCME: self.calc_torque()
		
    def calc_torque(self):
        # Use the deflection forces to calculate the average torque about the CME nose
        # which we can use to determine the rotation which changes the tilt

        # copy terms need for the moment of inertia to short variable names 
        ee = self.ecc # eccentricity
        bb = self.rp * rsun # now in cm
        # set cc to largest of [a,c] and aa to the smallest
        cc = np.max([self.Lp, self.Lr]) * rsun 
        aa = np.min([self.Lp, self.Lr]) * rsun

        ome2 = 1. - ee**2 # one minus e squared

        # Calculate the moment of inertia using the analytic expression 
        # version for a=c
        if ee == 0: Irot = self.rho * math.pi**2 * cc * bb**2 * (cc**2 + 1.25 * bb**2)
        # more complicated general version
        else:
            eps = ee**2 / (ee**2 - 1)
            Irot = 0.5 * math.pi * self.rho * bb**2 * cc * ( 4*cc**2*(ellipk(eps) + (ee**2-1)*ellipe(eps))/ee**2/np.sqrt(ome2) + bb**2 * (ellipk(ee**2) + 3*((ee**2 - 1)*ellipk(ee**2) + ellipe(ee**2))/ee**2))
 
            
        # switch between GPU and CPU
        tottor = FF.calc_torqueCPU(self)
        # Determine average torque and adjust angular momentum + rotational velocity accordingly
        avgtor = tottor  / Npoints 
        cmevol = self.M / self.rho # easy way to get volume
        self.angmom += avgtor * cmevol * self.dt * 60. # change in ang mom from torque
        # calc rot speed from new ang mom and new Irot
        # if torque -> 0 ang mom should be conserved
        # took out - when switching tilt from old version
        self.angvel = self.angmom / Irot * radeg
	

    def addDefDrag(self, vrmag):
        # Determine velocities in latitude and longitude directions - have to split it up this
        # way so that we can conserve angular momentum in each direction once forces -> 0
        # determine colat and lon unit vectors
        colat = (90. - self.cone[1,1]) * dtor
        lon = self.cone[1,2] * dtor
        colathat = np.array([np.cos(lon) * np.cos(colat), np.sin(lon) * np.cos(colat), -np.sin(colat)]) 
        lonhat = np.array([-np.sin(lon), np.cos(lon), 0.])
        # calculate the change in velocity from def and drag in lat and lon dirs
        deltavdefLL = dt * np.array([np.dot(self.acc[0,:], colathat), np.dot(self.acc[0,:], lonhat)])
        deltavdragLL = dt * np.array([np.dot(self.acc[1,:], colathat), np.dot(self.acc[1,:], lonhat)])
        # determine the change in the lat and lon components of the def and drag
        # add in new and scale old by 1 / R to conserve momentum
        newvdefLL = self.vdefLL * self.points[idcent][1,0] / (self.points[idcent][1,0] + vrmag * dt / rsun) + deltavdefLL
        newvdragLL = self.vdragLL * self.points[idcent][1,0] / (self.points[idcent][1,0] +  vrmag * dt / rsun) + deltavdragLL
        # if the drag velocity exceeds the def set drag = opposite of def so that the total is 0
        if np.abs(newvdragLL[0]) > np.abs(newvdefLL[0]): newvdragLL[0] = -newvdefLL[0]
        if np.abs(newvdragLL[1]) > np.abs(newvdefLL[1]): newvdragLL[1] = -newvdefLL[1]
        # combine the vectors
        vddLL =  0.5 * (newvdefLL + self.vdefLL + newvdragLL + self.vdragLL)
        # store the new values
        self.vdefLL = newvdefLL
        self.vdragLL = newvdragLL
	
        # Determine the new position
        # calc new nose sph pos, update lat/lon using vdd[0]/vdd[1]
        self.points[idcent][1,1] += - radeg * vddLL[0] * dt /  self.points[idcent][1,0] / rsun 
        self.points[idcent][1,2] += radeg * vddLL[1] * dt /  self.points[idcent][1,0] / rsun
        
    def getvs(self, vmag):
        self.vs[0] = vmag
        rp = np.tan(self.AWp) * self.points[idcent][1,0] / (1. + self.deltaAx * np.tan(self.AWp))
        rr = rp * self.deltaCS
        alpha = np.sqrt(1 + 16 * self.deltaAx**2) / 4. / self.deltaAx
        Delta = rr / self.points[idcent][1,0]
        Lp = (np.tan(self.AW)*(1-Delta) - alpha * Delta) / (1 + self.deltaAx * np.tan(self.AW)) *  self.points[idcent][1,0]
        Lr = self.deltaAx * Lp
        # vs = [vFront, vEdge, vBulk, vexpBr, vexpBp, vexpA, vexpC]
        self.vs[3] = rr / self.points[idcent][1,0] * vmag
        self.vs[4] = rp / self.points[idcent][1,0] * vmag
        self.vs[5] = Lr / self.points[idcent][1,0] * vmag
        self.vs[6] = Lp / self.points[idcent][1,0] * vmag
        self.vs[2] = self.vs[0] - self.vs[3] - self.vs[5]
        self.vs[1] = self.vs[2] * np.tan(self.AW)

    # Called externally by programs using CME class, update the CME a time step
    def update_CME(self, user_vr, user_exp, user_mass):
    # Determines the forces and updates the CME accordingly.

        # Calculate the drag acting on the CME
        self.Fdrag = FC.calc_drag(self)
                
        # Calculate the deflection forces on the CME  
        FF.calc_forcesCPU(self)
		        
        # convert forces to accelerations
        self.get_center_acc()

        # calculate magnitude of vr
        vrmag = np.sqrt(np.sum(self.vels[0,:]**2))
        
        self.addDefDrag(vrmag)
        # account for solar rotation -> slips toward lower Carrington lon
        self.points[idcent][1,2] -= dt * radeg * FC.rotrate # add in degrees	
        # make sure doesn't go above 360
        self.points[idcent][1,2] = self.points[idcent][1,2] % 360.
        # move forward radially
        self.points[idcent][1,0] += vrmag * dt / rsun

        # rotate the CME using angular velocity
        #just need to change tilt before calc'ing the new points
        self.tilt += self.angvel * dt  
		        
        # Update the angular width using user_exp from ForeCAT.py
        # could eventually replace this forces updating CME lens
        self.AW = user_exp(self.points[idcent][1,0]) * dtor
        self.AWp = self.AWratio(self.points[idcent][1,0]) * self.AW 
        #self.deltaCSAx = self.AWp / self.AW / self.deltaCS
        self.calc_lens()
        self.deltaCSAx = self.rr / self.Lp
               		
        # determine new mass
        self.M = user_mass(self.points[idcent][1,0])

        # determine new density
        self.calc_rho()

        # determine new position of grid points with updated AW and cone pos
        self.calc_points() # this also updates GPU shape and position

        # get radial velocity at new distance 
        vmag, self.vels[0,:]= user_vr(self.points[idcent][1,0], self.rhat)
        self.getvs(vmag)
        # update time (in minutes)
        self.t += self.dt

        # check if abs(lat) greater than 89.5 since ForeCAT gets wonky at poles
        if np.abs(self.cone[1,1]) > 89.5:
            self.points[idcent][1,0] = 999999 # stop simulation
            print ("Hit pole, stopped simulation")
            print (self.cone[1,:])
