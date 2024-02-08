import numpy as np
import sys
import datetime


# OSPREI wants format:
#Time                            R [AU]          lat [deg]       lon [deg]
#2022-01-26 00:00:00             0.69215         3.81529         -99.39884
#where lon can be any inertial coord sys (but Stony makes my brain hurt least)

# helioWeb files are in
#YEAR DAY  RAD_AU  HGI_LAT  HGI_LON
#1990   1    0.98    -3.0    24.5

myPaths = np.genfromtxt('myPaths.txt', dtype=str)
heliopath = myPaths[3,1]
earthHGI = np.genfromtxt(heliopath+'planets/earth.lst', dtype=float, skip_header=1)
earthCar = np.genfromtxt(heliopath+'planets/earthCar.lst', dtype=float, skip_header=1)


def getSatLoc(satName, start, stop, pad=30, fracYr=True, outName=None, satpath0=heliopath):
    #satpath0 = 'helioweb/'
    planetlist = ['earth', 'jupiter', 'mars', 'mercury', 'neptune', 'pluto', 'saturn', 'uranus', 'venus']
    satlist = ['bepicolombo', 'cassini', 'dawn', 'galileo', 'helios1', 'helios2', 'juno', 'maven', 'messenger', 'msl', 'newhorizons', 'osirisrex', 'psp', 'rosetta', 'solo', 'spitzer', 'stereoa', 'stereob', 'ulysses']
    
    if satName in planetlist:
        satpath = satpath0 + 'planets/'
    elif satName in satlist:
        satpath = satpath0 + 'spacecraft/'
    else:
        sys.exit('Unknown satellite name')
    
    # read in appropriate helioweb file    
    hdata = np.genfromtxt(satpath+satName+'.lst', dtype=float, skip_header=1)
    
    # get year and doy est
    if fracYr:
        yr1 = int(start)
        yr2 = int(stop)
        doy1 = (start-yr1)*365 
        doy2 = (stop-yr2)*365
        stDT = datetime.datetime(year=yr1, month=1, day=1) + datetime.timedelta(days=doy1-1)
        endDT = datetime.datetime(year=yr1, month=1, day=1) + datetime.timedelta(days=doy1-1)
    # string yr
    else:
        lenSt = len(start)
        if (start[4] == '/') and (lenSt in [10,16]):
            if len(start) == 10:
                fmt = '%Y/%m/%d'
            elif len(start) == 16:
                fmt = '%Y/%m/%d %H:%M'
        elif (start[4] == ':') and (lenSt in [10,16]):
            if len(start) == 10:
                fmt = '%Y-%m-%d'
            elif len(start) == 16:
                fmt = '%Y-%m-%d %H:%M'
        else:
            sys.exit('Unknown format for start/stop time. Use either YYYY/MM/DD or YYYY-MM-DD (can also include HH:MM)')
        try:
            stDT = datetime.datetime.strptime(start, fmt)
            endDT = datetime.datetime.strptime(stop, fmt)
        except:
            sys.exit('Unknown format for start/stop time. Use either YYYY/MM/DD or YYYY-MM-DD')
    
    
    stDTp = stDT + datetime.timedelta(days=-pad)
    endDTp = endDT + datetime.timedelta(days=pad)
    
    fracyr = hdata[:,0] + hdata[:,1]/365.
    fracSt = stDTp.year + stDTp.timetuple().tm_yday/365.
    fracEnd = endDTp.year + endDTp.timetuple().tm_yday/365.
    idx = np.where((fracyr >= fracSt) & (fracyr <= fracEnd))[0]
    
    fname = satName+'_temp.sat', 'w'
    if outName:
        fname = satName+'_'+outName+'.sat'
    f1 = open(fname, 'w')
    
    f1.write('#Time          R [AU]   Sto_lat [deg]   Sto_lon [deg] \n')

    for i in idx:    
        thisDT = datetime.datetime(year=int(hdata[i,0]), month=1, day=1) + datetime.timedelta(hdata[i,1]-1)
        # Convert to not HGI (=HCI) coordinates
        # Get the Earth coords at that time
        # interp version, avoids sunpy
        HGIidx = np.where((earthHGI[:,0]==hdata[i,0]) & (earthHGI[:,1]==hdata[i,1]))[0][0]        
        ElonHGI = earthHGI[HGIidx, 4]

        # adjust sat lon from HGI to Stony     
        Sto_lon = (hdata[i,4] - ElonHGI) % 360.
        if Sto_lon > 180: Sto_lon -=360.
        if satName == 'earth':
            Sto_lon =0.
        
        outstr = thisDT.strftime('%Y-%m-%d %H:%M:%S') + ' ' + str(hdata[i,2]) + ' '  + str(hdata[i,3])+ ' '  + '{:.2f}'.format(Sto_lon) +  '\n'
        f1.write(outstr)
    f1.close()
        
    return fname
    
def getEarthClon(date):
    # Takes in a date time object    
    Caridx = np.where((earthCar[:,0]==date.year) & (earthCar[:,1]==date.timetuple().tm_yday))[0][0]
    fracDay = (date -  datetime.datetime(year=date.year, month=date.month, day=date.day)).total_seconds() / (24.*3600)
    lonA, lonB = float(earthCar[Caridx,4]), float(earthCar[Caridx+1,4])
    carLon = lonA * (1-fracDay) + lonB * fracDay
    return carLon
    
    