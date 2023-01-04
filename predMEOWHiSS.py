import matplotlib.pyplot as plt
import numpy as np
import empHSS as emp
import datetime

dtor = np.pi / 180.
plt.rcParams['font.size'] = '12'



def makeHSSprofile(HSSsize):
    t0, xs0, vs, a1 = emp.getxmodel(HSSsize, 0.0)

    vfuncs = emp.getvmodel(HSSsize)
    nfuncs = emp.getnmodel(HSSsize)
    Brfuncs = emp.getBrmodel(HSSsize)
    Blonfuncs = emp.getBlonmodel(HSSsize)
    Tfuncs = emp.getTmodel(HSSsize)
    vlonfuncs = emp.getvlonmodel(HSSsize)

    nt = 24 * 10
    dt = 1.
    allpoints = np.zeros([nt, 6])
    allregs = np.zeros(nt)
    for i in range(nt):
        thisR = 213/215.
        thispoint, reg = emp.getHSSprops(thisR, t0+i*dt, t0, xs0, vs, a1, [nfuncs, vfuncs, Brfuncs, Blonfuncs, Tfuncs, vlonfuncs], returnReg=True)
        allpoints[i, :] = thispoint
        allregs[i] = reg
    return allpoints, allregs
    


def predfromCH(HSSsize, gridscale=False, userAmb=None, plotdefAmb=True, t0=None, omniname=None, shadezones=True, plotstart=None, plotstop=None, obsstart=None, obsstop=None, savename=None):
    # HSS size in 1e8 km^2 or flag gridscale to use JHelio 5x5 grid
    
    # default ambient sw params
    # n v Bx By T vlon (vlon unused but scales to km)
    # midpoint in ranges from Owens 'Solar-Wind Structure' article (oxford)
    # assume Bx = By = |B\ / sqrt(2)
    ambSWdef = [7.5, 350, 4.2, 4.2, 75000, 1e-5]
    
    # check if given userAmb and add the 1e-5 on the end
    if userAmb:        
        userAmb = userAmb +[1e-5]
    
    # update the area if using gridscale
    if gridscale:
        HSSsize *= 37.3 # using 5x5 deg grid on Sun
        
    # get the scaled HSS profile    
    data, regs = makeHSSprofile(HSSsize)
    
    # check if given t0 and convert generic hours to dates
    time0 = 0
    nx = len(data[:,0])
    if t0:
        time0 = datetime.datetime.strptime(t0, '%Y/%m/%d %H:%M')
        # want to shift times so that MH calcs on the hour to match OMNI
        offset = time0.minute / 60.
        
        MHtimes = []
        for i in range(nx):
            thistime = time0 + datetime.timedelta(hours=(i+offset))
            MHtimes.append(thistime)
        MHtimes = np.array(MHtimes) 
    else:   
        MHtimes = np.array(range(nx))/24
        
    # check if given an obs data file
    # meant to read in OMNI data - year, DoY, hr, Bx, By, Bz, T, n, v, flow lon, flow lat
    if omniname:
        omnidata = np.genfromtxt(omniname, dtype=float)
        vr = omnidata[:,8] * np.cos(omnidata[:,9]*dtor) * np.cos(omnidata[:,10]*dtor)
        vlon = omnidata[:,8] * np.cos(omnidata[:,9]*dtor) * np.sin(omnidata[:,10]*dtor)
        omniSW = [omnidata[:,7], vr, omnidata[:,3], omnidata[:,4], omnidata[:,6], vlon]
        base = datetime.datetime(int(omnidata[0][0]), 1, 1, 0, 0)
        # add in FC time (if desired)
        SWt = omnidata[:,1] + omnidata[:,2] / 24.
        obstimes = np.array([base + datetime.timedelta(days=i-1) for i in SWt])
        
    # set up shading of HSS zones
    if shadezones:
        # get reg times
        zones = [5,4,3,2,1,0]
        zbounds = []
        for zone in zones:
            zbounds.append(MHtimes[np.min(np.where(regs == zone)[0])])
        c1 =  '#26C6DA'
        c2 = '#FF6F00'
        fills = [c1, c2, c1, c2, c1]
        
    # read in observed HSS stop/start (if given)
    if obsstart:
        obsstart = datetime.datetime.strptime(obsstart, '%Y/%m/%d %H:%M')
    if obsstop:
        obsstop = datetime.datetime.strptime(obsstop, '%Y/%m/%d %H:%M')
        
        
    # set up figure
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(6.5,8))
    colors = plt.cm.magma(np.linspace(0.1,0.8,7))
    labels = ['n cm$^{-3}$', 'v$_r$ (km/s)', 'B$_r$ (nT)', 'B$_{lon}$ (nT)', 'T (10$^5$ K)', 'v$_{Lon}$ (km/s)' ]
    #normFactors = [1.67e-24, 1e5, 1e-5, 1e-5, 1, 1e5]
    normFactors = [1, 1, 1, 1, 1e5, 1]
    
    for i in range(6):
        if omniname:
            axes[i].plot(obstimes, np.abs(omniSW[i])/normFactors[i], 'k', lw=3)
        if plotdefAmb:
            axes[i].plot(MHtimes, np.abs(data[:,i]*ambSWdef[i])/normFactors[i], '--', color='b', lw=3)
        if userAmb:
            axes[i].plot(MHtimes, np.abs(data[:,i]*userAmb[i])/normFactors[i], color= 'r', lw=3)
        
        ylims = axes[i].get_ylim()    
        if shadezones:
            # add shading
            for j in range(5):
                axes[i].fill_between([zbounds[j], zbounds[j+1]], [ylims[0], ylims[0]], y2=[ylims[1], ylims[1]],color = fills[j], alpha=0.15, zorder=0)  
        axes[i].set_ylabel(labels[i])
        
        if obsstart:
            axes[i].plot([obsstart, obsstart], [ylims[0], ylims[1]], 'k--')
        if obsstop:
            axes[i].plot([obsstop, obsstop], [ylims[0], ylims[1]], 'k--')
            
        axes[i].set_ylim(ylims)
    
    # check if given plotstart/stop, should be in same datetime str format as t0
    xlims = axes[i].get_xlim()
    if plotstart and plotstop:
        pltstart = datetime.datetime.strptime(plotstart, '%Y/%m/%d %H:%M')
        pltstop = datetime.datetime.strptime(plotstop, '%Y/%m/%d %H:%M')
        axes[i].set_xlim([pltstart, pltstop])
    
    # finish setting up figure and save      
    if t0:
        axes[-1].set_xlabel('Time')
    else:
        axes[-1].set_xlabel('Time (days)')
    xticks = axes[-1].get_xticks()
    axes[-1].set_xticks(xticks[::2])
    fig.subplots_adjust(top=0.95, right=0.95, left=0.12)
    
    
    # if given both stop and start then calculate the accuracy
    if obsstart and obsstop:
         MHidx = np.where((MHtimes >= obsstart) & (MHtimes <= obsstop))[0]       
         obsidx = np.where((obstimes >= obsstart) & (obstimes <= obsstop))[0]   
         for i in range(6):
            outline = labels[i].ljust(20)
            if userAmb:
                # unweighted error
                unw8 = np.mean(np.abs(np.abs(data[MHidx,i]*userAmb[i] - np.abs(omniSW[i][obsidx]))/normFactors[i]))
                # weighted error
                w8   = np.mean(np.abs(np.abs(data[MHidx,i]*userAmb[i] - np.abs(omniSW[i][obsidx])))/np.mean(np.abs(omniSW[i][obsidx])))
                # add to print line
                outline = outline + 'userAmb: ' '{:6.2f}'.format(unw8)  + '{:6.3f}'.format(w8) + '   '
            if plotdefAmb:
                # unweighted error
                unw8 = np.mean(np.abs(np.abs(data[MHidx,i]*ambSWdef[i] - np.abs(omniSW[i][obsidx]))/normFactors[i]))
                # weighted error
                w8   = np.mean(np.abs(np.abs(data[MHidx,i]*ambSWdef[i] - np.abs(omniSW[i][obsidx])))/np.mean(np.abs(omniSW[i][obsidx])))
                # add to print line
                outline = outline + 'defAmb: ' '{:6.2f}'.format(unw8)  + '{:6.3f}'.format(w8)
            print (outline)
         print ('')
    
    bonusPrint = ''
    if not t0:
        bonusPrint = ' days'
    print ('First contact at: ', MHtimes[np.min(np.where(regs == 5))], bonusPrint)
    print ('HSS front at:     ', MHtimes[np.min(np.where(regs == 3))], bonusPrint)
    print ('Final contact at: ', MHtimes[np.max(np.where(regs == 1))], bonusPrint)
    print ('')
    print ('')
    
    if savename:
        plt.savefig(savename)    
    else:
        plt.show()
    plt.close()
    
    
#predfromCH(28, gridscale=True)  

# Sep 2016  
predfromCH(34, gridscale=True, userAmb=[8, 330, 2.5, 3.5, 50000], t0='2016/09/16 13:30', omniname='omniSep2016.dat', plotstart='2016/09/18 13:30', plotstop='2016/09/23 18:30', obsstart='2016/09/19 04:00', obsstop='2016/09/22 22:00', savename='MHpred_Sep16.png')

# July 2017 
predfromCH(52, gridscale=True, userAmb=[6, 330, 2, -2, 40000, 1e-5], t0='2017/07/05 19:30', omniname='omniJul2017.dat', plotstart='2017/07/07 12:00', plotstop='2017/07/14 12:00', obsstart='2017/07/09 23:00', obsstop='2017/07/12 18:00', savename='MHpred_Jul17.png')

# Aug 2017 
predfromCH(50, gridscale=True, userAmb=[4.5, 350, -2, -3, 40000, 1e-5], t0='2017/07/31 20:00', omniname='omniAug2017.dat', plotstart='2017/08/02 12:00', plotstop='2017/08/09 12:00', obsstart='2017/08/03 10:00', obsstop='2017/08/08 04:00', savename='MHpred_Aug17.png')

# Nov 2017 
predfromCH(26, gridscale=True, userAmb=[10, 330, -1, -1.5, 30000, 1e-5], t0='2017/11/17 14:00', omniname='omniNov2017.dat', plotstart='2017/11/20 02:00', plotstop='2017/11/24 20:00', obsstart='2017/11/20 12:00', obsstop='2017/11/24 13:00', savename='MHpred_Nov17.png')