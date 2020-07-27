from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np

class PARADEres:
    "Container for results from a PARADE simulation"
    def __init__(self, fname, tau=1.):
        data = np.genfromtxt(fname, dtype=float)
        # i/60./24., rFront/rsun, AW*180/pi,  AWp*180/pi, a/rsun, br/rsun, bp/rsun, c/rsun, d/rsun, vFront/1e5, vEdge/1e5, vexpA/1e5, vexpBr/1e5, vexpBp/1e5, vexpC/1e5, vBulk/1e5, rho/1.67e-24, B0*1e5, cnm, rhoSWn/1.67e-24, BSWr*1e5, BSWphi*1e5, coeff1N, coeff2N, coeff1E, coeff2E, coeff3, coeff4, coeff4sw, dragR, dragC, dragBp
        self.t      = data[:,0]
        self.rFront = data[:,1]
        self.AW     = data[:,2]
        self.AWp    = data[:,3]
        self.a      = data[:,4]
        self.br     = data[:,5]
        self.bp     = data[:,6]
        self.b      = np.sqrt(self.br*self.bp)
        # Prints at shape ratio at end of time step not beginning 
        # will cause issues for converting CS forces so fix here
        self.delta  = self.br/self.bp 
        # Prints at shape ratio at end of time step not beginning 
        # will cause issues for converting CS forces so fix here
        #self.delta[1:] = self.delta[0:-1]
        #self.delta[0] = 1.
        self.c      = data[:,7]
        self.d      = data[:,8]
        self.CMEA   = self.a / self.c
        self.CMEBr  = self.br / self.c 
        self.CMEBp  = self.bp / self.c
        # calculate the radii of curvature
        kNose = 1.3726 * self.CMEA / self.c
        self.RcNose = 1./ kNose
        kEdge = 40 * self.CMEA / self.c / np.sqrt(16*self.CMEA+1)**3
        self.RcEdge = 1./ kEdge
        # Velocities
        self.vFront = data[:,9]
        self.vEdge  = data[:,10]
        self.vA     = data[:,11]
        self.vBr    = data[:,12]
        self.vBp    = data[:,13]
        self.vC     = data[:,14]
        self.vD     = data[:,15]
        # Density
        self.n      = data[:,16]
        # Magnetic Field
        self.B0     = data[:,17]
        self.cnm    = data[:,18]
        # calc Btor, Bpol
        self.Bt     = self.delta * tau * self.B0
        self.Bp     = 2 * self.delta**2 * self.B0 / (self.delta**2+1) / self.cnm
        self.nSW    = data[:,19]
        self.BswR   = data[:,20]
        self.BswP   = data[:,21]
        # to convert coeffs to accels
        self.toAcc  =  self.B0**2  / 1e10 / (self.delta * 3.14159 * self.bp*7e10 * self.n * 1.67e-24)
        # nose and edge accels    
        self.aPGN   = data[:,22] * self.toAcc # coeff of polodal grad at nose
        self.aTTN   = data[:,23] * self.toAcc # tor tension at nose
        self.aPGE   = data[:,24] * self.toAcc # pol grad at edge
        self.aTTE   = data[:,25] * self.toAcc # tor tension at edge
        # cross section accels
        self.aPT    = data[:,26] * self.toAcc # pol tension
        self.aTG    = data[:,27] * self.toAcc # tor grad
        self.aPTr   = self.aPT/self.delta
        self.aPTp   = self.aPT
        self.aTGr   = self.aTG/self.delta
        self.aTGp   = self.aTG
        # SW pressure grad
        self.aPSW   = data[:,28] 
        self.dragR  = data[:,29]
        self.dragE  = data[:,30]
        self.dragP  = data[:,31]

if __name__ == 'main':                

    plt.rcParams.update({'font.size':12})
    path = '/Users/ckay/OSPREI/ANTEATR/results/'
    name = 'temp'
    ensnum = 0
    res = PARADEres(path+'PARADE_'+name+'.dat')

    flagRange = True
    plotRange = False

    if name == 'A':
        flagRange = True
    if ensnum > 0:
        plotRange = True

    fig = plt.figure(figsize=(14,7))
    ax1 = plt.subplot(241)
    ax2 = plt.subplot(242, sharex=ax1)
    ax3 = plt.subplot(243, sharex=ax1)
    ax4 = plt.subplot(244, sharex=ax1)
    ax5 = plt.subplot(245, sharex=ax1)
    ax6 = plt.subplot(246, sharex=ax1)
    ax7 = plt.subplot(247, sharex=ax1)
    ax8 = plt.subplot(248, sharex=ax1)

    #ax3a = ax3.twinx()

    # set color names to make things easy 
    ind = '#332288'
    cyan = '#88CCEE'
    teal = '#44AA99'
    green = '#117733'
    wine = '#882255'
    rose = '#CC6677'

    colrp = '#660099'
    colrr = '#660033'
    colar = '#332288'
    colap = '#3399CC'
    cold  = 'darkgray'

    def logify(x):
        newx = np.sign(x)*np.log10(np.abs(100*x))
        badIDs = np.where(np.abs(x) < .01)
        newx[badIDs] = 0.
        return newx


    ax1.plot(res.rFront, res.AW, linewidth=3, color=colap, zorder=20)
    ax1.plot(res.rFront, res.AWp, linewidth=3, color=colrp, zorder=20)
    ax1.set_ylim([0,1.1*np.max(res.AW)])
    ax1.set_xlim([np.min(res.rFront)-5, np.max(res.rFront)+5])


    ax2.plot(res.rFront, res.CMEA, linewidth=3, color=colar, zorder=20)
    ax2.plot(res.rFront, res.delta, linewidth=3, color=colrr, zorder=20)
    '''ax2.plot(res.rFront, res.CMEBr, linewidth=3, color=colrr, zorder=20)
    ax2.plot(res.rFront, res.CMEBp, linewidth=3, color=colrp, zorder=20)
    ymin = np.min([np.min(res.CMEA), np.min(res.CMEBp)])
    ymax = np.max([np.max(res.CMEA), np.max(res.CMEBp)])'''

    ax3.plot(res.rFront, res.vFront, linewidth=3, color=colar, zorder=20)
    #ax3.plot(res.rFront, res.vA, linewidth=3, color=colar, zorder=20)
    ax3.plot(res.rFront, res.vBr, linewidth=3, color=colrr, zorder=20)
    ax3.plot(res.rFront, res.vBp, linewidth=3, color=colrp, zorder=20)
    ax3.plot(res.rFront, res.vEdge, linewidth=3, color=colap, zorder=20)
    #ax3.plot(res.rFront, res.vD, linewidth=3, color=cold, zorder=20)


    ax4.plot(res.rFront, np.log10(res.Bt), linewidth=3, color=colar, zorder=20)
    ax4.plot(res.rFront, np.log10(res.Bp), linewidth=3, color=colrr, zorder=20)
    ax4.plot(res.rFront, np.log10(res.BswR), linewidth=3, color='k', zorder=20)
    ax4.plot(res.rFront, np.log10(res.BswP), linewidth=3, color='darkgray', zorder=20)
    #ax4.set_yscale('log')

    ax5.plot(res.rFront, logify(res.aPGN), '-.', linewidth=3, color=colar, zorder=20)
    ax5.plot(res.rFront, logify(res.aTTN), '--', linewidth=3, color=colar, zorder=20)
    ax5.plot(res.rFront, logify(res.aPGE),  '-.', linewidth=3, color=colap, zorder=20)
    ax5.plot(res.rFront, logify(res.aTTE), '--', linewidth=3, color=colap, zorder=20)
    ax5.plot(res.rFront, logify(res.aPGN+res.aTTN), linewidth=3, color=colar, zorder=20)
    ax5.plot(res.rFront, logify(res.aPGE+res.aTTE), linewidth=3, color=colap, zorder=20)

    ax6.plot(res.rFront, logify(res.aPTr), '--', linewidth=3, color=colrr, zorder=20)
    ax6.plot(res.rFront, logify(res.aPTp), '--', linewidth=3, color=colrp, zorder=20)
    ax6.plot(res.rFront, logify(res.aTGr), '-.', linewidth=3, color=colrr, zorder=20)
    ax6.plot(res.rFront, logify(res.aTGp), '-.', linewidth=3, color=colrp, zorder=20)
    ax6.plot(res.rFront, logify(res.aPTr+res.aTGr), linewidth=3, color=colrr, zorder=20)
    ax6.plot(res.rFront, logify(res.aPTp+res.aTGp),  linewidth=3, color=colrp, zorder=20)
    #ax6.plot(res.rFront, res.fR,  linewidth=3, color=ind, zorder=20)
    #ax6.plot(res.rFront, res.fE,  linewidth=3, color=teal, zorder=20)
    #ax6.plot(res.rFront, res.fCS,  linewidth=3, color=wine, zorder=20)

    ax7.plot(res.rFront, logify(res.dragR), linewidth=3, color=colar)
    #ax7.plot(res.rFront, logify(res.dragR * res.vA/res.vFront), linewidth=3, color=colar)
    ax7.plot(res.rFront, logify(res.dragR * res.vBr/res.vFront), linewidth=3, color=colrr)
    #ax7.plot(res.rFront, logify(res.dragR * res.vD/res.vFront), linewidth=3, color=cold)
    ax7.plot(res.rFront, logify(res.dragP), linewidth=3, color=colrp)
    ax7.plot(res.rFront, logify(res.dragE), linewidth=3, color=colap)

    totNose = res.dragR + res.aPGN + res.aTTN
    ax8.plot(res.rFront, logify(totNose), linewidth=3, color=colar)
    totBr = res.aPTr + res.aTGr + res.dragR * res.vBr/res.vFront
    ax8.plot(res.rFront, logify(totBr), linewidth=3, color=colrr)
    #totD  = res.dragR * res.vD/res.vFront
    #ax8.plot(res.rFront, np.log10(np.abs(totD))*np.sign(totD), linewidth=3, color=cold)
    totBp = res.aPTp + res.aTGp + res.dragP
    ax8.plot(res.rFront, logify(totBp), linewidth=3, color=colrp)
    totE = res.aPGE + res.aTTE + res.dragE
    ax8.plot(res.rFront, logify(totE), linewidth=3, color=colap)
    #totA = totNose - totBr - totD
    #ax8.plot(res.rFront, np.log10(np.abs(totA))*np.sign(totA), linewidth=3, color=colar)

    pos = [0, 1, 2, 3, 4]
    labels = ['1', '10', '100', '1e3', '1e4' ]
    ax4.set_yticks(pos)
    ax4.set_yticklabels(labels)

    for ax in [ax5, ax6, ax7, ax8]:
        ax.plot([0,225], [0,0], 'k--', linewidth=1)
        rng = 7
        pos = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
        labels = ['-1e6','-1e5', '-1e4', '-1e3', '-100', '-10', '-1', '0', '1', '10', '100', '1e3', '1e4', '1e5', '1e6']
        ax.set_ylim(-rng-1,rng+1)
        for i in range(rng):
            ax.plot([0,225],[1+i, 1+i], 'k--', linewidth=0.25)
            ax.plot([0,225],[-1-i, -1-i], 'k--', linewidth=0.25)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
    
    

    ax1.set_xlabel('Distance (Rs)')
    ax2.set_xlabel('Distance (Rs)')
    ax3.set_xlabel('Distance (Rs)')
    ax4.set_xlabel('Distance (Rs)')
    ax5.set_xlabel('Distance (Rs)')
    ax6.set_xlabel('Distance (Rs)')
    ax7.set_xlabel('Distance (Rs)')
    ax8.set_xlabel('Distance (Rs)')
    ax1.set_ylabel('Angular Width ($^{\circ}$)')
    ax2.set_ylabel('Shape Ratio')
    ax3.set_ylabel('Velocity (km/s)')
    ax4.set_ylabel('B (nT)')
    ax5.set_ylabel('Acceleration (cm/s$^2$)')
    ax6.set_ylabel('Acceleration (cm/s$^2$)')
    ax7.set_ylabel('Acceleration (cm/s$^2$)')
    ax8.set_ylabel('Acceleration (cm/s$^2$)')

    ax1.set_ylim(0, 1.15*np.max(res.AW))
    ax1.text(0.4, 0.9, 'AW', fontweight='bold', color=colap, transform=ax1.transAxes, horizontalalignment='center')
    ax1.text(0.6, 0.9, 'AW$_{\perp}$', fontweight='bold', color=colrp, transform=ax1.transAxes, horizontalalignment='center')

    ax2.text(0.4, 0.9, '$\mathbf{\delta_a}$', color=colar, fontweight='bold', transform=ax2.transAxes, horizontalalignment='center')
    ax2.text(0.6, 0.9,  '$\mathbf{\delta_{CS}}$', color=colrr, fontweight='bold', transform=ax2.transAxes, horizontalalignment='center')
    '''ax2.text(0.35, 0.9, '$\mathbf{\epsilon_a}$', color=colar, fontweight='bold', transform=ax2.transAxes, horizontalalignment='center')
    ax2.text(0.5, 0.9,  '$\mathbf{\epsilon_r}$', color=colrr, fontweight='bold', transform=ax2.transAxes, horizontalalignment='center')
    ax2.text(0.65, 0.9,  '$\mathbf{\epsilon}_{\perp}$', color=colrp, fontweight='bold', transform=ax2.transAxes, horizontalalignment='center')'''

    ax3.text(0.29, 0.9, 'v$_{\mathbf{F}}$', fontweight='bold', color=colar, transform=ax3.transAxes, horizontalalignment='center')
    ax3.text(0.43, 0.9, 'v$_{\mathbf{E}}$', fontweight='bold', color=colap, transform=ax3.transAxes, horizontalalignment='center')
    ax3.text(0.57, 0.9, 'v$_{\mathbf{Er}}$', fontweight='bold', color=colrr, transform=ax3.transAxes, horizontalalignment='center')
    ax3.text(0.71, 0.9, 'v$_{\mathbf{E}\perp}$', fontweight='bold', color=colrp, transform=ax3.transAxes, horizontalalignment='center')

    ax4.text(0.2, 0.9, 'B$_{\mathbf{tor}}$', color=colar, fontweight='bold', transform=ax4.transAxes, horizontalalignment='center')
    ax4.text(0.4, 0.9, 'B$_{\mathbf{pol}}$', color=colrr, fontweight='bold', transform=ax4.transAxes, horizontalalignment='center')
    ax4.text(0.6, 0.9, 'B$_{\mathbf{SW,r}}$', color='k', fontweight='bold', transform=ax4.transAxes, horizontalalignment='center')
    ax4.text(0.83, 0.9, 'B$_{\mathbf{SW,\phi}}$', color='darkgray', fontweight='bold', transform=ax4.transAxes, horizontalalignment='center')

    ax5.text(0.35, 0.9, 'Nose', color=colar, fontweight='bold', transform=ax5.transAxes, horizontalalignment='center')
    ax5.text(0.65, 0.9, 'Edge', color=colap, fontweight='bold', transform=ax5.transAxes, horizontalalignment='center')
    ax5.plot([140,175], [-5,-5], '-.', linewidth=3, color='k')
    ax5.plot([140,175], [-6,-6], '--', linewidth=3, color='k')
    ax5.plot([140,175], [-7,-7], linewidth=3, color='k')
    ax5.text(185, -5, 'a$_{\mathbf{\\nabla}}$', fontweight='bold', horizontalalignment='left', verticalalignment='center')
    ax5.text(185, -6, 'a$_{\mathbf{\kappa}}$', fontweight='bold', horizontalalignment='left', verticalalignment='center')
    ax5.text(185, -7, 'a$_{\mathbf{tot}}$', fontweight='bold', horizontalalignment='left', verticalalignment='center')

    ax6.text(0.35, 0.9, 'Cross$_{\mathbf{r}}$', color=colrr, fontweight='bold', transform=ax6.transAxes, horizontalalignment='center')
    ax6.text(0.65, 0.9, 'Cross$_{\perp}$', color=colrp, fontweight='bold', transform=ax6.transAxes, horizontalalignment='center')
    ax6.plot([140,175], [-5,-5], '-.', linewidth=3, color='k')
    ax6.plot([140,175], [-6,-6], '--', linewidth=3, color='k')
    ax6.plot([140,175], [-7,-7], linewidth=3, color='k')
    ax6.text(185, -5, 'a$_{\mathbf{\\nabla}}$', fontweight='bold', horizontalalignment='left', verticalalignment='center')
    ax6.text(185, -6, 'a$_{\mathbf{\kappa}}$', fontweight='bold', horizontalalignment='left', verticalalignment='center')
    ax6.text(185, -7, 'a$_{\mathbf{tot}}$', fontweight='bold', horizontalalignment='left', verticalalignment='center')


    ax7.text(0.2, 0.9, 'Nose', color=colar, fontweight='bold', transform=ax7.transAxes, horizontalalignment='left')
    ax7.text(0.55, 0.9, 'Edge', color=colap, fontweight='bold', transform=ax7.transAxes, horizontalalignment='left')
    ax7.text(0.2, 0.82, 'Cross$_{\mathbf{r}}$', color=colrr, fontweight='bold', transform=ax7.transAxes, horizontalalignment='left')
    ax7.text(0.55, 0.82, 'Cross$_{\perp}$', color=colrp, fontweight='bold', transform=ax7.transAxes, horizontalalignment='left')
    ax7.text(200, -7, 'Drag', fontweight='bold', horizontalalignment='right', verticalalignment='center')

    ax8.text(0.2, 0.9, 'Nose', color=colar, fontweight='bold', transform=ax8.transAxes, horizontalalignment='left')
    ax8.text(0.55, 0.9, 'Edge', color=colap, fontweight='bold', transform=ax8.transAxes, horizontalalignment='left')
    ax8.text(0.2, 0.82, 'Cross$_{\mathbf{r}}$', color=colrr, fontweight='bold', transform=ax8.transAxes, horizontalalignment='left')
    ax8.text(0.55, 0.82, 'Cross$_{\perp}$', color=colrp, fontweight='bold', transform=ax8.transAxes, horizontalalignment='left')
    ax8.text(200, -7, 'Total', fontweight='bold', horizontalalignment='right', verticalalignment='center')



    plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.08, right=0.95, top=0.95, bottom=0.1)
    #plt.show()
    if not plotRange:
        plt.savefig(path+'PARADE_'+name+'2.png')
    else:
        plt.savefig(path+'PARADE_'+name+str(ensnum)+'.png')