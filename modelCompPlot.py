from plotPARADE import PARADEres 
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':12})
path = '/Users/ckay/OSPREI/ANTEATR/results/'
labels = ['$\delta_{cs}$','$\delta_{ax}$', 'v$_{F}$ (km/s)', "v$_{ex}$ (km/s)", 'AW ($^{\circ}$)', 'AW$_{\perp}$ ($^{\circ}$)', 'B$_0$ (nT)', 'C', 'v$_{ex \perp}$ (km/s)']

# Slow Cases
res1 = PARADEres(path+'PARADE_'+'SS1D'+'.dat')
res2 = PARADEres(path+'PARADE_'+'SS2D'+'.dat')
res3 = PARADEres(path+'PARADE_'+'SS3D'+'.dat')
res4 = PARADEres(path+'PARADE_'+'SS4D'+'.dat')
SSD = [res1, res2, res3, res4]
res1 = PARADEres(path+'PARADE_'+'SP1D'+'.dat')
res2 = PARADEres(path+'PARADE_'+'SP2D'+'.dat')
res3 = PARADEres(path+'PARADE_'+'SP3D'+'.dat')
res4 = PARADEres(path+'PARADE_'+'SP4D'+'.dat')
SPD = [res1, res2, res3, res4]
res1 = PARADEres(path+'PARADE_'+'SS10'+'.dat')
res2 = PARADEres(path+'PARADE_'+'SS20'+'.dat')
res3 = PARADEres(path+'PARADE_'+'SS30'+'.dat')
res4 = PARADEres(path+'PARADE_'+'SS40'+'.dat')
SS0 = [res1, res2, res3, res4]
res1 = PARADEres(path+'PARADE_'+'SP10'+'.dat')
res2 = PARADEres(path+'PARADE_'+'SP20'+'.dat')
res3 = PARADEres(path+'PARADE_'+'SP30'+'.dat')
res4 = PARADEres(path+'PARADE_'+'SP40'+'.dat')
SP0 = [res1, res2, res3, res4]

# Fast Cases
res1 = PARADEres(path+'PARADE_'+'FS1D'+'.dat')
res2 = PARADEres(path+'PARADE_'+'FS2D'+'.dat')
res3 = PARADEres(path+'PARADE_'+'FS3D'+'.dat')
res4 = PARADEres(path+'PARADE_'+'FS4D'+'.dat')
FSD = [res1, res2, res3, res4]
res1 = PARADEres(path+'PARADE_'+'FP1D'+'.dat')
res2 = PARADEres(path+'PARADE_'+'FP2D'+'.dat')
res3 = PARADEres(path+'PARADE_'+'FP3D'+'.dat')
res4 = PARADEres(path+'PARADE_'+'FP4D'+'.dat')
FPD = [res1, res2, res3, res4]
res1 = PARADEres(path+'PARADE_'+'FS10'+'.dat')
res2 = PARADEres(path+'PARADE_'+'FS20'+'.dat')
res3 = PARADEres(path+'PARADE_'+'FS30'+'.dat')
res4 = PARADEres(path+'PARADE_'+'FS40'+'.dat')
FS0 = [res1, res2, res3, res4]
res1 = PARADEres(path+'PARADE_'+'FP10'+'.dat')
res2 = PARADEres(path+'PARADE_'+'FP20'+'.dat')
res3 = PARADEres(path+'PARADE_'+'FP30'+'.dat')
res4 = PARADEres(path+'PARADE_'+'FP40'+'.dat')
FP0 = [res1, res2, res3, res4]

# Extreme Cases
res1 = PARADEres(path+'PARADE_'+'ES1D'+'.dat')
res2 = PARADEres(path+'PARADE_'+'ES2D'+'.dat')
res3 = PARADEres(path+'PARADE_'+'ES3D'+'.dat')
res4 = PARADEres(path+'PARADE_'+'ES4D'+'.dat')
ESD = [res1, res2, res3, res4]
res1 = PARADEres(path+'PARADE_'+'EP1D'+'.dat')
res2 = PARADEres(path+'PARADE_'+'EP2D'+'.dat')
res3 = PARADEres(path+'PARADE_'+'EP3D'+'.dat')
res4 = PARADEres(path+'PARADE_'+'EP4D'+'.dat')
EPD = [res1, res2, res3, res4]
res1 = PARADEres(path+'PARADE_'+'ES10'+'.dat')
res2 = PARADEres(path+'PARADE_'+'ES20'+'.dat')
res3 = PARADEres(path+'PARADE_'+'ES30'+'.dat')
res4 = PARADEres(path+'PARADE_'+'ES40'+'.dat')
ES0 = [res1, res2, res3, res4]
res1 = PARADEres(path+'PARADE_'+'EP10'+'.dat')
res2 = PARADEres(path+'PARADE_'+'EP20'+'.dat')
res3 = PARADEres(path+'PARADE_'+'EP30'+'.dat')
res4 = PARADEres(path+'PARADE_'+'EP40'+'.dat')
EP0 = [res1, res2, res3, res4]


c3 = '#660099'
c2 = '#660033'
c1 = '#332288'
c4 = '#3399CC'
cs = [c1,c2,c3,c4]

fig = plt.figure(figsize=(11,7))
ax1 = plt.subplot(231)
ax2 = plt.subplot(232, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(233, sharex=ax1, sharey=ax1)
ax4 = plt.subplot(234, sharex=ax1, sharey=ax1)
ax5 = plt.subplot(235, sharex=ax1, sharey=ax1)
ax6 = plt.subplot(236, sharex=ax1, sharey=ax1)
    
for i in range(4):
    ax4.plot(SSD[i].rFront, SSD[i].vBp, color=cs[i], linewidth=3)
    ax4.plot(SPD[i].rFront, SPD[i].vBp, '--', color=cs[i], linewidth=3)
    ax5.plot(FSD[i].rFront, FSD[i].vBp, color=cs[i], linewidth=3)
    ax5.plot(FPD[i].rFront, FPD[i].vBp, '--', color=cs[i], linewidth=3)
    ax6.plot(ESD[i].rFront, ESD[i].vBp, color=cs[i], linewidth=3)
    ax6.plot(EPD[i].rFront, EPD[i].vBp, '--', color=cs[i], linewidth=3)
    ax1.plot(SS0[i].rFront, SS0[i].vBp, color=cs[i], linewidth=3)
    ax1.plot(SP0[i].rFront, SP0[i].vBp, '--', color=cs[i], linewidth=3)
    ax2.plot(FS0[i].rFront, FS0[i].vBp, color=cs[i], linewidth=3)
    ax2.plot(FP0[i].rFront, FP0[i].vBp, '--', color=cs[i], linewidth=3)
    ax3.plot(ES0[i].rFront, ES0[i].vBp, color=cs[i], linewidth=3)
    ax3.plot(EP0[i].rFront, EP0[i].vBp, '--', color=cs[i], linewidth=3)

ax1.set_title('Slow', fontweight='bold', fontsize=12)    
ax2.set_title('Fast', fontweight='bold', fontsize=12)    
ax3.set_title('Extreme', fontweight='bold', fontsize=12)    
ax4.set_xlabel('Distance (R$_S$)')    
ax5.set_xlabel('Distance (R$_S$)')    
ax6.set_xlabel('Distance (R$_S$)')  
paramID = 8
ax1.set_ylabel(labels[paramID])
ax4.set_ylabel(labels[paramID])  
if paramID == 6:
    ax1.set_yscale('log')
    
# Add a legend
ax3.text(1.05, 0.2, 'Convective v$_0$', color='k', transform=ax3.transAxes, horizontalalignment='left', fontweight='bold')
ax3.text(1.05, 0.15, '--------------------', color='k', transform=ax3.transAxes, horizontalalignment='left')
ax3.text(1.05, 0.05, 'Self-Sim v$_0$', color='k', transform=ax3.transAxes, horizontalalignment='left', fontweight='bold')
ax3.text(1.05, 0.03, '______________', color='k', fontweight='bold', transform=ax3.transAxes, horizontalalignment='left')
ax3.text(1.05, -0.2, 'B Forces:', color='k', transform=ax3.transAxes, horizontalalignment='left', fontweight='bold')
ax3.text(1.05, -0.28, 'None', color=c1, transform=ax3.transAxes, horizontalalignment='left', fontweight='bold')
ax3.text(1.05, -0.35, 'CS $\\nabla$B Only', color=c2, transform=ax3.transAxes, horizontalalignment='left', fontweight='bold')
ax3.text(1.05, -0.42, 'Full CS Only', color=c3, transform=ax3.transAxes, horizontalalignment='left', fontweight='bold')
ax3.text(1.05, -0.49, 'CS and Axis', color=c4, transform=ax3.transAxes, horizontalalignment='left', fontweight='bold')
ax3.text(1.01, 0.9, 'No Drag', color='k', transform=ax3.transAxes, rotation=-90, fontweight='bold')
ax6.text(1.01, 0.17, 'Drag', color='k', transform=ax6.transAxes, rotation=-90, fontweight='bold')


plt.subplots_adjust(wspace=0.23, hspace=0.2, left=0.08, right=0.85, top=0.95, bottom=0.1)
plt.savefig('modelvExpP.png')
#plt.show()