# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FIDOgui.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import sys
import FIDO
import numpy as np

global Bout, tARR, Bsheath, tsheath, radfrac

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')    


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # CME Parameters Title
        self.CMElabel = QtWidgets.QLabel(self.centralwidget)
        self.CMElabel.setGeometry(QtCore.QRect(10, 10, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.CMElabel.setFont(font)
        # CME Latitude
        self.LatLab = QtWidgets.QLabel(self.centralwidget)
        self.LatLab.setGeometry(QtCore.QRect(10, 40, 90, 20))
        self.LatLE = QtWidgets.QLineEdit(self.centralwidget)
        self.LatLE.setGeometry(QtCore.QRect(10, 60, 90, 20))
        # CME Longitude
        self.LonLab = QtWidgets.QLabel(self.centralwidget)
        self.LonLab.setGeometry(QtCore.QRect(10, 85, 90, 20))
        self.LonLE = QtWidgets.QLineEdit(self.centralwidget)
        self.LonLE.setGeometry(QtCore.QRect(10, 105, 90, 20))
        # CME Tilt
        self.TiltLab = QtWidgets.QLabel(self.centralwidget)
        self.TiltLab.setGeometry(QtCore.QRect(10, 130, 90, 20))
        self.TiltLE = QtWidgets.QLineEdit(self.centralwidget)
        self.TiltLE.setGeometry(QtCore.QRect(10, 150, 90, 20))
        # Angular Width
        self.AWLab = QtWidgets.QLabel(self.centralwidget)
        self.AWLab.setGeometry(QtCore.QRect(10, 175, 90, 20))
        self.AWLE = QtWidgets.QLineEdit(self.centralwidget)
        self.AWLE.setGeometry(QtCore.QRect(10, 195, 90, 20))
        # Radial/bulk velocity
        self.vrLab = QtWidgets.QLabel(self.centralwidget)
        self.vrLab.setGeometry(QtCore.QRect(10, 220, 90, 20))
        self.vrLE = QtWidgets.QLineEdit(self.centralwidget)
        self.vrLE.setGeometry(QtCore.QRect(10, 240, 90, 20))
        # CME Shape Parameters
        self.shapeALab = QtWidgets.QLabel(self.centralwidget)
        self.shapeALab.setGeometry(QtCore.QRect(10, 265, 90, 20))
        self.shapeALE = QtWidgets.QLineEdit(self.centralwidget)
        self.shapeALE.setGeometry(QtCore.QRect(10, 285, 90, 20))        
        self.shapeBLab = QtWidgets.QLabel(self.centralwidget)
        self.shapeBLab.setGeometry(QtCore.QRect(10, 310, 90, 20))
        self.shapeBLE = QtWidgets.QLineEdit(self.centralwidget)
        self.shapeBLE.setGeometry(QtCore.QRect(10, 330, 90, 20))
        # Expansion mode
        self.ExpLab = QtWidgets.QLabel(self.centralwidget)
        self.ExpLab.setGeometry(QtCore.QRect(10, 375, 110, 20))
        self.ExpBox = QtWidgets.QComboBox(self.centralwidget)
        self.ExpBox.setGeometry(QtCore.QRect(10, 395, 90, 20))
        # Expansion Velocity 
        self.vExpLab = QtWidgets.QLabel(self.centralwidget)
        self.vExpLab.setGeometry(QtCore.QRect(10, 420, 90, 20))
        self.vExpLE = QtWidgets.QLineEdit(self.centralwidget)
        self.vExpLE.setGeometry(QtCore.QRect(10, 440, 90, 20))
        # Flux rope magnitude
        self.B0Lab = QtWidgets.QLabel(self.centralwidget)
        self.B0Lab.setGeometry(QtCore.QRect(10, 485, 90, 20))
        self.B0LE = QtWidgets.QLineEdit(self.centralwidget)
        self.B0LE.setGeometry(QtCore.QRect(10, 505, 90, 20))
        # Flux Rope polarity
        self.PolLab = QtWidgets.QLabel(self.centralwidget)
        self.PolLab.setGeometry(QtCore.QRect(10, 530, 90, 20))
        self.PolBox = QtWidgets.QComboBox(self.centralwidget)
        self.PolBox.setGeometry(QtCore.QRect(10, 550, 90, 20))        
        # CME (flux rope) start time
        self.StartLab = QtWidgets.QLabel(self.centralwidget)
        self.StartLab.setGeometry(QtCore.QRect(10, 595, 90, 20))
        self.StartLE = QtWidgets.QLineEdit(self.centralwidget)
        self.StartLE.setGeometry(QtCore.QRect(10, 615, 90, 20))
        # shift relative to start time
        self.tShiftLab = QtWidgets.QLabel(self.centralwidget)
        self.tShiftLab.setGeometry(QtCore.QRect(10, 640, 90, 20))
        self.tShiftLE = QtWidgets.QLineEdit(self.centralwidget)
        self.tShiftLE.setGeometry(QtCore.QRect(10, 660, 90, 20))
                
        # Plot windows
        self.graphB = pg.PlotWidget(self.centralwidget)
        self.graphB.setGeometry(QtCore.QRect(130, 50, 500, 150))  
        #self.graphB.hideAxis('bottom')     
        self.graphBx = pg.PlotWidget(self.centralwidget)
        self.graphBx.setGeometry(QtCore.QRect(130, 200, 500, 150))       
        #self.graphBx.hideAxis('bottom')     
        self.graphBy = pg.PlotWidget(self.centralwidget)
        self.graphBy.setGeometry(QtCore.QRect(130, 350, 500, 150))       
        #self.graphBy.hideAxis('bottom')     
        self.graphBz = pg.PlotWidget(self.centralwidget)
        self.graphBz.setGeometry(QtCore.QRect(130, 500, 500, 150))  
        global PWs
        PWs = [self.graphBx, self.graphBy, self.graphBz, self.graphB]
        
        # Spacecraft Values
        self.SClabel = QtWidgets.QLabel(self.centralwidget)
        self.SClabel.setGeometry(QtCore.QRect(655, 10, 131, 21))
        self.SClabel.setFont(font)
        # Spacecraft Latitude
        self.SCLatLab = QtWidgets.QLabel(self.centralwidget)
        self.SCLatLab.setGeometry(QtCore.QRect(655, 40, 90, 20))
        self.SCLatLE = QtWidgets.QLineEdit(self.centralwidget)
        self.SCLatLE.setGeometry(QtCore.QRect(655, 60, 90, 20))
        # Spacecraft Longitude
        self.SCLonLab = QtWidgets.QLabel(self.centralwidget)
        self.SCLonLab.setGeometry(QtCore.QRect(655, 85, 90, 20))
        self.SCLonLE = QtWidgets.QLineEdit(self.centralwidget)
        self.SCLonLE.setGeometry(QtCore.QRect(655, 105, 90, 20))
        # Spacecraft Distance
        self.SCRLab = QtWidgets.QLabel(self.centralwidget)
        self.SCRLab.setGeometry(QtCore.QRect(655, 130, 90, 20))
        self.SCRLE = QtWidgets.QLineEdit(self.centralwidget)
        self.SCRLE.setGeometry(QtCore.QRect(655, 150, 90, 20))
        # Spacecraft Orbit
        self.SCOrbLab = QtWidgets.QLabel(self.centralwidget)
        self.SCOrbLab.setGeometry(QtCore.QRect(655, 175, 90, 20))
        self.SCOrbLE = QtWidgets.QLineEdit(self.centralwidget)
        self.SCOrbLE.setGeometry(QtCore.QRect(655, 195, 90, 20))
                
        # Sheath Values
        self.Shlabel = QtWidgets.QLabel(self.centralwidget)
        self.Shlabel.setGeometry(QtCore.QRect(655, 250, 131, 21))
        self.Shlabel.setFont(font)
        self.ShBoxLab = QtWidgets.QLabel(self.centralwidget)
        self.ShBoxLab.setGeometry(QtCore.QRect(655, 280, 90, 20))
        self.ShBox = QtWidgets.QComboBox(self.centralwidget)
        self.ShBox.setGeometry(QtCore.QRect(645, 300, 105, 20)) 
        # Sheath Duration
        self.DurLab = QtWidgets.QLabel(self.centralwidget)
        self.DurLab.setGeometry(QtCore.QRect(655, 330, 90, 20))
        self.DurLE = QtWidgets.QLineEdit(self.centralwidget)
        self.DurLE.setGeometry(QtCore.QRect(655, 350, 90, 20))
        # Sheath compression
        self.CompLab = QtWidgets.QLabel(self.centralwidget)
        self.CompLab.setGeometry(QtCore.QRect(655, 375, 90, 20))
        self.CompLE = QtWidgets.QLineEdit(self.centralwidget)
        self.CompLE.setGeometry(QtCore.QRect(655, 395, 90, 20))
 
        # SW
        self.SWLab = QtWidgets.QLabel(self.centralwidget)
        self.SWLab.setGeometry(QtCore.QRect(655, 425, 90, 20))
        self.BxLab = QtWidgets.QLabel(self.centralwidget)
        self.BxLab.setGeometry(QtCore.QRect(655, 445, 25, 20))
        self.BxLE = QtWidgets.QLineEdit(self.centralwidget)
        self.BxLE.setGeometry(QtCore.QRect(680, 445, 65, 20))
        self.ByLab = QtWidgets.QLabel(self.centralwidget)
        self.ByLab.setGeometry(QtCore.QRect(655, 470, 25, 20))
        self.ByLE = QtWidgets.QLineEdit(self.centralwidget)
        self.ByLE.setGeometry(QtCore.QRect(680, 470, 65, 20))
        self.BzLab = QtWidgets.QLabel(self.centralwidget)
        self.BzLab.setGeometry(QtCore.QRect(655, 495, 25, 20))
        self.BzLE = QtWidgets.QLineEdit(self.centralwidget)
        self.BzLE.setGeometry(QtCore.QRect(680, 495, 65, 20))
        self.nLab = QtWidgets.QLabel(self.centralwidget)
        self.nLab.setGeometry(QtCore.QRect(655, 520, 25, 20))
        self.nLE = QtWidgets.QLineEdit(self.centralwidget)
        self.nLE.setGeometry(QtCore.QRect(680, 520, 65, 20))
        self.vLab = QtWidgets.QLabel(self.centralwidget)
        self.vLab.setGeometry(QtCore.QRect(655, 545, 25, 20))
        self.vLE = QtWidgets.QLineEdit(self.centralwidget)
        self.vLE.setGeometry(QtCore.QRect(680, 545, 65, 20))
        self.csLab = QtWidgets.QLabel(self.centralwidget)
        self.csLab.setGeometry(QtCore.QRect(655, 570, 25, 20))
        self.csLE = QtWidgets.QLineEdit(self.centralwidget)
        self.csLE.setGeometry(QtCore.QRect(680, 570, 65, 20))
        self.vALab = QtWidgets.QLabel(self.centralwidget)
        self.vALab.setGeometry(QtCore.QRect(655, 595, 25, 20))
        self.vALE = QtWidgets.QLineEdit(self.centralwidget)
        self.vALE.setGeometry(QtCore.QRect(680, 595, 65, 20))
        
        self.vTransLab = QtWidgets.QLabel(self.centralwidget)
        self.vTransLab.setGeometry(QtCore.QRect(655, 625, 90, 20))
        self.vTransLE = QtWidgets.QLineEdit(self.centralwidget)
        self.vTransLE.setGeometry(QtCore.QRect(655, 645, 90, 20))
        
        # Button for turining the wireframe on or off -----------------------------------|
        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveButton.setGeometry(QtCore.QRect(363, 655, 80, 25))
        
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FIDO"))
        self.CMElabel.setText(_translate("MainWindow", "CME"))
        self.LatLab.setText(_translate("MainWindow", "Latitude"))
        self.LonLab.setText(_translate("MainWindow", "Longitude"))
        self.TiltLab.setText(_translate("MainWindow", "Tilt"))
        self.AWLab.setText(_translate("MainWindow", "Ang Width"))
        self.vrLab.setText(_translate("MainWindow", "Bulk V"))
        self.ExpLab.setText(_translate("MainWindow", "Expansion Model"))
        self.vExpLab.setText(_translate("MainWindow", "Expansion v"))
        self.shapeALab.setText(_translate("MainWindow", "Shape A"))
        self.shapeBLab.setText(_translate("MainWindow", "Shape B"))
        self.B0Lab.setText(_translate("MainWindow", "FR B0"))
        self.PolLab.setText(_translate("MainWindow", "FR Polarity"))
        self.StartLab.setText(_translate("MainWindow", "FR Start Time"))
        self.tShiftLab.setText(_translate("MainWindow", "Time Shift"))
        self.SClabel.setText(_translate("MainWindow", "Spacecraft"))
        self.SCLatLab.setText(_translate("MainWindow", "Latitude"))
        self.SCLonLab.setText(_translate("MainWindow", "Longitude"))
        self.SCRLab.setText(_translate("MainWindow", "Distance (Rs)"))
        self.SCOrbLab.setText(_translate("MainWindow", "Orbit (deg/s)"))
        self.Shlabel.setText(_translate("MainWindow", "Sheath"))
        self.ShBoxLab.setText(_translate("MainWindow", "Add Sheath")) 
        self.DurLab.setText(_translate("MainWindow", "Duration (hr)"))
        self.CompLab.setText(_translate("MainWindow", "Compression"))
        self.SWLab.setText(_translate("MainWindow", "Solar Wind"))
        self.BxLab.setText(_translate("MainWindow", "Bx"))
        self.ByLab.setText(_translate("MainWindow", "By"))
        self.BzLab.setText(_translate("MainWindow", "Bz"))
        self.nLab.setText(_translate("MainWindow", "n"))
        self.vLab.setText(_translate("MainWindow", "v"))
        self.csLab.setText(_translate("MainWindow", "cs"))
        self.vALab.setText(_translate("MainWindow", "vA"))
        self.vTransLab.setText(_translate("MainWindow", "Transit Vel"))
        self.saveButton.setText(_translate("MainWindow", "Save"))
        
        
class mywindow(QtWidgets.QMainWindow):
    # This takes the generic but properly labeled window and adapts it to ---------------| 
    # our specific needs
    def __init__(self):
        # -------------------------------------------------------------------------------|            
        # ESSENTIAL GUI SETUP THAT I TOTALLY UNDERSTAND! --------------------------------|           
        super(mywindow, self).__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)
        
        # -------------------------------------------------------------------------------|            
        # Get a generic MainWindow then add our labels ----------------------------------|
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Set up plot windows
        self.plots = []
        self.zeroplots = []
        self.obsplots = []
        cols = ['k', 'b', 'r','k']
        for i in range(4):
            PW = PWs[i]
            pen = pg.mkPen(cols[i], width=8)
            # Start with things drawn first (-> FIDO added on top)
            # Add a dashed line at 0 for xyz frames
            plot2 = PW.plot()
            plot2.setPen(pg.mkPen(0.5,width=4,style=QtCore.Qt.DotLine))
            self.zeroplots.append(plot2)
            # Obs data
            plot3 = PW.plot()
            plot3.setPen(pg.mkPen(0.5,width=8))
            self.obsplots.append(plot3)
            # FIDO data
            plot = PW.plot()
            plot.setPen(pen)
            self.plots.append(plot)
        PWs[2].setLabels(bottom='Day of Year', left='Bz (nT)')
        PWs[0].setLabels(left='Bx (nT)')    
        PWs[1].setLabels(left='By (nT)')    
        PWs[3].setLabels(left='B (nT)')    
        
        # Add things to menu Boxes
        self.ui.ExpBox.addItems(["None", "Self-Similar", "vExp"])
        self.ui.ShBox.addItems(["No", "Dur/Comp", "From SW"])
        self.ui.PolBox.addItems(["+", "-"])
        
        # Pull in inputs from file if given one
        if len(sys.argv) == 2:
            FIDO.startFromText()
            
        # Check if given obs data file
        global Obs 
        Obs = None
        if FIDO.ObsData != None:
            ObsData = np.genfromtxt(FIDO.ObsData,dtype=float, encoding='utf8', skip_header=10)   
            Obst = ObsData[:,1]+ObsData[:,2]/24.
            Obs = [Obst, ObsData[:,4], ObsData[:,5], ObsData[:,6], ObsData[:,3]]
            
        
        # Put defaults/givens in the boxes
        self.ui.SCLatLE.setText(str(FIDO.inps[0]))
        self.ui.SCLonLE.setText(str(FIDO.inps[1]))
        self.ui.LatLE.setText(str(FIDO.inps[2]))
        self.ui.LonLE.setText(str(FIDO.inps[3]))
        self.ui.TiltLE.setText(str(FIDO.inps[4]))
        self.ui.AWLE.setText(str(FIDO.inps[5]))
        self.ui.shapeALE.setText(str(FIDO.inps[6]))
        self.ui.shapeBLE.setText(str(FIDO.inps[7]))
        self.ui.vrLE.setText(str(FIDO.inps[8]))
        self.ui.B0LE.setText(str(FIDO.inps[9]))        
        if np.sign(int(FIDO.inps[10])) == -1:
            self.ui.PolBox.setCurrentText('-')
        self.ui.tShiftLE.setText(str(FIDO.inps[11]))        
        self.ui.StartLE.setText(str(FIDO.inps[12]))        
        self.ui.vExpLE.setText(str(FIDO.inps[13]))        
        self.ui.SCRLE.setText(str(FIDO.inps[14]))        
        self.ui.SCOrbLE.setText(str(FIDO.inps[15]))   
        if FIDO.expansion_model == 'Self-Similar':
            self.ui.ExpBox.setCurrentText('Self-Similar')
        elif FIDO.expansion_model == 'vExp':
            self.ui.ExpBox.setCurrentText('vExp')
        if FIDO.hasSheath:
            if FIDO.calcSheath:
                self.ui.ShBox.setCurrentText('From SW')
            else:
                self.ui.ShBox.setCurrentText('Dur/Comp')
            self.ui.DurLE.setText('{:6.2f}'.format(FIDO.shinps[1]))
            self.ui.CompLE.setText('{:6.2f}'.format(FIDO.shinps[2]))
            self.ui.BxLE.setText('{:6.2f}'.format(FIDO.shinps[4]))
            self.ui.ByLE.setText('{:6.2f}'.format(FIDO.shinps[5]))
            self.ui.BzLE.setText('0')
            self.ui.nLE.setText(str(FIDO.moreShinps[0]))
            self.ui.vLE.setText(str(FIDO.moreShinps[1]))
            self.ui.csLE.setText(str(FIDO.moreShinps[2]))
            self.ui.vALE.setText(str(FIDO.moreShinps[3]))
            self.ui.vTransLE.setText(str(FIDO.moreShinps[4]))
            
        # Connect the boxes to actions    
        self.ui.SCLatLE.returnPressed.connect(self.recalcProfile)
        self.ui.SCLonLE.returnPressed.connect(self.recalcProfile)
        self.ui.LatLE.returnPressed.connect(self.recalcProfile)
        self.ui.LonLE.returnPressed.connect(self.recalcProfile)
        self.ui.TiltLE.returnPressed.connect(self.recalcProfile)
        self.ui.AWLE.returnPressed.connect(self.recalcProfile)
        self.ui.shapeALE.returnPressed.connect(self.recalcProfile)
        self.ui.shapeBLE.returnPressed.connect(self.recalcProfile)
        self.ui.vrLE.returnPressed.connect(self.recalcProfile)
        self.ui.B0LE.returnPressed.connect(self.recalcProfile)
        self.ui.tShiftLE.returnPressed.connect(self.recalcProfile)
        self.ui.StartLE.returnPressed.connect(self.recalcProfile)
        self.ui.vExpLE.returnPressed.connect(self.recalcProfile)
        self.ui.SCRLE.returnPressed.connect(self.recalcProfile)
        self.ui.SCOrbLE.returnPressed.connect(self.recalcProfile)
        self.ui.DurLE.returnPressed.connect(self.recalcProfile)
        self.ui.CompLE.returnPressed.connect(self.recalcProfile)
        self.ui.BxLE.returnPressed.connect(self.recalcProfile)
        self.ui.ByLE.returnPressed.connect(self.recalcProfile)
        self.ui.BzLE.returnPressed.connect(self.recalcProfile)
        self.ui.nLE.returnPressed.connect(self.recalcProfile)
        self.ui.vLE.returnPressed.connect(self.recalcProfile)
        self.ui.csLE.returnPressed.connect(self.recalcProfile)
        self.ui.vALE.returnPressed.connect(self.recalcProfile)
        self.ui.vTransLE.returnPressed.connect(self.recalcProfile)
        self.ui.PolBox.currentIndexChanged.connect(self.recalcProfile)
        self.ui.ExpBox.currentIndexChanged.connect(self.recalcProfile)
        self.ui.ShBox.currentIndexChanged.connect(self.recalcProfile)        
        self.ui.saveButton.clicked.connect(self.saveResults)
            
        self.recalcProfile()
            
    def recalcProfile(self):
        FIDO.inps[0] = float(self.ui.SCLatLE.text())
        FIDO.inps[1] = float(self.ui.SCLonLE.text())
        FIDO.inps[2] = float(self.ui.LatLE.text())
        FIDO.inps[3] = float(self.ui.LonLE.text())
        FIDO.inps[4] = float(self.ui.TiltLE.text())
        FIDO.inps[5] = float(self.ui.AWLE.text())
        FIDO.inps[6] = float(self.ui.shapeALE.text())
        FIDO.inps[7] = float(self.ui.shapeBLE.text())
        FIDO.inps[8] = float(self.ui.vrLE.text())
        FIDO.inps[9] = float(self.ui.B0LE.text())
        FIDO.inps[11] = float(self.ui.tShiftLE.text())
        FIDO.inps[12] = float(self.ui.StartLE.text())
        FIDO.inps[13] = float(self.ui.vExpLE.text())
        FIDO.inps[14] = float(self.ui.SCRLE.text())
        FIDO.inps[15] = float(self.ui.SCOrbLE.text())
        
        if self.ui.PolBox.currentText() == '+':
            FIDO.inps[10] = 1
        else:
            FIDO.inps[10] = -1
        
        if self.ui.ExpBox.currentText() == 'None':
            FIDO.expansion_model = 'None'
        elif self.ui.ExpBox.currentText() == 'Self-Similar':
            FIDO.expansion_model = 'Self-Similar'
        elif self.ui.ExpBox.currentText() == 'vExp':
            FIDO.expansion_model = 'vExp'
            
        if self.ui.ShBox.currentText() != 'No':
            FIDO.hasSheath = True
            FIDO.shinps[4] = float(self.ui.BxLE.text())
            FIDO.shinps[5] = float(self.ui.ByLE.text())
            FIDO.shinps[6] = float(self.ui.BzLE.text())
            if self.ui.ShBox.currentText() == 'Dur/Comp':
                FIDO.calcSheath = False
                FIDO.shinps[1] = float(self.ui.DurLE.text())
                FIDO.shinps[2] = float(self.ui.CompLE.text())
            elif self.ui.ShBox.currentText() == 'From SW':
                FIDO.calcSheath = True
                FIDO.moreShinps[0] = float(self.ui.nLE.text())
                FIDO.moreShinps[1] = float(self.ui.vLE.text())
                FIDO.moreShinps[2] = float(self.ui.csLE.text())
                FIDO.moreShinps[3] = float(self.ui.vALE.text())
                FIDO.moreShinps[4] = float(self.ui.vTransLE.text())
                vels = [FIDO.inps[8], FIDO.inps[13], FIDO.moreShinps[4], FIDO.moreShinps[1]]
                Bvec = [FIDO.shinps[4], FIDO.shinps[5], FIDO.shinps[6]]
                # recalculate things
                BSW = np.sqrt(FIDO.shinps[4]**2 + FIDO.shinps[5]**2 + FIDO.shinps[6]**2)
                FIDO.shinps = FIDO.calcSheathInps(FIDO.inps[12], vels, FIDO.moreShinps[0], BSW, FIDO.inps[14], B=Bvec, cs=FIDO.moreShinps[2], vA=FIDO.moreShinps[3])                             
        else:
            FIDO.hasSheath = False
        
        # recalculate with the new values
        global Bout, tARR, Bsheath, tsheath, radfrac
        Bout, tARR, Bsheath, tsheath, radfrac = FIDO.run_case(FIDO.inps, FIDO.shinps)
        for i in range(len(self.plots)):
            plot = self.plots[i]
            pw = PWs[i]
            if FIDO.hasSheath:
                plotx = np.append(tsheath, tARR)
                ploty = np.append(Bsheath[i], Bout[i])
            else:
                plotx, ploty = tARR, Bout[i]
            # Set plot range (must do first)
            pw.setRange(xRange=[plotx[0]-0.1, plotx[-1]+0.1], padding=0.01)
            # Plot zero lines
            if i != 3:
                self.zeroplots[i].setData([plotx[0]-0.5,plotx[-1]+0.5], [0,0])
            # Plot obs lines
            if Obs != None:
                self.obsplots[i].setData(Obs[0],Obs[i+1])
            # Plot FIDO results
            plot.setData(plotx, ploty)
            
    def saveResults(self):
        FIDO.saveResults(Bout, tARR, Bsheath, tsheath, radfrac)
        FIDO.saveRestartFile()
            
            
# Simple code to set up and run the GUI -------------------------------------------------|        
def runGUI():
    # Make an application
    app = QtWidgets.QApplication([])
    # Make a widget
    application = mywindow()
    # Run it
    application.show()
    # Exit nicely
    sys.exit(app.exec())

if __name__ == "__main__":
    runGUI()

'''if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_()) '''
