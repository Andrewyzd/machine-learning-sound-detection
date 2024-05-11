# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'loadingPage.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets
from Segmentation_FeaturesExtract import Segmentation, FeaturesExtraction
from PyQt5.QtWidgets import*
from PyQt5.QtGui import*
from PyQt5.QtCore import*
from threading import *
import threading
from PyQt5.uic import loadUi
from Interface3_AnalysisResult import ResultGUI
import sys


class LoadingActivity(QMainWindow):
    """
    A class for loading GUI window

    ...

    Attributes
    ----------
    signal : ndarray
        The array of storing the heart sound signal
    isRealTime : boolean
        The boolean to identify between the real-time analyse and non-realtime analysis
    information_collection : list
        The list to store the information of the login user
        
    Methods
    -------
    progress()
        Manage the element of the loading GUI
    progressBarValue
        Manage the progress value of the loading GUI
    """
    def __init__(self, signal, isRealTime, information_collection):
        """
        Parameters
        ----------
        signal : ndarray
            The array of storing the heart sound signal
        isRealTime : boolean
            The boolean to identify between the real-time analyse and non-realtime analysis
        information_collection : list
            The list to store the information of the login user

        Attributes
        ----------
        counter : int
            The increment of the progress bar to track the progress bar value
        jumper : int
            The increment value of the progress bar that is displayed to user
        """
        super().__init__()
        self.signal = signal
        self.isRealTime = isRealTime
        self.information_collection = information_collection
        self.counter = 0
        self.jumper = 0

        #Read ui file
        loadUi("loadingPage.ui",self)

        #remove standard title bar
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)#remove title bar
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)#set background to transparent

        #apply shadow effect
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0,0,0,200))
        self.circularBig.setGraphicsEffect(self.shadow)

        #Load and resize the image
        oImage = QImage("Icon/loding.png")
        sImage = oImage.scaled(QSize(260,260)) 
        self.label_image.setPixmap(QtGui.QPixmap(sImage))
        self.label_image.setAlignment(Qt.AlignCenter)

        #Initialize and start the timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)
        self.timer.start(50)

        #Design the circular progress bar
        self.circularProgress.setStyleSheet("QFrame{" +
                                            "border-radius: 150px;" +
                                            "background-color:qconicalgradient(cx:0.5, cy:0.5, angle:90.2, stop:0.0 rgba(0, 0, 127, 0), stop:0.0 rgba(69, 139, 103, 255));}")
        #Initialize the progress bar value by 0
        self.progressBarValue(0)


    def progress(self):
        """
        Manage the element of the loading GUI
        """
        #Initialize the value
        value = self.counter

        #Change the text of the progress bar
        QtCore.QTimer.singleShot(1500, lambda: self.label_textChanging.setText("""<p align='center'><span style="font-size:9pt;color:#cd6155;">
                                                                                <strong>Control the code, control the health</strong></span></p>"""))
        QtCore.QTimer.singleShot(4500, lambda: self.label_textChanging.setText("""<p align='center'><span style="font-size:9pt;color:#cd6155;">
                                                                                <strong>Be aware Be save</strong></span></p>"""))

        QtCore.QTimer.singleShot(1, lambda: self.label_processing.setText("<p align='center'>processing..</p>"))
        QtCore.QTimer.singleShot(1, lambda: self.label_processing.setText("<p align='center'>processing...</p>"))
        QtCore.QTimer.singleShot(1, lambda: self.label_processing.setText("<p align='center'>processing....</p>"))

        #Design the text
        progressText = """<p align="center"><span style="font-size:11pt; verticle-align:super; color:#117864;">[value]</span>
                        <span style="font-size:11pt; verticle-align:super; color:#117864;">%</span></p>"""

        #replace value in html
        newProgressText = progressText.replace("[value]", str(int(self.jumper)))

        #Increase the jump value by 1
        if (value > self.jumper):
            self.label_percentage.setText(newProgressText)
            self.jumper+=1
                
        #reset the value, counter, and jumper
        if value >= 100:
            value = 1.000

        #set value to progress bar
        self.progressBarValue(value)

        #close and open screen
        if self.counter > 100:
            self.timer.stop()#stop for a second
            #open the result gui window
            self.resultWindow = ResultGUI(self.signal, self.isRealTime, self.information_collection)
            self.resultWindow.run()
            
            #set the progress bar value to 100
            self.progressBarValue(100)
            self.timer.stop()#stop for a second
            #close the loading page
            self.close()

        #Increase the counter by 0.3
        self.counter+=0.3
        

    def progressBarValue(self, value):
        """
        Manage the progress value of the loading GUI
        """
        #progress bar stylesheet
        styleSheet = """QFrame{border-radius: 150px;
                        background-color:qconicalgradient(cx:0.5, cy:0.5, angle:90.2,
                        stop:[STOP_1] rgba(0, 0, 127, 0), stop:[STOP_2] rgba(0, 255, 232, 255));}"""

        #get progress bar value, convert to float and invert values
        #stop works of 1000 to 0.000
        progress = (100 - value) / 100.0

        #Convert the value to str
        stop_1 = str(progress - 0.001)
        stop_2 = str(progress)

        # set values to stylesheet
        new_styleSheet = styleSheet.replace("[STOP_1]", stop_1).replace("[STOP_2]", stop_2)

        #apply stylesheet with new values
        self.circularProgress.setStyleSheet(new_styleSheet)

class LoadingProgress(QtCore.QThread):
    """
    A class for loading GUI window

    ...

    Attributes
    ----------
    signal : ndarray
        The array of storing the heart sound signal
    isRealTime : boolean
        The boolean to identify between the real-time analyse and non-realtime analysis
    information_collection : list
        The list to store the information of the login user

    Methods
    -------
    run()
        Start the thread
    """
    def __init__(self, signal, isRealTime, information_collection):
        """
        Parameters
        ----------
        signal : ndarray
            The array of storing the heart sound signal
        isRealTime : boolean
            The boolean to identify between the real-time analyse and non-realtime analysis
        information_collection : list
            The list to store the information of the login user
        """
        super(LoadingProgress, self).__init__()
        self.signal = signal
        self.isRealTime = isRealTime
        self.information_collection = information_collection

    def run(self):
        """
        Initialize and start the loading activity interface window
        """
        self.loading_window = LoadingActivity(self.signal, self.isRealTime, self.information_collection)
        self.loading_window.show()#show the interface
