from time import sleep
from threading import *
from pyAudioAnalysis import ShortTermFeatures as stf
from Segmentation_FeaturesExtract import Segmentation, CSV
import librosa, librosa.display
import numpy as np
import pandas as pd
import noisereduce as nr
import time
import math
import csv
import os

class Process_normal(Thread):
    """
    A class for a thread to process the normal heart sound signal

    Attributes
    ----------
    directory : str
        The directory path to the normal heart sound audio dataset
    column_names : list
        The column name of the features attributes
    file_name : str
        The file name to save the features value of the audio file
    identifier : int
        The label to differentiate between four types of heart sound

    Methods
    -------
    run()
        Start the thread for processing
    pathName(filename, directory)
        Pass the audio file path for segmentation
    """
    def __init__(self, directory, column_names, file_name, identifier):
        """
        Parameters
        ----------
        directory : str
            The directory path to the normal heart sound audio dataset
        column_names : list
            The column name of the features attributes
        file_name : str
            The file name to save the features value of the audio file
        identifier : int
            The label to differentiate between four types of heart sound
        """
        super(Process_normal, self).__init__()
        self.directory = directory
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier

    def run(self):
        """
        Start the thread for processing
        Iterating through the normal heart sound audio dataset
        """
        for filename in os.listdir(self.directory):
            self.pathName(filename, self.directory)
            time.sleep(1)#stop for 1 second
            
    def pathName(self, filename, directory):
        """
        Parameters
        ----------
        filename : str
            The audio file name in .WAV format
        directory : str
            The directory path to the audio file name
        """
        print("\nStart processing file: "+directory +"/"+ filename)
        filePath = directory +"/"+ filename
        #Initialize the Segmentation class
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, 0)
        #Start segmentation
        segmentation.loadFile(filePath, proceed=True)
        

class Process_murmur(Thread):
    """
    A class for a thread to process the murmur heart sound signal

    Attributes
    ----------
    directory : str
        The directory path to the murmur heart sound audio dataset
    column_names : list
        The column name of the features attributes
    file_name : str
        The file name to save the features value of the audio file
    identifier : int
        The label to differentiate between four types of heart sound

    Methods
    -------
    run()
        Start the thread for processing
    pathName(filename, directory)
        Pass the audio file path for segmentation
    """
    def __init__(self, directory, column_names, file_name, identifier):
        """
        Parameters
        ----------
        directory : str
            The directory path to the murmur heart sound audio dataset
        column_names : list
            The column name of the features attributes
        file_name : str
            The file name to save the features value of the audio file
        identifier : int
            The label to differentiate between four types of heart sound
        """
        super(Process_murmur, self).__init__()
        self.directory = directory
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier

    def run(self):
        """
        Start the thread for processing
        Iterating through the murmur heart sound audio dataset
        """
        for filename in os.listdir(self.directory):
            self.pathName(filename, self.directory)
            time.sleep(1)#stop for 1 second
            
    def pathName(self, filename, directory):
        """
        Parameters
        ----------
        filename : str
            The audio file name in .WAV format
        directory : str
            The directory path to the audio file name
        """
        print("\nStart processing file: "+directory +"/"+ filename)
        filePath = directory +"/"+ filename
        #Initialize the Segmentation class
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, 0)
        #Start segmentation
        segmentation.loadFile(filePath, proceed=True)
        

class Process_extrahls(Thread):
    """
    A class for a thread to process the extra heart sound signal

    Attributes
    ----------
    directory : str
        The directory path to the extra heart sound audio dataset
    column_names : list
        The column name of the features attributes
    file_name : str
        The file name to save the features value of the audio file
    identifier : int
        The label to differentiate between four types of heart sound

    Methods
    -------
    run()
        Start the thread for processing
    pathName(filename, directory)
        Pass the audio file path for segmentation
    """
    def __init__(self, directory, column_names, file_name, identifier):
        """
        Parameters
        ----------
        directory : str
            The directory path to the extra heart sound audio dataset
        column_names : list
            The column name of the features attributes
        file_name : str
            The file name to save the features value of the audio file
        identifier : int
            The label to differentiate between four types of heart sound
        """
        super(Process_extrahls, self).__init__()
        self.directory = directory
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier

    def run(self):
        """
        Start the thread for processing
        Iterating through the extra heart sound audio dataset
        """
        for filename in os.listdir(self.directory):
            self.pathName(filename, self.directory)
            time.sleep(1)#stop for 1 second
            
    def pathName(self, filename, directory):
        """
        Parameters
        ----------
        filename : str
            The audio file name in .WAV format
        directory : str
            The directory path to the audio file name
        """
        print("\nStart processing file: "+directory +"/"+ filename)
        filePath = directory +"/"+ filename
        #Initialize the Segmentation class
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, 0)
        #Start segmentation
        segmentation.loadFile(filePath, proceed=True)


class Process_extrastole(Thread):
    """
    A class for a thread to process the extrasystole heart sound signal

    Attributes
    ----------
    directory : str
        The directory path to the extrasystole heart sound audio dataset
    column_names : list
        The column name of the features attributes
    file_name : str
        The file name to save the features value of the audio file
    identifier : int
        The label to differentiate between four types of heart sound

    Methods
    -------
    run()
        Start the thread for processing
    pathName(filename, directory)
        Pass the audio file path for segmentation
    """
    def __init__(self, directory, column_names, file_name, identifier):
        """
        Parameters
        ----------
        directory : str
            The directory path to the extrasystole heart sound audio dataset
        column_names : list
            The column name of the features attributes
        file_name : str
            The file name to save the features value of the audio file
        identifier : int
            The label to differentiate between four types of heart sound
        """
        super(Process_extrastole, self).__init__()
        self.directory = directory
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier

    def run(self):
        """
        Start the thread for processing
        Iterating through the extrasystole heart sound audio dataset
        """
        for filename in os.listdir(self.directory):
            self.pathName(filename, self.directory)
            time.sleep(1)#stop for 1 second
            
    def pathName(self, filename, directory):
        """
        Parameters
        ----------
        filename : str
            The audio file name in .WAV format
        directory : str
            The directory path to the audio file name
        """
        print("\nStart processing file: "+directory +"/"+ filename)
        filePath = directory +"/"+ filename
        #Initialize the Segmentation class
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, 0)
        #Start segmentation
        segmentation.loadFile(filePath, proceed=True)



if __name__=='__main__':
    #Initialize the csv
    csv_document = CSV()
    column_names, file_name = csv_document.create_csv_file(file='heartSound_data.csv', createFile=True)
    #define the class name for each Threading process
    process_normal_signal = Process_normal("Heart_Sound_Dataset/normal", column_names, file_name, 0)
    process_murmur_signal = Process_murmur("Heart_Sound_Dataset/murmur", column_names, file_name, 1)
    process_extrastole_signal = Process_extrastole("Heart_Sound_Dataset/extrastole", column_names, file_name, 2)
    process_extrahls_signal = Process_extrahls("Heart_Sound_Dataset/extrahls", column_names, file_name, 3)
    #create and start the Threading process
    process_normal_signal.start()
    time.sleep(1)#Stop for 1 second
    process_murmur_signal.start()
    time.sleep(1)#Stop for 1 second
    process_extrastole_signal.start()
    time.sleep(1)#Stop for 1 second
    process_extrahls_signal.start()
    time.sleep(1)#Stop for 1 second
