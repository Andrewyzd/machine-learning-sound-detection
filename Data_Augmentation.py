from audiomentations import Compose, AddImpulseResponse, FrequencyMask, TimeMask, AddGaussianSNR, AddGaussianNoise, TimeStretch, PitchShift, Shift
from audiomentations import Trim, Resample, ClippingDistortion
from audiomentations import AddBackgroundNoise, AddShortNoises, PolarityInversion, Gain
from Segmentation_FeaturesExtract import Segmentation, CSV
from threading import *
import time
import numpy as np
import pandas as pd
import noisereduce as nr
import librosa, librosa.display
import os


class Process_Augmented_extrastole(Thread):
    """
    A class for a thread to process the data augmentation for extrasystole heart sound signal

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
        Pass the audio file path for data augmentation
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
        super(Process_Augmented_extrastole, self).__init__()
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
            time.sleep(1)

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
        #Load the signal and the sampling rate
        signal, sample_rate = segmentation.loadFile(filePath, proceed=False)
        #Remove the background noise
        denoised_signal = segmentation.noiseRemoval(signal, sample_rate, proceed=False)

        #Initialize the thread of the Gaussian SNR
        gaussianSNRThread = GaussianSNRThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        gaussianSNRThread.start()

        time.sleep(1)
        
        #Initialize the thread of the Frequency Mask
        frequencyMaskThread = FrequencyMaskThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        frequencyMaskThread.start()

        time.sleep(1)

        #Initialize the thread of the Gaussian Noise
        gaussianNoiseThread = GaussianNoiseThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        gaussianNoiseThread.start()

        time.sleep(1)

        #Initialize the thread of the Time Mask
        timeMaskThread = TimeMaskThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        timeMaskThread.start()

        time.sleep(1)

        #Initialize the thread of the Time Stretch
        timeStretchThread = TimeStretchThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        timeStretchThread.start()

        time.sleep(1)

        #Initialize the thread of the Pitch Shift
        pitchShiftThread = PitchShiftThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        pitchShiftThread.start()

        time.sleep(1)

        #Initialize the thread of the Shift
        shiftThread = ShiftThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        shiftThread.start()

        time.sleep(1)

        #Initialize the thread of the Trim
        trimThread = TrimThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        trimThread.start()

        time.sleep(1)

        #Initialize the thread of the Resample
        resampleThread = ResampleThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        resampleThread.start()

        time.sleep(1)

        #Initialize the thread of the Clipping distortion
        clippingDistortionThread = ClippingDistortionThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        clippingDistortionThread.start()

        time.sleep(1)

        #Initialize the thread of the Polarity Inversion
        polarityInversionThread = PolarityInversionThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        polarityInversionThread.start()

        time.sleep(1)

        #Initialize the thread of the Gain
        gainThread = GainThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        gainThread.start()

        time.sleep(1)
        
        

class Process_Augmented_extrahls(Thread):
    """
    A class for a thread to process the data augmentation for extra heart sound signal

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
        Pass the audio file path for data augmentation
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
        super(Process_Augmented_extrahls, self).__init__()
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
            time.sleep(1)

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
        #Load the signal and the sampling rate
        signal, sample_rate = segmentation.loadFile(filePath, proceed=False)
        #Remove the background noise
        denoised_signal = segmentation.noiseRemoval(signal, sample_rate, proceed=False)

        gaussianSNRThread = GaussianSNRThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        gaussianSNRThread.start()

        time.sleep(1)

        #Initialize the thread of the Frequency Mask
        frequencyMaskThread = FrequencyMaskThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        frequencyMaskThread.start()

        time.sleep(1)

        #Initialize the thread of the Gaussian Noise
        gaussianNoiseThread = GaussianNoiseThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        gaussianNoiseThread.start()

        time.sleep(1)

        #Initialize the thread of the Time Mask
        timeMaskThread = TimeMaskThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        timeMaskThread.start()

        time.sleep(1)

        #Initialize the thread of the Time Stretch
        timeStretchThread = TimeStretchThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        timeStretchThread.start()

        time.sleep(1)

        #Initialize the thread of the Pitch Shift
        pitchShiftThread = PitchShiftThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        pitchShiftThread.start()

        time.sleep(1)

        #Initialize the thread of the Shift
        shiftThread = ShiftThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        shiftThread.start()

        time.sleep(1)

        #Initialize the thread of the Trim
        trimThread = TrimThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        trimThread.start()

        time.sleep(1)

        #Initialize the thread of the Resample
        resampleThread = ResampleThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        resampleThread.start()

        time.sleep(1)

        #Initialize the thread of the Clipping Distortion
        clippingDistortionThread = ClippingDistortionThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        clippingDistortionThread.start()

        time.sleep(1)

        #Initialize the thread of the Polarity Inversion
        polarityInversionThread = PolarityInversionThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        polarityInversionThread.start()

        time.sleep(1)

        #Initialize the thread of the Gain
        gainThread = GainThread(denoised_signal, sample_rate, self.column_names, self.file_name, self.identifier)
        #Start the thread
        gainThread.start()

        time.sleep(1)
        

class GaussianSNRThread(Thread):
    """
    A class for a thread of Gaussian SNR

    Attributes
    ----------
    denoised_signal : ndarray
        The list of the denoised signal
    sample_rate : int
        The standard sample per rate / sampling rate
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
    """
    def __init__(self, denoised_signal, sample_rate, column_names, file_name, identifier):
        super(GaussianSNRThread, self).__init__()
        self.denoised_signal = denoised_signal
        self.sample_rate = sample_rate
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = 0

    def run(self):
        """
        Start the thread to augment the audio data with Gaussian SNR
        """
        #Initialize the Gaussina SNR class
        augment_snr = Compose([AddGaussianSNR()])
        signal_snr = augment_snr(samples=self.denoised_signal, sample_rate=self.sample_rate)
        #Initalize the segmentation process
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, self.isTrain)
        #Start the segmentation with signal normalization
        segmentation.normalizeSignal(signal_snr, signal_snr, self.sample_rate, proceed=True)
        

class FrequencyMaskThread(Thread):
    """
    A class for a thread of Frequency Mask

    Attributes
    ----------
    denoised_signal : ndarray
        The list of the denoised signal
    sample_rate : int
        The standard sample per rate / sampling rate
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
    """
    def __init__(self, denoised_signal, sample_rate, column_names, file_name, identifier):
        super(FrequencyMaskThread, self).__init__()
        self.denoised_signal = denoised_signal
        self.sample_rate = sample_rate
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = 0

    def run(self):
        #Initialize the Frequency Mask class
        augment_frqMask = Compose([FrequencyMask()])
        signal_frqMask = augment_frqMask(samples=self.denoised_signal, sample_rate=self.sample_rate)
        #Initalize the segmentation process
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, self.isTrain)
        #Start the segmentation with signal normalization
        segmentation.normalizeSignal(signal_frqMask, signal_frqMask, self.sample_rate, proceed=True)

class GaussianNoiseThread(Thread):
    """
    A class for a thread of Gaussian Noise

    Attributes
    ----------
    denoised_signal : ndarray
        The list of the denoised signal
    sample_rate : int
        The standard sample per rate / sampling rate
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
    """
    def __init__(self, denoised_signal, sample_rate, column_names, file_name, identifier):
        super(GaussianNoiseThread, self).__init__()
        self.denoised_signal = denoised_signal
        self.sample_rate = sample_rate
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = 0

    def run(self):
        #Initialize the Gaussina noise class
        augment_gauNoise = Compose([AddGaussianNoise()])
        signal_gauNoise = augment_gauNoise(samples=self.denoised_signal, sample_rate=self.sample_rate)
        #Initalize the segmentation process
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, self.isTrain)
        #Start the segmentation with signal normalization
        segmentation.normalizeSignal(signal_gauNoise, signal_gauNoise, self.sample_rate, proceed=True)

class TimeMaskThread(Thread):
    """
    A class for a thread of Time Mask

    Attributes
    ----------
    denoised_signal : ndarray
        The list of the denoised signal
    sample_rate : int
        The standard sample per rate / sampling rate
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
    """
    def __init__(self, denoised_signal, sample_rate, column_names, file_name, identifier):
        super(TimeMaskThread, self).__init__()
        self.denoised_signal = denoised_signal
        self.sample_rate = sample_rate
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = 0

    def run(self):
        #Initialize the Time mask class
        augment_timeMask = Compose([TimeMask()])
        signal_timeMask = augment_timeMask(samples=self.denoised_signal, sample_rate=self.sample_rate)
        #Initalize the segmentation process
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, self.isTrain)
        #Start the segmentation with signal normalization
        segmentation.normalizeSignal(signal_timeMask, signal_timeMask, self.sample_rate, proceed=True)

class TimeStretchThread(Thread):
    """
    A class for a thread of Time Stretch

    Attributes
    ----------
    denoised_signal : ndarray
        The list of the denoised signal
    sample_rate : int
        The standard sample per rate / sampling rate
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
    """
    def __init__(self, denoised_signal, sample_rate, column_names, file_name, identifier):
        super(TimeStretchThread, self).__init__()
        self.denoised_signal = denoised_signal
        self.sample_rate = sample_rate
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = 0

    def run(self):
        #Initialize the Time stretch class
        augment_timeStretch = Compose([TimeStretch()])
        signal_timeStretch = augment_timeStretch(samples=self.denoised_signal, sample_rate=self.sample_rate)
        #Initalize the segmentation process
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, self.isTrain)
        #Start the segmentation with signal normalization
        segmentation.normalizeSignal(signal_timeStretch, signal_timeStretch, self.sample_rate, proceed=True)

class PitchShiftThread(Thread):
    """
    A class for a thread of Pitch shift

    Attributes
    ----------
    denoised_signal : ndarray
        The list of the denoised signal
    sample_rate : int
        The standard sample per rate / sampling rate
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
    """
    def __init__(self, denoised_signal, sample_rate, column_names, file_name, identifier):
        super(PitchShiftThread, self).__init__()
        self.denoised_signal = denoised_signal
        self.sample_rate = sample_rate
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = 0

    def run(self):
        #Initialize the pitch shift class
        augment_pitchShift = Compose([PitchShift()])
        signal_pitchShift = augment_pitchShift(samples=self.denoised_signal, sample_rate=self.sample_rate)
        #Initalize the segmentation process
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, self.isTrain)
        #Start the segmentation with signal normalization
        segmentation.normalizeSignal(signal_pitchShift, signal_pitchShift, self.sample_rate, proceed=True)

class ShiftThread(Thread):
    """
    A class for a thread of Shift

    Attributes
    ----------
    denoised_signal : ndarray
        The list of the denoised signal
    sample_rate : int
        The standard sample per rate / sampling rate
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
    """
    def __init__(self, denoised_signal, sample_rate, column_names, file_name, identifier):
        super(ShiftThread, self).__init__()
        self.denoised_signal = denoised_signal
        self.sample_rate = sample_rate
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = 0

    def run(self):
        #Initialize the Shift class
        augment_shift = Compose([Shift()])
        signal_shift = augment_shift(samples=self.denoised_signal, sample_rate=self.sample_rate)
        #Initalize the segmentation process
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, self.isTrain)
        #Start the segmentation with signal normalization
        segmentation.normalizeSignal(signal_shift, signal_shift, self.sample_rate, proceed=True)

class TrimThread(Thread):
    """
    A class for a thread of Trim

    Attributes
    ----------
    denoised_signal : ndarray
        The list of the denoised signal
    sample_rate : int
        The standard sample per rate / sampling rate
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
    """
    def __init__(self, denoised_signal, sample_rate, column_names, file_name, identifier):
        super(TrimThread, self).__init__()
        self.denoised_signal = denoised_signal
        self.sample_rate = sample_rate
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = 0

    def run(self):
        #Initialize the Trim class
        augment_trim = Compose([Trim()])
        signal_trim = augment_trim(samples=self.denoised_signal, sample_rate=self.sample_rate)
        #Initalize the segmentation process
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, self.isTrain)
        #Start the segmentation with signal normalization
        segmentation.normalizeSignal(signal_trim, signal_trim, self.sample_rate, proceed=True)

class ResampleThread(Thread):
    """
    A class for a thread of resample

    Attributes
    ----------
    denoised_signal : ndarray
        The list of the denoised signal
    sample_rate : int
        The standard sample per rate / sampling rate
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
    """
    def __init__(self, denoised_signal, sample_rate, column_names, file_name, identifier):
        super(ResampleThread, self).__init__()
        self.denoised_signal = denoised_signal
        self.sample_rate = sample_rate
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = 0

    def run(self):
        #Initialize the Resemple class
        augment_resample = Compose([Resample()])
        signal_resample = augment_resample(samples=self.denoised_signal, sample_rate=self.sample_rate)
        #Initalize the segmentation process
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, self.isTrain)
        #Start the segmentation with signal normalization
        segmentation.normalizeSignal(signal_resample, signal_resample, self.sample_rate, proceed=True)

class ClippingDistortionThread(Thread):
    """
    A class for a thread of Clipping distortion

    Attributes
    ----------
    denoised_signal : ndarray
        The list of the denoised signal
    sample_rate : int
        The standard sample per rate / sampling rate
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
    """
    def __init__(self, denoised_signal, sample_rate, column_names, file_name, identifier):
        super(ClippingDistortionThread, self).__init__()
        self.denoised_signal = denoised_signal
        self.sample_rate = sample_rate
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = 0

    def run(self):
        #Initialize the clipping distortion class
        augment_clippingDistortion = Compose([ClippingDistortion()])
        signal_clippingDistortion = augment_clippingDistortion(samples=self.denoised_signal, sample_rate=self.sample_rate)
        #Initalize the segmentation process
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, self.isTrain)
        #Start the segmentation with signal normalization
        segmentation.normalizeSignal(signal_clippingDistortion, signal_clippingDistortion, self.sample_rate, proceed=True)

class PolarityInversionThread(Thread):
    """
    A class for a thread of Polarity Inversion

    Attributes
    ----------
    denoised_signal : ndarray
        The list of the denoised signal
    sample_rate : int
        The standard sample per rate / sampling rate
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
    """
    def __init__(self, denoised_signal, sample_rate, column_names, file_name, identifier):
        super(PolarityInversionThread, self).__init__()
        self.denoised_signal = denoised_signal
        self.sample_rate = sample_rate
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = 0

    def run(self):
        #Initialize the polarity inversion class
        augment_polarityInversion = Compose([PolarityInversion()])
        signal_polarityInversion = augment_polarityInversion(samples=self.denoised_signal, sample_rate=self.sample_rate)
        #Initalize the segmentation process
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, self.isTrain)
        #Start the segmentation with signal normalization
        segmentation.normalizeSignal(signal_polarityInversion, signal_polarityInversion, self.sample_rate, proceed=True)

class GainThread(Thread):
    """
    A class for a thread of Gain

    Attributes
    ----------
    denoised_signal : ndarray
        The list of the denoised signal
    sample_rate : int
        The standard sample per rate / sampling rate
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
    """
    def __init__(self, denoised_signal, sample_rate, column_names, file_name, identifier):
        super(GainThread, self).__init__()
        self.denoised_signal = denoised_signal
        self.sample_rate = sample_rate
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = 0

    def run(self):
        #Initialize the gain class
        augment_gain = Compose([Gain()])
        signal_gain = augment_gain(samples=self.denoised_signal, sample_rate=self.sample_rate)
        #Initalize the segmentation process
        segmentation = Segmentation(self.column_names, self.file_name, self.identifier, self.isTrain)
        #Start the segmentation with signal normalization
        segmentation.normalizeSignal(signal_gain, signal_gain, self.sample_rate, proceed=True)


if __name__=='__main__':
    #Initialize the CSV class
    csv_document = CSV()
    column_names, file_name = csv_document.create_csv_file(file='heartSound_data.csv',createFile=False)
    #Process the extrasystole heart sound 
    process_extrastole_signal = Process_Augmented_extrastole("Heart_Sound_Dataset/extrastole", column_names, file_name, 2)
    #Process the extra heart sound
    process_extrahls_signal = Process_Augmented_extrahls("Heart_Sound_Dataset/extrahls", column_names, file_name, 3)
    #Start the thread for extrasystole
    process_extrastole_signal.start()
    #Stop for 1 second
    time.sleep(1)
    #Start the thread for extra heart sound
    process_extrahls_signal.start()
    #Stop for 1 second
    time.sleep(1)
