from pyAudioAnalysis import ShortTermFeatures as stf
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import noisereduce as nr
import time
import math
import csv
import os

class Segmentation():
    """
    A class for segmentation

    Attributes
    ----------
    column_names : list
        The list for storing the column names
    file_name : str
        The file name of the heart sound dataset in CSV format
    identifier : int
        The identifier to classify different type of heart sounds
    isTrain : int
        The identifier to differenciate the data is passed for training or not

    Methods
    -------
    loadFile(filePath, proceed)
        Load the signal and the sample per rate from the audio file
    noiseRemoval(signal, sample_rate, proceed)
        Remove the background noise of the signal
    normalizeSignal(reduced_signal_noise, clean_signal, sample_rate, proceed)
        Normalize the denoised signal
    shannonEnergy(normalized_signal, sample_rate, clean_signal, proceed)
        Calculate the Shannon energy using the normalized signal
    average_Shannon_Energy(shannon_energy_signal, sample_rate, clean_signal, proceed)
        Averaging the Shannon energy 
    shannon_Envelope(segment_energy_signal, clean_signal, sample_rate, proceed)
        Calculate the Shannon envelope
    extractSignalBasedOnThreshold(shannon_envelope, mean_Shannon_Energy, std_Shannon_Energy, clean_signal, sample_rate, proceed)
        Extract the signal based on the threshold value 
    """
    def __init__(self, column_names, file_name, identifier, isTrain):
        """
        Parameters
        ----------
        column_names : list
            The list for storing the column names
        file_name : str
            The file name of the heart sound dataset in CSV format
        identifier : int
            The identifier to classify different type of heart sounds
        isTrain : int
            The identifier to differenciate the data is passed for training or not
        """
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = isTrain
        self.file = 0
        
    def loadFile(self, filePath, proceed):
        """
        Load the signal and sampling rate of the audio file
        
        Parameters
        ----------
        filePath : str
            The directory path of the selected audio file
        proceed : boolean
            True to process to next stage of segmentation, otherwise stop

        Returns
        -------
        signal : ndarray
            The list of signal from the audio file
        sample_rate : int
            The standard sample per rate / sampling rate which is 44,100 hertz
        """
        self.file = filePath
        # load audio file with Librosa
        signal, sample_rate = librosa.load(filePath, sr=44100)
        if proceed:
            #pass the loaded signal for noise removal
            self.noiseRemoval(signal, sample_rate, proceed=True)

        return signal, sample_rate

    def noiseRemoval(self, signal, sample_rate, proceed):
        """
        Remove the background noise of the heart sound

        Parameters
        ----------
        signal : ndarray
            The list of signal either from the load audio file or ECG sensor
        sample_rate : int
            The standard sample per rate / sampling rate which is 44,100 hertz
        proceed : boolean
            True to process to next stage of segmentation, otherwise stop

        Returns
        -------
        reduced_signal_noise : ndarray
            The list of denoised signal 
        """
        #remove the noise of the signal
        reduced_noise = nr.reduce_noise(audio_clip=signal, noise_clip=signal, verbose=False)
        #change to ndarray
        reduced_signal_noise = np.array(reduced_noise)
        np.set_printoptions(threshold=np.inf)
        #Proceeds to normalize the signal
        if proceed:
            self.normalizeSignal(reduced_signal_noise, reduced_signal_noise, sample_rate, proceed=True)

        return reduced_signal_noise

    def normalizeSignal(self, reduced_signal_noise, clean_signal, sample_rate, proceed):
        """
        Normalized the signal of denoised function

        Parameters
        ----------
        reduced_signal_noise : ndarray
            The list of denoised signal
        clean_signal : ndarray
            The list of denoised signal
        sample_rate : int
            The standard sample per rate / sampling rate which is 44,100 hertz
        proceed : boolean
            True to process to next stage of segmentation, otherwise stop

        Returns
        -------
        normalized_signal : ndarray
            The list of normalized signal
        """
        #extract the maximum signal
        maximum_signal = max(np.abs(reduced_signal_noise))
        #normalize the signal
        normalized_signal = np.array([(abs(signal) / maximum_signal) for signal in reduced_signal_noise])
        #Proceeds to Shannon energy computation
        if proceed:
            self.shannonEnergy(normalized_signal, sample_rate, clean_signal, proceed=True)

        return normalized_signal

    def shannonEnergy(self, normalized_signal, sample_rate, clean_signal, proceed):
        """
        Calculate the Shannon energy

        Parameters
        ----------
        normalized_signal : ndarray
            The list of normalized signal
        sample_rate : int
            The standard sample per rate / sampling rate which is 44,100 hertz
        clean_signal : ndarray
            The list of denoised signal
        proceed : boolean
            True to process to next stage of segmentation, otherwise stop

        Return
        ------
        normalized_signal : ndarray
            The list of Shannon energy of each signal
        """
        for x in range(0, len(normalized_signal)): #iterate through the normalized signal
            signal_sample = abs(normalized_signal[x]) ** 2 #power the signal by 2
            if signal_sample <= 0: #set the signal to 1 if it is empty
                signal_sample = 1.0
                
            shannon_energy = signal_sample * math.log(signal_sample) #calculate Shannon energy
            normalized_signal[x] = shannon_energy#replace the normalized signal with Shannon energy
        #proceeds to average Shannon energy 
        if proceed:
            self.average_Shannon_Energy(normalized_signal, False, sample_rate, clean_signal, proceed=True)

        return normalized_signal

    def average_Shannon_Energy(self, shannon_energy_signal, realtime, sample_rate, clean_signal, proceed):
        """
        Calculate the average Shannon energy

        Parameters
        ----------
        shannon_energy_signal : ndarray
            The list of Shannon energy
        sample_rate : int
            The standard sample per rate / sampling rate which is 44,100 hertz
        clean_signal : ndarray
            The list of denoised signal
        proceed : boolean
            True to process to next stage of segmentation, otherwise stop

        Returns
        -------
        segment_energy_signal : ndarray
            The list of average Shannon energy
        """
        length_of_signal = len(shannon_energy_signal)#obtain the length of signal
        segment_signal = 0 #Initialize the signal
        #Set the segmented signal to 0.0002 seconds for realtime analysis, otherwise 0.02 seconds for audio recorder
        if realtime:
            segment_signal = int(sample_rate * 0.0002)#set the segment of 0.0002 seconds
        else:
            segment_signal = int(sample_rate * 0.02)#set the segment of 0.02 seconds
        segment_energy = [] #initialize the array
        for x in range(0, len(shannon_energy_signal), segment_signal):
            sum_signal = 0
            current_segment_energy = shannon_energy_signal[x:x+segment_signal] #retrieve the signal in a segment of 0.02 seconds
            for i in range(0, len(current_segment_energy)):
                sum_signal += current_segment_energy[i]#sum up the Shannon energy

            segment_energy.append(-(sum_signal/segment_signal))#assign the average Shannon energy to array

        segment_energy_signal = np.array(segment_energy)#convert to numpy array
        
        #Proceeds to Shannon envelope
        if proceed:
            self.shannon_Envelope(segment_energy_signal, clean_signal, sample_rate, proceed=True)

        return segment_energy_signal

    def shannon_Envelope(self, segment_energy_signal, clean_signal, sample_rate, proceed):
        """
        Compute the Shannon envelope

        Parameters
        ----------
        segment_energy_signal : ndarray
            The list of average Shannon energy
        clean_signal : ndarray
            The list of denoised signal
        sample_rate : int
            The standard sample per rate / sampling rate which is 44,100 hertz
        proceed : boolean
            True to process to next stage of segmentation, otherwise stop

        Returns
        -------
        shannon_envelope : ndarray
            The list of Shannon envelope
        mean_Shannon_Energy : ndarray
            The mean value of average Shannon energy
        std_Shannon_Energy : ndarray
            The standard deviation of average Shannon energy
        envelope_time : ndarray
            The duration of the envelope
        """
        #calculate mean
        mean_Shannon_Energy = np.mean(segment_energy_signal)
        #calculate standard deviation
        std_Shannon_Energy = np.std(segment_energy_signal)
        #calculate Shannon Envelope
        for x in range(0, len(segment_energy_signal)):
            envelope = 0
            envelope = (segment_energy_signal[x] - mean_Shannon_Energy) / std_Shannon_Energy
            segment_energy_signal[x] = envelope

        shannon_envelope = segment_energy_signal
        #calculate envelope size
        envelope_size = range(0, shannon_envelope.size)
        #calculate envelope time
        envelope_time = librosa.frames_to_time(envelope_size, hop_length=442)
        #Proceeds to extract the signal based on threshold value
        if proceed:
            self.extractSignalBasedOnThreshold(shannon_envelope, mean_Shannon_Energy, std_Shannon_Energy, clean_signal, sample_rate, proceed=True)

        return shannon_envelope, mean_Shannon_Energy, std_Shannon_Energy, envelope_time
    
    def extractSignalBasedOnThreshold(self, shannon_envelope, mean_Shannon_Energy, std_Shannon_Energy, clean_signal, sample_rate, proceed):
        """
        Extract the signal based on threshold value

        Parameters
        ----------
        shannon_envelope : ndarray
            The list of Shannon envelope
        mean_Shannon_Energy : ndarray
            The mean value of average Shannon energy
        std_Shannon_Energy : ndarray
            The standard deviation of average Shannon energy
        clean_signal : ndarray
            The list of denoised signal
        sample_rate : int
            The standard sample per rate / sampling rate which is 44,100 hertz
        proceed : boolean
            True to process to next stage of segmentation, otherwise stop

        Returns
        -------
        segmented_signal : ndarray
            The list of segmented signal value
        clean_segmented_signal : ndarray
            The list of cleaned segmented signal value
        threshold : float
            The threshold value
        """
        segment_signal = [0] * len(clean_signal)
        threshold = 0
        k = 0.001
        #calculate threshold
        if std_Shannon_Energy < mean_Shannon_Energy:
            threshold = abs(k * mean_Shannon_Energy * (1 - std_Shannon_Energy ** 2))

        elif std_Shannon_Energy > mean_Shannon_Energy:
            threshold = abs(k * std_Shannon_Energy * (1 -  mean_Shannon_Energy ** 2))

        #extract the signal that is greater than threshold   
        for x in range(0, len(clean_signal)):
            if np.abs(clean_signal[x]) > threshold:
                segment_signal[x] = clean_signal[x]

        segmented_signal = np.array(segment_signal)

        #remove 0 
        clean_segmented_signal = np.delete(segmented_signal, np.where(segmented_signal == 0))

        if proceed:
            # define object for feature class
            features = FeaturesExtraction(self.column_names, self.file_name, self.identifier, self.isTrain)
            #perform features extraction
            features.extractFeatures(clean_segmented_signal, sample_rate, threshold)

        return segmented_signal, clean_segmented_signal, threshold

    
class FeaturesExtraction():
    """
    A class for feature extraction on the heart sound signal

    ...

    Attributes
    ----------
    column_names : list
        The list for storing the column names
    file_name : str
        The file name of the heart sound dataset in CSV format
    identifier : int
        The identifier to classify different type of heart sounds
    isTrain : int
        The identifier to differenciate the data is passed for training or not

    Methods
    -------
    extractFeatures(clean_segmented_signal, sample_rate, threshold)
        Extracts and saves the feature value
    zero_crossing_rate(clean_segmented_signal)
        Extracts the zero-crossing rate
    mel_frequency_cepstral_coefficients(clean_segmented_signal, sample_rate)
        Extract the mel-frequency cepstral coefficients
    spectral_centroid(clean_segmented_signal, sample_rate)
        Extract the spectral centroid
    spectral_rolloff(clean_segmented_signal, sample_rate)
        Extract the spectral rolloff
    spectral_flux(clean_segmented_signal)
        Extract the spectral flux
    frequency_domain(clean_segmented_signal)
        Extract the frequency domain
    energy_entropy(clean_segmented_signal)
        Extract the energy entropy
    """
    def __init__(self, column_names, file_name, identifier, isTrain):
        """
        Parameters
        ----------
        column_names : list
            The list for storing the column names
        file_name : str
            The file name of the heart sound dataset in CSV format
        identifier : int
            The identifier to classify different type of heart sounds
        isTrain : int
            The identifier to differenciate the data is passed for training or not
        """
        self.column_names = column_names
        self.file_name = file_name
        self.identifier = identifier
        self.isTrain = isTrain
        
    def extractFeatures(self, clean_segmented_signal, sample_rate, threshold):
        """
        Extract and save the features to the csv file

        Parameters
        ----------
        clean_segmented_signal : ndarray
            The signal with lub and dub sound
        sample_rate : int
            The standard sample per rate / sampling rate which is 44,100 hertz
        threshold : float
            The threshold value

        Returns
        -------

        features_data : list
            The list of all the extracted feature values
        """
        features_data = []
        features_data.append(threshold)
        #extract zero-crossing rate
        zero_crossing_rate = self.zero_crossing_rate(clean_segmented_signal)
        features_data.append(zero_crossing_rate)
        #extract mel_frequency cepstral coefficients
        mean_mfcc, std_mfcc = self.mel_frequency_cepstral_coefficients(clean_segmented_signal, sample_rate)
        features_data.append(mean_mfcc)
        features_data.append(std_mfcc)
        #spectral centroid
        spectral_centroid = self.spectral_centroid(clean_segmented_signal, sample_rate)
        features_data.append(spectral_centroid)
        #spectral rolloff
        spectral_rolloff = self.spectral_rolloff(clean_segmented_signal, sample_rate)
        features_data.append(spectral_rolloff)
        #spectral flux
        spectral_flux = self.spectral_flux(clean_segmented_signal)
        features_data.append(spectral_flux)
        #frequency domain
        mean_frequency_domain_real, mean_frequency_domain_imaginary, std_frequency_domain = self.frequency_domain(clean_segmented_signal)
        features_data.append(mean_frequency_domain_real)
        features_data.append(std_frequency_domain)
        #energy entropy
        energy_entropy = self.energy_entropy(clean_segmented_signal)
        features_data.append(energy_entropy)

        #Create and defined CSV class
        if (self.isTrain == 0):
            #add identifier / label for the features
            features_data.append(self.identifier)
            #Initialize the CSV object
            write_csv_data = CSV()
            #Save the data to CSV
            write_csv_data.write_data_to_csv(self.column_names, self.file_name, features_data)

        return features_data
        
    def zero_crossing_rate(self, clean_segmented_signal):
        """
        Extract the zero-crossing rate of the signal

        Parameter
        ---------
        clean_segmented_signal : ndarray
            The signal with lub and dub sound

        Return
        ------
        zero_crossing_rate : str
            The value of zero-crossing rate
        """
        zero_crossing_rate = stf.zero_crossing_rate(clean_segmented_signal)
        return str(zero_crossing_rate)

    def mel_frequency_cepstral_coefficients(self, clean_segmented_signal, sample_rate):
        """
        Extract the mel-frequency cepstral coefficients (MFCCs) of the signal

        Parameter
        ---------
        clean_segmented_signal : ndarray
            The signal with lub and dub sound
        sample_rate : int
            The standard sample per rate / sampling rate which is 44,100 hertz

        Return
        ------
        mean_mfcc : float
            The mean of MFCCs
        std_mfcc : float
            The standard deviation of MFCCs
        """
        mfcc = librosa.feature.mfcc(clean_segmented_signal.astype('float32'), sr=sample_rate)
        mean_mfcc = np.mean(mfcc)
        std_mfcc = np.std(mfcc)
        return mean_mfcc, std_mfcc

    def spectral_centroid(self, clean_segmented_signal, sample_rate):
        """
        Extract the spectral centroid of the signal

        Parameter
        ---------
        clean_segmented_signal : ndarray
            The signal with lub and dub sound
        sample_rate : int
            The standard sample per rate / sampling rate which is 44,100 hertz

        Return
        ------
        spectral_centroid[0][0] : str
            The value of spectral centroid
        """
        spectral_centroid = librosa.feature.spectral_centroid(clean_segmented_signal, sr=sample_rate)
        return str(spectral_centroid[0][0])

    def spectral_rolloff(self, clean_segmented_signal, sample_rate):
        """
        Extract the spectral rolloff of the signal

        Parameter
        ---------
        clean_segmented_signal : ndarray
            The signal with lub and dub sound
        sample_rate : int
            The standard sample per rate / sampling rate which is 44,100 hertz

        Return
        ------
        spectral_rolloff[0][0] : str
            The value of spectral rolloff
        """
        spectral_rolloff = librosa.feature.spectral_rolloff(clean_segmented_signal, sr=sample_rate)
        return str(spectral_rolloff[0][0])

    def spectral_flux(self, clean_segmented_signal):
        """
        Extract the spectral flux of the signal

        Parameter
        ---------
        clean_segmented_signal : ndarray
            The signal with lub and dub sound

        Return
        ------
        spectral_flux : str
            The value of spectral flux
        """ 
        #divide the segmented signal length by half
        fft_frame_length = len(clean_segmented_signal) / 2
        #extract the signal by half
        first_frame = clean_segmented_signal[:int(fft_frame_length)]
        second_frame = clean_segmented_signal[int(fft_frame_length):]

        frame_step = 1
        while(first_frame.shape != second_frame.shape):
            first_frame = clean_segmented_signal[:frame_step+int(fft_frame_length)]
            second_frame = clean_segmented_signal[int(fft_frame_length):]
            frame_step = frame_step + 1

        #calculate the fft of the signal
        fft_first_frame = np.array([np.fft.fft(first_frame)])
        fft_second_frame = np.array([np.fft.fft(second_frame)])

        #extract the spectral flux features
        spectral_flux = np.array(stf.spectral_flux(np.abs(fft_first_frame), np.abs(fft_second_frame)))
        return str(spectral_flux)

    def frequency_domain(self, clean_segmented_signal):
        """
        Extract the frequency domain of the signal

        Parameter
        ---------
        clean_segmented_signal : ndarray
            The signal with lub and dub sound

        Return
        ------
        mean_frequency_domain_real : float
            The mean of frequency domain using real number
        mean_frequency_domain_imaginary : float
            The mean of frequency domain using imaginary number
        std_frequency_domain : str
            The standard deviation of the frequency domain
        """
        frequency_domain = np.array([np.fft.fft(clean_segmented_signal)])
        #calculate mean
        mean_frequency_domain = np.mean(frequency_domain)
        #calculate standard deviation
        std_frequency_domain = np.std(frequency_domain)
        #extract the real and the imaginary number from complex number
        mean_frequency_domain_real = mean_frequency_domain.real
        mean_frequency_domain_imaginary = mean_frequency_domain.imag
        
        return str(mean_frequency_domain_real), str(mean_frequency_domain_imaginary), str(std_frequency_domain)

    def energy_entropy(self, clean_segmented_signal):
        """
        Extract the energy entropy of the signal

        Parameter
        ---------
        clean_segmented_signal : ndarray
            The signal with lub and dub sound

        Return
        ------
        energy_entropy : str
            The value of energy entropy
        """
        #Extract the energy entropy
        energy_entropy = np.array(stf.energy_entropy(clean_segmented_signal))
        return str(energy_entropy)

class CSV():
    """
    A class for creating the CSV file

    Methods
    ----------
    create_csv_file(file, createFile)
        Create the csv file
    write_data_to_csv(column_names, file_name, features_data)
        Save the data to csv file
    """
    def create_csv_file(self, file, createFile):
        """
        Create the csv file

        Parameters
        ----------
        file : str
            The name of the .csv file
        createFile : boolean
            True to create a file, otherwise don't

        Returns
        -------
        column_names : list
            The list of the column name
        file_name : str
            The name of the .csv file
        """
        #define the column name
        column_names = ['threshold',
                        'zero-crossing_rate',
                        'mean_mfcc', 'std_mfcc',
                        'spectral_centroid',
                        'spectral_rolloff',
                        'spectral_flux',
                        'mean_frequency_domain_real', 'std_frequency_domain',
                        'energy_entropy',
                        'sound_type']

        #define the file name
        file_name = file

        if createFile:
            #create the csv file
            with open(file_name, 'w', newline='') as f:
                #Assign the header
                writer = csv.DictWriter(f, fieldnames=column_names)
                #Write the header to the file
                writer.writeheader()

        return column_names, file_name

    def write_data_to_csv(self, column_names, file_name, features_data):
        """
        Save the data to csv file by row

        Parameters
        ----------
        column_names : list
            The list of the column name
        file_name : str
            The name of the .csv file
        features_data : list
            The list of the feature data
        """
        #Structure the data and the column as the dataframe
        data = pd.DataFrame([features_data], columns=[column_names])
        #Save the data to the .csv file
        data.to_csv(file_name, mode='a', header=False, index=False)
        
