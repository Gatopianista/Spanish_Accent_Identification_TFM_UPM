#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: andreafontalvo

Spanish accent identification
Master on Automation and Robotics
Final Project 
UPM - Universidad Politécnica de Madrid

Audio processing class
This python class was written and used on this master final project
to record, play, save, extract mfcc features and normalize audio files.
It was used along many scripts during this project as tool according 
to the needs. 
"""  

import wave  
import librosa
from glob import glob
import numpy as np
import speechpy
import pyaudio

class audio_processing():
    def __init__(self):
        #define stream chunk   
        self.chunk = 1024  
        self.format = pyaudio.paInt16
        self.channels = 2
        self.rate = 44100
        self.rec_seconds = 3
        self.audio = pyaudio.PyAudio()

    def record(self,filename):
        # start Recording
        stream = self.audio.open(format=self.format, channels=self.channels,
                        rate=self.rate, input=True,
                        frames_per_buffer=self.chunk)
        print ('GRABANDO...')
        frames = []
        
        for i in range(0, int(self.rate / self.chunk * self.rec_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)
        print ('GRABACIÓN TERMINADA')
        
        # stop Recording
        stream.stop_stream()
        stream.close()
        
        waveFile = wave.open(filename, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.rate)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

    def play(self,filename):
        f = wave.open(filename)
        #open stream  
        stream = self.audio.open(format = self.audio.get_format_from_width(f.getsampwidth()),  
                        channels = self.channels,  
                        rate = self.rate,
                        output = True)  
        #read data  
        data = f.readframes(self.chunk)  
        #play stream
        print('REPRODUCIENDO...')
        while data:  
            stream.write(data)  
            data = f.readframes(self.chunk)  
        #stop stream  
        stream.stop_stream()  
        stream.close()
        #close PyAudio  
        # self.audio.terminate()  

    def files(self,data_dir):
        # data_dir = 'audio/'+dataset_name
        audio_files = glob(data_dir + '/*.wav')
        number_of_files = len(audio_files)
        print(f'Número de archivos: {number_of_files}')
        return audio_files,number_of_files

    def mfcc_extraction(self,filename):
        y, sr = librosa.load(filename)
        y_len = len(y)
        mfccs = librosa.feature.mfcc(y, sr, n_mfcc=19, hop_length=int(0.010*sr), n_fft=int(0.025*sr))
        return mfccs,y_len

    def extract_features_librosa(self,files_vector,n_mfccs):
        mfccs = []
        size = 41396
        bands = n_mfccs
        for i in range(len(files_vector)):
            sound_clip, sr = librosa.load(files_vector[i])
            signal = sound_clip[:size]
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc = bands,hop_length=int(0.010*sr), n_fft=int(0.025*sr))
            mfccs.append(mfcc)
        features = np.asarray(mfccs)
        return features

    def extract_features_speechpy(self,files_vector,n_mfccs,formato):
        features = []
        for i in range(len(files_vector)):
            signal, sr = librosa.load(files_vector[i])
            diff = int(len(signal)/2)
            signal=signal[diff-20000:diff+20000]
            mfcc_speechpy = speechpy.feature.mfcc(signal,sr,\
                                            frame_length = 0.025,\
                                            frame_stride = 0.010,\
                                            num_cepstral = n_mfccs)
            #---------------------------------------derivadas--------------------------------
            if(formato == 0):
                derivatives_speechpy = speechpy.feature.extract_derivative_feature(mfcc_speechpy)
                mfccs = derivatives_speechpy[:,:,0]
                derv1  = derivatives_speechpy[:,:,1]
                derv2  = derivatives_speechpy[:,:,2]
                features_per_signal = np.concatenate((mfccs,derv1,derv2),axis=1)
                features.append(features_per_signal)
            # --------------------------------------energia----------------------------------
            if(formato == 1):
                energy = speechpy.feature.mfe(signal, sr, frame_length=0.020, frame_stride=0.01,\
                            num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
                enen = np.array(energy[1])
                feat = np.zeros((mfcc_speechpy.shape[0],14))
                feat[:,:13] = mfcc_speechpy
                feat[:,13] = enen 
                features.append(feat)
            # -------------------------------------- normal------------------------------------
            if(formato == 2):
                features.append(mfcc_speechpy)
        features = np.asarray(features)
        return features

    def normalize(self,features_ar, features_es):
        features = features_ar[0]
        for i in range(features_ar.shape[0]-1):
            features = np.concatenate((features,features_ar[i+1]),axis=0)
        for i in range(features_es.shape[0]):
            features = np.concatenate((features,features_es[i]),axis=0)
        
        features_cmvn = speechpy.processing.cmvn(features)

        features_ar_cmvn = []
        features_es_cmvn = []
        ind1 = 0
        ind2 = 0
        for j in range(len(features_ar)):
            ind1 = ind2
            ind2 = ind2 + 179
            features_ar_cmvn.append(features_cmvn[ind1:ind2,:])
        for j in range(len(features_es)):
            ind1 = ind2
            ind2 = ind2 + 179
            features_es_cmvn.append(features_cmvn[ind1:ind2,:])

        features_ar_cmvn = np.array(features_ar_cmvn)
        features_es_cmvn = np.array(features_es_cmvn)

        return features_ar_cmvn,features_es_cmvn