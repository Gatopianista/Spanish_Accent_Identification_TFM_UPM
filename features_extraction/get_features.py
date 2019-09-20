'''
@author: andreafontalvo

Spanish accent identification
Master on Automation and Robotics
Final Project 
UPM - Universidad Politécnica de Madrid

Features extraction
This script allows the user to import sets of audio and extract
the desired set of features on study based on the audio processing 
class written. It outputs the dataset as a tensor which are then saved.  

Three cases of datasets were considered under study on this master final project.
'''
# %% 
from audio_processing import audio_processing
import time
import librosa
import numpy as np
from glob import glob
import speechpy

path = '/Users/andreafontalvo/Dropbox/MUAR UPM/TFM/audio/'
audio = audio_processing()
print(' * Listo para empezar *') # ready to begin

#%% Archivos español peninsular - peninsular spanish - es_ES
print('Cargar dataset es-es') #load audio filess
print('-------------------------------')
files_es,numberfiles_es = audio.files(path+'/audio/es-es')
print('Done.')

#%% Archivos español argentino - argentinian spanish - es_AR
print('Cargar dataset es-ar') #load audio files
print('-------------------------------')
files_ar,numberfiles_ar = audio.files(path+'/audio/es-ar')
print('Done.')

#%% Extraction of the first 13 MFCC coefficients
print('EXTRACCIÓN DE 13 PRIMEROS COEFICIENTES MFCC')
print('-------------------------------')
num_mfccs = 13
features_es_speechpy_13 = audio.extract_features_speechpy(files_es,num_mfccs,0)
print(features_es_speechpy_13.shape)

#%% First case - 13 features
# CASO 13 CARACTERÍSTICAS
print('-------------------------------')
features_ar_speechpy_13 = audio.extract_features_speechpy(files_ar,num_mfccs,2)
print(features_ar_speechpy_13.shape)

features_ar_13,features_es_13 = audio.normalize(features_ar_speechpy_13,features_es_speechpy_13)

np.save('data/features_ar_13', features_ar_13)
np.save('data/features_es_13', features_es_13)

#%% Second case - 14 features
# CASO 14 CARACTERÍSTICAS
print('-------------------------------')
features_ar_speechpy_14 = audio.extract_features_speechpy(files_ar,num_mfccs,1)
print(features_ar_speechpy_14.shape)

features_ar_14,features_es_14 = audio.normalize(features_ar_speechpy_14,features_es_speechpy_14)

np.save('data/features_ar_14', features_ar_14)
np.save('data/features_es_14', features_es_14)

#%% third case - 39 features
# CASO 39 CARACTERÍSTICAS
print('-------------------------------')
features_ar_speechpy_39 = audio.extract_features_speechpy(files_ar,num_mfccs,0)
print(features_ar_speechpy_39.shape)

features_ar_39,features_es_39 = audio.normalize(features_ar_speechpy_39,features_es_speechpy_39)

np.save('data/features_ar_39', features_ar_39)
np.save('data/features_es_39', features_es_39)