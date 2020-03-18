#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model
from tensorflow import keras
import pyaudio
import math
import struct
import wave
import time
import os
import librosa
import numpy as np


# In[2]:


#SETTINGS
saved_model_path = "C:\\Users\\Filip\\Desktop\\Jupyter\\modelCNN"
wav_file_path = "C:\\Users\\Filip\\Desktop\\Jupyter\\rec.wav"
#silence treshold
TRESHOLD = 12
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
#in seconds
REC_LENGTH = 1
S_WIDTH = 2
NORMALIZATION = (1.0/32768.0)
PADDING = 64
NUM_MFCC = 40
class_label = ["down","go","left","on","right","stop","up"]


# In[3]:


model = keras.models.load_model(saved_model_path)
#record file init
pyaud = pyaudio.PyAudio()
stream = pyaud.open(format=AUDIO_FORMAT, 
                    channels=NUM_CHANNELS, 
                    rate=SAMPLE_RATE, 
                    input=True, output=True, 
                    frames_per_buffer=CHUNK_SIZE)


# In[4]:


#get rms
def rms(frame):
        unpack_format = "%dh" % (len(frame)/S_WIDTH)
        unpacked = struct.unpack(unpack_format, frame)
        square_sum = 0
        
        for sample in unpacked:
            n = sample * NORMALIZATION
            square_sum = square_sum + n*n
            
        rms = math.sqrt(square_sum / (len(frame)/S_WIDTH))
        
        return rms*1000


# In[5]:


#save file as .wav
def save(recording):
        file = wave.open(wav_file_path,"wb")
        file.setnchannels(NUM_CHANNELS)
        file.setsampwidth(pyaud.get_sample_size(AUDIO_FORMAT))
        file.setframerate(SAMPLE_RATE)
        file.writeframes(recording)
        file.close()


# In[6]:


#make spectrogram out of .wav file
def get_spectrogram(file_name):
        try:
            audio, sample_rate = librosa.load(file_name,
                                              res_type="kaiser_fast")
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate,
                                        n_mfcc = NUM_MFCC)
            pad = PADDING - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0), (0,pad)), mode="constant")
        except:
            print("Error with file: ", file_name)
            return None, None
        return mfcc


# In[7]:


def make_prediction(file_name):
    spectro = get_spectrogram(file_name)
    spectro = spectro.reshape(1, NUM_MFCC, PADDING, NUM_CHANNELS)
    prediction = model.predict_classes(spectro)
    print("Predicted class: {}".format(class_label[int(prediction)]))     


# In[8]:


def record():
    print("rec start")
    recording = []
    start_time = time.time()
    finish_time = start_time + REC_LENGTH
        
    while start_time <= finish_time:
        frame = stream.read(CHUNK_SIZE)
        start_time = time.time()
        recording.append(frame)
    print("rec stop")
    save(b''.join(recording))
    make_prediction(wav_file_path)


# In[9]:


def loop():
    print("starting loop")
    while True:
        input = stream.read(CHUNK_SIZE)
        if rms(input) > TRESHOLD:
            record()


# In[10]:


loop()


# In[ ]:




