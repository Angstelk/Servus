#!/usr/bin/env python
# coding: utf-8

#import for mfcc
import scipy
from scipy import fftpack
from scipy import signal
from scipy.io import wavfile as wav



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
saved_model_path = "/home/apeiron/Servus/modelCNN"
wav_file_path = "/home/apeiron/Servus/rec.wav"
#silence treshold
TRESHOLD = 15
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
#in seconds
REC_LENGTH = 1
S_WIDTH = 2
NORMALIZATION = (1.0/32768.0)
PADDING = 68
NUM_MFCC = 40
class_label = ["down","go","left","on","right","stop","up"]

###################################################################

def add_eps(x):
    x[scipy.where(x == 0)] = scipy.finfo(dtype=x.dtype).eps
    return x


def preemphasis(seq, coeff):
    return scipy.append(seq[0], seq[1:] - coeff * seq[:-1])


# http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
def freq_to_mel(freq):
    return 1125.0 * scipy.log(1.0 + freq / 700.0)


def mel_to_freq(mel):
    return 700.0 * (scipy.exp(mel / 1125.0) - 1.0)


def iter_bin(out, curr_bin, next_bins, backward=False):
    next_bin = next_bins[scipy.where(next_bins > curr_bin)][0]
    if backward:
        sign = -1
        bias = next_bin
    else:
        sign = 1
        bias = curr_bin
    for f in range(int(curr_bin), int(next_bin)):
        out[f] = sign * (f - bias) / (next_bin - curr_bin)


def mel_filterbank(num_bank, num_freq, sample_freq, low_freq, high_freq):
    num_fft = (num_freq - 1) * 2
    low_mel = freq_to_mel(low_freq)
    high_mel = freq_to_mel(high_freq)
    banks = scipy.linspace(low_mel, high_mel, num_bank + 2)
    bins = scipy.floor((num_fft + 1) * mel_to_freq(banks) / sample_freq)
    out = scipy.zeros((num_bank, num_fft // 2 + 1))
    for b in range(num_bank):
        iter_bin(out[b], bins[b], bins[b+1:])
        iter_bin(out[b], bins[b+1], bins[b+2:], backward=True)
    return out

def MFCC_spectro(data):
    
    # config is based on Kaldi compute-mfcc-feats

    # STFT conf
    frame_length = 25  # frame / msec
    frame_shift = 10   # frame / msec
    remove_dc_offset = True
    window_type = "hamming"

    # Fbank conf
    preemphasis_coeff = 0.97
    use_power = True  # else use magnitude
    high_freq = 0.0  # offset from Nyquist freq [Hz]
    low_freq = 20.0  # offset from 0 [Hz]
    num_mel_bins = 80  # (default 23)
    num_ceps = 40
    num_lifter = 22

    sample_freq, raw_seq = wav.read(data)
    

    assert raw_seq.ndim == 1  # assume mono
    seq = raw_seq.astype(scipy.float64)
    if remove_dc_offset:
        seq -= scipy.mean(seq)

    # STFT feat
    seq = preemphasis(seq, preemphasis_coeff)
    num_samples = sample_freq // 1000
    window = signal.get_window(window_type, frame_length * num_samples)
    mode = "psd" if use_power else "magnitude"
    f, t, spectrogram = signal.spectrogram(seq, sample_freq, window=window, noverlap=frame_shift*num_samples, mode=mode)

    # log-fbank feat
    banks = mel_filterbank(num_mel_bins, spectrogram.shape[0], sample_freq, low_freq, sample_freq // 2 - high_freq)
    fbank_spect = scipy.dot(banks, spectrogram)
    logfbank_spect = scipy.log(add_eps(fbank_spect))

    # mfcc feat
    dct_feat = fftpack.dct(logfbank_spect, type=2, axis=0, norm="ortho")[:num_ceps]
    lifter = 1 + num_lifter / 2.0 * scipy.sin(scipy.pi * scipy.arange(num_ceps) / num_lifter)
    mfcc_feat = lifter[:, scipy.newaxis] * dct_feat
    mfcc_feat = np.asarray(mfcc_feat, dtype=np.float32)

    pad = PADDING - mfcc_feat.shape[1]
    print(pad)
    print(PADDING)
    if pad<0 :
        return None
    mfcc = np.pad(mfcc_feat, pad_width=((0,0), (0,pad)), mode="constant")
    #scaled =np.mean(mfcc_feat.T, axis=0)
    #plt.matshow(mfcc_feat)
    #plt.savefig("mfcc.png")
    #plt.show()
    return mfcc
###################################################################

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
    spectro = MFCC_spectro(file_name)
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




