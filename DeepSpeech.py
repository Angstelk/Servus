#!/usr/bin/env python
# coding: utf-8
#spectrogram imports
import scipy
from scipy import fftpack
from scipy import signal


#imports
import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
from scipy.io import wavfile as wav
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics 
#keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Nadam
from keras.utils import np_utils



#Settings
audio_dataset_path = "/home/apeiron/Servus/dataset_wave"

relative_dir = os.path.dirname(__file__)

"""
test files for prediction
test_file_on = audio_dataset_path+"\\"+"on"+"\\"+"3cc595de_nohash_1.wav"
test_file_down = audio_dataset_path+"\\"+"down"+"\\"+"b87bdb22_nohash_1.wav"
test_file_right = audio_dataset_path+"\\"+"right"+"\\"+"2aca1e72_nohash_1.wav"
"""
#relative and universal (daj boże) paths (LINUX)
test_file_on = os.path.join(relative_dir,"on","3cc595de_nohash_1.wav")
test_file_down = os.path.join(relative_dir, "down","87bdb22_nohash_1.wav")
test_file_right = os.path.join(relative_dir, "right","aca1e72_nohash_1.wav")


class_label = ["down","go","left","on","right","stop","up"]
wav_sample_rate = 16000
num_mfcc = 40

#number of spectograms to make (per class)
num_files = 500
num_epochs = 120
num_batch = 32

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

def MFCC_svfig(data):
    
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
    #num_ceps = 13
    num_ceps = 40
    num_lifter = 22
    #num_lifter =40 

    sample_freq, raw_seq = wav.read(data)
    
    #sample_freq=16000

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
    
    scaled =np.mean(mfcc_feat.T, axis=0)
    #plt.matshow(mfcc_feat)
    #plt.savefig("mfcc.png")
    #plt.show()
    return scaled
###################################################################
#makes mfc spectrogram out of .wav file and rescales it
def get_spectrogram(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc = num_mfcc)
        scaled = np.mean(mfcc.T, axis=0)
    except Except as e:
        print("Error with file: ", file_name)
        return None, None
    return scaled



#iterates through all of files in dataset and makes spectrograms out of them
#saves spectrograms in numpy DataFrame (excel-like sheet)



def spectro_bot(dataset_path):
    entries = []
    start_time = time.time()
    for dir_name in class_label:
        print(dir_name)
        label_index = class_label.index(dir_name)
        dir_path = os.path.join(dataset_path,dir_name)
        i = 0
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path,file_name)
            #data = get_spectrogram(file_path)
            data = MFCC_svfig(file_path)
            entries.append([data, label_index])
            i=i+1
            if (i==num_files):
                break
    entries_data_frame = pd.DataFrame(entries, columns=["entries", "label"])
    entries_data_frame = entries_data_frame.sample(frac=1).reset_index(drop=True)
    finish_time = time.time()
    print("Finished processing {} files in {} seconds".
          format(len(entries_data_frame), finish_time-start_time))
    return entries_data_frame



#make spectrograms
data_frame = spectro_bot(audio_dataset_path)
# Convert features and corresponding classification labels into numpy arrays
X = np.array(data_frame.entries.tolist())
y = np.array(data_frame.label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

#split dataset
x_train, x_test, y_train, y_test = train_test_split(X, yy,
                                                    test_size=0.25,
                                                    random_state = 42)



num_labels = yy.shape[1]
#=====================MODEL===========================
model = Sequential()
#input layer
model.add(Dense(300, input_shape=(num_mfcc,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
#second layer
model.add(Dense(600))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#output layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
#====================================================



#compile
model.compile(optimizer='Nadam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

#train
start_time = time.time()
model.fit(x_train, y_train, 
          batch_size=num_batch, 
          epochs=num_epochs, 
          validation_data=(x_test, y_test), 
          verbose=1)
finish_time = time.time()
print("Training finished in {} seconds".format(finish_time-start_time))



# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])



#makes spectrogram out of .wav file for prediction
#returns different format than get_spectrogram(), usable only in
#print_prediction() function
def extract_feature(file_name):
   
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None, None

    return np.array([mfccsscaled])



#prints prediction in 
def print_prediction(file_name,model):
    prediction_feature = extract_feature(file_name)
    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", class_label[predicted_class[0]], '\n') 
    predicted_proba_vector = model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )



print_prediction(test_file_right, model)






