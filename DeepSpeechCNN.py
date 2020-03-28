#!/usr/bin/env python
# coding: utf-8

# In[1]:
#import for mfcc
import scipy
from scipy import fftpack
from scipy import signal

#imports
import pandas as pd
import numpy as np
import os
import librosa
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
from keras.layers import Convolution2D,Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils


# In[2]:


#Settings
audio_dataset_path = "/home/apeiron/Servus/dataset_wave"
save_model_path = "/home/apeiron/Servus/modelCNN"
#test files for prediction
test_file_on = audio_dataset_path+"/"+"on"+"/"+"3cc595de_nohash_1.wav"
test_file_down = audio_dataset_path+"/"+"down"+"/"+"b87bdb22_nohash_1.wav"
test_file_right = audio_dataset_path+"/"+"right"+"/"+"2aca1e72_nohash_1.wav"
class_label = ["down","go","left","on","right","stop","up"]
#wav_sample_rate = 16000
num_mfcc = 40
#number of spectrograms to make (per class)
num_files = 1500

num_epochs = 40
num_batch = 64
#padding for mfcc spectrograms 
PADDING = 68


# In[3]:

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

    pad = 68 - mfcc_feat.shape[1]
    mfcc = np.pad(mfcc_feat, pad_width=((0,0), (0,pad)), mode="constant")
    #scaled =np.mean(mfcc_feat.T, axis=0)
    #plt.matshow(mfcc_feat)
    #plt.savefig("mfcc.png")
    #plt.show()
    return mfcc
###################################################################

#make mfc spectrogram out of .wav file and apply padding to it
def get_spectrogram(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc = num_mfcc)
        pad = PADDING - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0), (0,pad)), mode="constant")
    except Except as err:
        print("Error with file: ", file_name)
        return None, None
    return mfcc


# In[4]:


#iterate through all of files in dataset and make spectrograms out of them
#save spectrograms in numpy DataFrame (excel-like sheet)
def spectro_bot(dataset_path):
    entries = []
    start_time = time.time()
    for dir_name in class_label:
        print(dir_name)
        label_index = class_label.index(dir_name)
        dir_path = dataset_path+"/"+dir_name
        i = 0
        for file_name in os.listdir(dir_path):
            file_path = dir_path+"/"+file_name
            #data = get_spectrogram(file_path)
            data=MFCC_svfig(file_path) 
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


# In[5]:


#make spectrograms
data_frame = spectro_bot(audio_dataset_path)
#move dataframe entries into np. array
X = np.array(data_frame.entries.tolist())
y = np.array(data_frame.label.tolist())

#encode labels
encoder = LabelEncoder()
encoded_labels = to_categorical(encoder.fit_transform(y))

#split dataset
x_train, x_test, y_train, y_test = train_test_split(X, encoded_labels,
                                                    test_size=0.25,
                                                    random_state = 42)

#reshape
num_rows = num_mfcc
num_columns = PADDING
num_channels = 1
x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)
num_labels = encoded_labels.shape[1]


# In[6]:


#=============================CNN MODEL===============================
model = Sequential()
#first conv
model.add(Conv2D(filters=16, kernel_size=3,
                 input_shape=(num_rows, num_columns, num_channels),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2, padding="same"))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2, padding="same"))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2, padding="same"))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2, padding="same"))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))


# In[7]:


#compile
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')
model.summary()
#pre-training acc
acc = model.evaluate(x_test,y_test,verbose=1)
print("Pre training: {}".format(100*acc[1]))


# In[8]:


#train
start_time = time.time()
history = model.fit(x_train, y_train,
                    batch_size=num_batch, 
                    epochs=num_epochs, 
                    validation_data=(x_test, y_test), 
                    verbose=1)
finish_time = time.time() - start_time
print("Training time: {} seconds".format(finish_time))

#save model
model.save(save_model_path)


# In[9]:


#evaluate
acc = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", acc[1])
acc = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", acc[1])


# In[10]:


#make prediction and print it in human-readable format
def make_prediction(file_name, model):
    spectro = get_spectrogram(file_name)
    spectro = spectro.reshape(1, num_rows, num_columns, num_channels)
    prediction = model.predict_classes(spectro)
    predicted_class = encoder.inverse_transform(prediction)
    
    print("Predicted class:", class_label[predicted_class[0]], '\n') 
    
    #probabilities
    prediction_prob = model.predict_proba(spectro) 
    predicted = prediction_prob[0]
    for i in range(len(predicted)): 
        category = encoder.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted[i], '.32f') )


# In[11]:


make_prediction(test_file_right, model)


# In[12]:


#plot
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.title('Training and validation accuracy')
plt.legend()
fig = plt.figure()
fig.savefig('acc.png')


plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss')

plt.legend()
plt.show()


# In[ ]:




