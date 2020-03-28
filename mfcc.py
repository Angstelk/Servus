import scipy
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import time
from scipy.io import wavfile as wav
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics 

"""
librosa.feature.mfcc 

"""

PADDING = 64
num_mfcc = 40
def get_spectrogram(file_name):
        audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc = num_mfcc)
        pad = PADDING - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0), (0,pad)), mode="constant")
   # except Except as err:
   #     print("Error with file: ", file_name)
    #    return None, None
        return mfcc
######################################################################################################################
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
    
    print(sample_freq)
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
    
    mfcc_feat = np.asarray(mfcc_feat, dtype=np.float32)

    pad = PADDING - mfcc_feat.shape[1]
    
    #mfcc = np.pad(mfcc_feat, pad_width=((0,0), (0,pad)), mode="constant")
    #scaled =np.mean(mfcc_feat.T, axis=0)
    #plt.matshow(mfcc_feat)
    plt.savefig("mfcc.png")
    #plt.show()
    return mfcc_feat
def main():
    shape1=MFCC_svfig("/home/apeiron/Servus/go_test.wav")
    print(shape1.shape)
    print("ma być : \n")
    shape2=get_spectrogram("/home/apeiron/Servus/go_test.wav")
    print(shape2.shape)
if __name__ == '__main__':
    main()
