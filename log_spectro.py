from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

sf, audio = wavfile.read('go_test.wav')
sig = np.mean(audio, axis=1)
f, t, Sxx = signal.spectrogram(sig, sf, scaling='spectrum')

plt.pcolormesh(t, f, np.log10(Sxx))
plt.ylabel('f [Hz]')
plt.xlabel('t [sec]')
plt.show()

