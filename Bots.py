
import librosa
import numpy as np


#przetwarza parametry sampli na te potrzebne do librosa
#Sample rate -> 22050
#Data values -> przedzial [-1,1]
#Stereo -> Mono
def SampleBot(file_name):
    lib_audio, lib_sample_rate = librosa.load(file_name,res_type="kaiser_fast")
    return lib_audio, lib_sample_rate



#robi MFCC - taka logarytmiczna wersja spektrogramu, podobno lepiej dziala
#dla glosu czlowieka
#Wejscie - nazwa_pliku.wav
#Wyjscie - przeskalowany MFCC, albo blad
def SpektroBot(file_name):
    try:
        audio, sample_rate = SampleBot(file_name)
        mfcc = librosa.feature.mfcc(y=audio, sr = sample_rate, n_mfcc = 40)
        scaled = np.mean(mfcc.T, axis=0)
    except Exception as err:
        print("Problem z przerabianiem pliku: ", file_name)
        return None

    return scaled

