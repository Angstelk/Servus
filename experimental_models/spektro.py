import os
import wave
import pylab

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def save_spectr(wav_file,png_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(8.5, 6))
    pylab.subplot(111)
    pylab.specgram(sound_info, Fs=frame_rate)
    dir_str=str(directory_png) 
    dir_str=dir_str[2:-1]
    name_spect= str(dir_str)+'/' +str( png_file)
    pylab.savefig(name_spect)
    pylab.close()

directory_in_str="/home/apeiron/Servus/dataset_wave"
directory_png="/home/apeiron/Servus/dataset_png"  
directory = os.fsencode(directory_in_str)
directory_png =os.fsencode(directory_png) 

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".wav"):
         # print(os.path.join(directory, filename))
         #graph_spectrogram(filename)
         continue
     else:
         continue
label=0
ind=0
for path0, subdir0,files0 in os.walk(directory):
    for sd in subdir0:
                            # dostaję subdir w dir
        name_dir = str(sd)
        name_dir=name_dir[2:-1]   
                            # mam stringa którego trzeba
        ind=0
        directory1 = str(os.path.join(path0,sd))
        directory1 =directory1[2:-1]
        #print(directory1)
       
        for path1, subdir1,files1 in os.walk(directory1):
            for name in files1:
            
                wave_file = str(os.path.join(directory1,name))
                wave_file = wave_file[:]
               # print(wave_file)
                name_ost=str(label) +'_'+name_dir+"_"+ str(ind)+".png"
                save_spectr(wave_file,name_ost)
                ind=ind+1
                if ind > 50:
                    break
        label=label+1
     
        # dla każdego pliku w directory_in_str
