#biblioteki
import pandas as pd
import os
from Bots import SpektroBot
import time

start_time = time.time()
#iteruj po kazdym pliku w directory i rob z niego MFCC
#zapisuj dane do data_array w formacie [dane, nazwa_komendy]
directory = "go"
data_array = []
with os.scandir(directory) as dir_name:
    for file in dir_name:
        data = SpektroBot(file)
        data_array.append([data, directory])


#wrzuc dane do DataFrame z biblioteki Panda
data_frame = pd.DataFrame(data_array, columns=["data","class_label"])
print("dlugosc frame: ", len(data_frame))
print("czas trwania: %s sekund" % (time.time() - start_time))
data_frame.to_csv("go_frame")
