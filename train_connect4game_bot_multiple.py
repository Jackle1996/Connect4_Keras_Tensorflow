import numpy as np
import pygame
import sys
import math
import keras
import pandas as pd
import subprocess
import os

from sklearn.preprocessing import StandardScaler


print(sys.argv[0]) # file.py

print("Load initial models")

modelfolder = sys.argv[1]
models = {}
for filename in os.listdir(modelfolder):
    print("load " + filename)
    models[filename] = keras.models.load_model(os.path.join(modelfolder, filename))

scaler = StandardScaler()

try:
    while True:
        
        for filename1, model1 in models.items():
            for filename2, model2 in models.items():
                model1path = os.path.join(modelfolder, filename1)
                model2path = os.path.join(modelfolder, filename2)
                
                print("starting another game")
                print(filename1 + " vs " + filename2)
                cmd = ['python', 'connect4game_bot_vs_bot_with_learning.py', model1path, model2path]
                subprocess.Popen(cmd).wait()
                print("game ended")

                print("loading board data of game")
                data = pd.read_csv('endboard.csv', sep=',', header=None)
                winner = data.values[:,-1]
                y_train = keras.utils.to_categorical(winner, num_classes=3) 
                x_train = data.values[:,:-1]
                x_train = scaler.fit_transform(x_train)

                print("training models with new data")
                history1 = model1.fit(x_train, y_train, epochs=1, batch_size=1)
                history2 = model2.fit(x_train, y_train, epochs=1, batch_size=1)

                print("saving models")
                model1.save(model1path)
                model2.save(model2path)
        
except KeyboardInterrupt:
    pass