import numpy as np
import pygame
import sys
import math
import keras
import pandas as pd
import subprocess

from sklearn.preprocessing import StandardScaler


print(sys.argv[0]) # file.py

print("Load initial model")

model1filename = sys.argv[1]
model2filename = sys.argv[2]

model1 = keras.models.load_model(model1filename)
model2 = keras.models.load_model(model2filename)
scaler = StandardScaler()

try:
    while True:
        print("starting another game")
        cmd = ['python', 'connect4game_bot_vs_bot_with_learning.py', model1filename, model2filename]
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
        model1.save(model1filename)
        model2.save(model2filename)
        
except KeyboardInterrupt:
    pass