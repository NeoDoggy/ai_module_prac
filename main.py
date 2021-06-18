from numpy import genfromtxt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pydot
data = genfromtxt('qq.csv', delimiter=',')
x=data[:,:-1]
y=data[:,-1]
xtr=x[:-100]
ytr=y[:-100]
xts=x[-100:]
yts=y[-100:]
ytr=np_utils.to_categorical(ytr,10)
yts=np_utils.to_categorical(yts,10)
model=Sequential()
model.add(Dense(input_dim=8,units=256,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=xtr,
                        y=ytr,
                        validation_split=0.2,
                        epochs=1000,
                        batch_size=600,
                        verbose=2)
#plt.plot(train_history.history['loss'])
plt.plot(train_history.history['accuracy'])
plt.show()
model.evaluate(xts,yts,batch_size=1000)
prediction=model.predict_classes(xts)
print(prediction[:10])