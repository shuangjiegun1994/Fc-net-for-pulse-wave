# Fc-net-for-pulse-wave

import scipy.io as scio
import numpy as np
 
dataFile = 'D://Pulsewave//3200083_0018m(2maibo6xueya).mat'
data = scio.loadmat(dataFile)

dataNew = 'D://Pulse_copy'

scio.savemat(dataNew, {'A':data['val']})
x_data=data['val'][1]
y_data=data['val'][5]

import numpy as np
x_zero=np.zeros((3000,100))
y_zero_max=np.zeros((3000,))
y_zero_min=np.zeros((3000,))
#rows=np.arange( )
x_train=[]
for i in range(3000):
    for j in range(100):
        x_zero[i,j]=abs(x_data[i*100+j])
        
print(x_zero.shape)
for i in range(3000):
    y_zero_max[i]=max(y_data[i*100:i*100+100])
    y_zero_min[i]=min(y_data[i*100:i*100+100])
print(y_zero_max,y_zero_max.shape)
print(y_zero_min,y_zero_min.shape)

x_zero=sequence.pad_sequences(x_zero,maxlen=100)
#x_test=sequence.pad_sequences(x_test,maxlen=100)
print(x_zero)
x_train=x_zero
y_train=y_zero_max

from keras.models import Sequential
from keras import layers
from keras.preprocessing import sequence

model=Sequential()
model.add(layers.Dense(1024,activation='sigmoid',input_shape=(100,),name='dense1'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024,activation='sigmoid'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='sigmoid'))
model.add(layers.Dense(512,activation='sigmoid'))
model.add(layers.Dense(256,activation='sigmoid'))
model.add(layers.Dense(128,activation='sigmoid',name='dense2'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64,activation='sigmoid',name='dense3'))
model.add(layers.Dense(32,activation='sigmoid'))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',loss='mse',metrics=['acc'])
history=model.fit(x_train,y_train,epochs=500,batch_size=400,validation_split=0.1)

import matplotlib.pyplot as plt
loss=history.history['loss']
val_loss=history.history['val_loss']

acc=history.history['acc']
val_acc=history.history['val_acc']
epochs=range(1,len(loss)+1)

plt.plot(epochs,loss,'b',label='Train_Loss')
#plt.show()
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(epochs,acc,'b',label='Train_acc')
plt.plot(epochs,val_acc,'r',label='val_acc')
plt.legend()
plt.show()


