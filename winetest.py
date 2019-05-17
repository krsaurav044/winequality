# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:46:52 2019

@author: saurav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset1=pd.read_csv('winequality-white.csv')
dataset1=dataset1.iloc[:,0].values
dataset1=dataset1.reshape(4898,1)

dataset2=pd.read_csv('winequality-red.csv')
dataset2=dataset2.iloc[:,0].values
dataset2=dataset2.reshape(1599,1)

X=[]

for i in range(0,4898):
    t=dataset1[i][0].split(';')
    X.append(t)
    
for i in range(0,1599):
    t=dataset2[i][0].split(';')
    X.append(t)

X=np.asarray(X)
X=X.astype(np.float)

y=X[:,11]
X=X[:,0:11]
y=y.reshape(6497,1)

for i in range(0,6497):
    y[i][0]=y[i][0]-3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
X_train=st.fit_transform(X_train)
X_test=st.transform(X_test)

X_train=X_train.reshape(4872,11,1)
X_test=X_test.reshape(1625,11,1)

from keras import backend as K
from keras import regularizers
from keras import losses

def correntropy(sigma=0.05):
    def func(y_true, y_pred):
        return (K.mean((1/(np.sqrt(2*np.pi))*sigma)*K.exp(-K.square(y_true - y_pred)/2*sigma*sigma), -1) + 
                K.mean((1/(np.sqrt(2*np.pi))*sigma)*K.exp(-K.square(losses.kullback_leibler_divergence(y_true, y_pred))/2*sigma*sigma)))
    return func



from sklearn.preprocessing import OneHotEncoder
en=OneHotEncoder(categorical_features=[0])
y_train=en.fit_transform(y_train).toarray()

from keras.models import Sequential,Model
from keras.layers import Dense,Input, Conv2D,Conv2DTranspose,Convolution2D, MaxPooling2D, Convolution2DTranspose, UpSampling2D
from keras.layers import BatchNormalization, Dropout, Flatten

input_ =Input(shape=(11,))

x=Dense(output_dim=32,activation='relu')(input_)

x=Dense(output_dim=16,activation='relu')(x)

x=Dense(output_dim=8,activation='relu')(x)

encoded=Dense(output_dim=4,activation='relu')(x)


x=Dense(output_dim=4,activation='relu')(encoded)

x=Dense(output_dim=8,activation='relu')(x)

x=Dense(output_dim=16,activation='relu')(x)

x=Dense(output_dim=32,activation='relu')(x)

decoded=Dense(output_dim=11,activation='relu')(x)

autoencoder=Model(input_,decoded)
autoencoder1=Model(input_,encoded)

corentropy=correntropy()
autoencoder.compile(optimizer='adam',loss=corentropy)
history1=autoencoder.fit(X_train,X_train,validation_split=0.1,batch_size=64,epochs=50)

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

for l1,L2 in zip(autoencoder1.layers[:9],autoencoder.layers[0:9]):
    l1.set_weights(L2.get_weights())
    
X_t = autoencoder.predict(X_train)
X_t = X_t.reshape(4872,11)
X_te = autoencoder.predict(X_test)
X_te = X_te.reshape(1625,11)

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
X_t=st.fit_transform(X_t)
X_te=st.transform(X_te)

model=Sequential()
model.add(Dense(output_dim=16,activation='relu',input_dim=11))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
model.add(Dense(output_dim=32,activation='relu'))
model.add(Dense(output_dim=64,activation='relu'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
model.add(Dense(output_dim=8,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01)))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history2=model.fit(X_t,y_train,validation_split=0.1,batch_size=64,nb_epoch=50)
y_pred = model.predict(X_te)

plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



y_pred=en.inverse_transform(y_pred)


from sklearn.metrics import accuracy_score
accuracy1=accuracy_score(y_pred,y_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test, y_pred)

















