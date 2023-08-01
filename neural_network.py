import pandas as pd
import numpy as np
import keras as kr
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical



#read music csv
df = pd.read_csv("genres_v2.csv")

#Splitting the Data by I/O
df = df.sample(frac=1).reset_index(drop=True)
inputs = ['danceability', 'valence','tempo',
            'energy','key','duration_ms','loudness',
            'mode', 'time_signature', 'speechiness',
            'acousticness', 'instrumentalness','liveness']
X = df[inputs]
y = df['genre'] 

X = np.array(X)

#PreProcessing: Encode Classes 
encoder = LabelEncoder()
encoder.fit(y)
enc_y = encoder.transform(y)
dummy_y = to_categorical(enc_y) #one hot encoded



#Train Test Split with validation set (80-10-10)
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, train_size=0.8)

#Building the Neural Network
ep = 1000
model = Sequential()
model.add(Dense(50, input_dim=13, activation='relu')) #input and hidden layer with ReLu activation
#model.add(Dense(14, activation='relu'))
#model.add(Dense(14, activation='relu')) #3 hidden layers
#model.add(Dense(14, activation='relu'))
#model.add(Dense(14, activation='relu')) #5 hidden layers
model.add(Dense(15, activation='softmax')) #output layer with softmax activation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

#Callback stop training when there is no improvement for 10 straight loops
es = kr.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=300, restore_best_weights=True)

#Fit Model
history = model.fit(X_train, y_train, callbacks=[es], 
                    epochs=ep, batch_size=64, shuffle=True,
                    verbose=1, validation_split = 0.2)

#Evaluation
history_dict = history.history

#learning curve
#accuracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

#loss
loss = history_dict['val_loss']
val_loss = history_dict['val_accuracy']

#range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

#plot
#"r" is for solid red line
plt.plot(epochs, acc, 'r', label='Training accuracy')
#b is for solid blue line
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#Model Evaluation
preds = model.predict(X_train)
matrix = confusion_matrix(y_train.argmax(axis=1), preds.argmax(axis=1))

print(classification_report(y_train.argmax(axis=1), preds.argmax(axis=1)))