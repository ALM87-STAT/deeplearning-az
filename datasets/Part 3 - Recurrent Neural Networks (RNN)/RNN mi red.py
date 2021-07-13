# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:35:57 2021

@author: Alejandro Lopera Marín
"""

# Parte 1 - Preprocesado de los datos

# Importar las librerias 
import numpy as np
import pandas as pd
import matplotlib as plt

# Importar el dataset de entrenamiento
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

# Escalado de caracteristicas
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Crear una estructura de datos con 60 timesteps y 1 salida y rimensionado de los datos
def lstm_data_transform(x_data, y_data, num_steps = 5):
    """ Changes data to the format for LSTM training for sliding window approach """
    # Prepare the list for the transformed data
    X, y = list(), list()
    # Loop of the entire data set
    for i in range(x_data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps
        # if index is larger than the size of the dataset, we stop
        if end_ix >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = x_data[i:end_ix]
        # Get only the last element of the sequency for y
        seq_y = y_data[end_ix]
        # Append the list with sequencies
        X.append(seq_X)
        y.append(seq_y)
    # Make final arrays
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array, y_array
X_train_prueba, y_train_prueba = lstm_data_transform(training_set_scaled, training_set_scaled, 60)

# Parte 2 - Construcción de la RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Inicialización del modelo 
regressor = Sequential()

# Añadir la primera capa LSTM y la regularización Dropuot
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Añadir la segunda capa LSTM y la regularización Dropuot
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Añadir la tercera capa LSTM y la regularización Dropuot
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Añadir la cuarta capa LSTM y la regularización Dropuot
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Añadir la capa de salida
regressor.add(Dense(units = 1))

# Compilar la RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Ajustar la RNR al conjunto de entrenamiento
history = regressor.fit(X_train, y_train, validation_split = 0.20, epochs = 100, batch_size = 32)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Parte 3 - Ajustar las preducciones y visualizar los resultados 

# Obtener el valor de las acciones reales  de Enero de 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Obtener la predicción de la acción con la RNR para Enero de 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test, _ = lstm_data_transform(inputs, inputs, 60)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

mse = mean_squared_error(X_test, predicted_stock_price)

# Visualizar los Resultados
plt.plot(real_stock_price, color = 'red', label = 'Precio Real de la Accion de Google')
plt.plot(predicted_stock_price, color = 'blue', label = 'Precio Predicho de la Accion de Google')
plt.title("Prediccion con una RNR del valor de las acciones de Google")
plt.xlabel("Fecha")
plt.ylabel("Precio de la accion de Google")
plt.legend()
plt.show()




