import numpy as np

# Imports de Keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN

def rossler_attractor(a, b, c, dt, num_steps):
    x = np.zeros(num_steps)
    y = np.zeros(num_steps)
    z = np.zeros(num_steps)

    x[0] = 0.1
    y[0] = 0.0
    z[0] = 0.0

    for i in range(1, num_steps):
        x[i] = x[i-1] + (-y[i-1] - z[i-1]) * dt
        y[i] = y[i-1] + (x[i-1] + a * y[i-1]) * dt
        z[i] = z[i-1] + (b + z[i-1] * (x[i-1] - c)) * dt

    return x, y, z

def split_sequence(sequence, look_back):
    X, y = list(), list()
    # Recorremos la serie correspondiente al train y armamos el dataset
    for i in range(look_back, len(sequence)):
        X.append(sequence[i-look_back:i])
        y.append(sequence[i])
    X, y = np.array(X), np.array(y)
    return X, y

def prediccion_pasos_adelante(model,vec_actual,pasos_adelante):

    # Preparamos una lista vacia que vamos a ir llenando con los valores predichos
    lista_valores = []
    # Recorremos n pasos hacia adelante
    for i in range(pasos_adelante):

        # Predecimos el paso siguiente
        # (El if determina si la estamos usando para la red recurrente o la densa)
        if len(vec_actual.shape)>1:
            # Prediccion Red Recurrente
            nuevo_valor = model.predict(vec_actual.reshape((1, vec_actual.shape[0], vec_actual.shape[1])))
        else:
            # Prediccion Red Densa
            nuevo_valor = model.predict(vec_actual.reshape(1, vec_actual.shape[0]))

        # Lo agregamos a la lista
        lista_valores.append(nuevo_valor[0][0])

        # Actualizmaos el vector actual con este paso
        vec_actual = np.roll(vec_actual, -1)
        vec_actual[-1] = nuevo_valor[0][0]

    lista_valores = np.asarray(lista_valores)

    return lista_valores

def make_predictions(sequence, look_back, n_features, raw_seq_window, model = None, **fit_kwrds):
    # split into samples
    X_train, y_train = split_sequence(sequence, look_back)
    # Le damos la forma adecuada a los datos para entrar a la red recurrente
    # reshape from [samples, timesteps] into [samples, look_back, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))


    X_test, y_test = split_sequence(raw_seq_window, look_back)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

    tamano_ventana = len(raw_seq_window)

    if not model:
    # Definimos el modelo Secuencial
        model = Sequential()
        # Agregamos la capa Recurrente (Puede ser LSTM o RNN)
        
        model.add(SimpleRNN(20, activation='tanh', input_shape=(look_back, n_features)))
        #model.add(SimpleRNN(20, activation='relu', input_shape=(look_back, n_features)))
        #model.add(LSTM(20, activation='tanh', input_shape=(look_back, n_features)))    
        # Agregamos la capa de salida (lee el hidden Space de la recurrente)
        model.add(Dense(1,activation='linear'))
        # Compilamos el modelo
        model.compile(optimizer='adam', loss='mse')

    if not fit_kwrds:
        fit_kwrds = {'epochs' : 200,
                     'verbose' : 0}    

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), **fit_kwrds)

    # Tomamos un valor inicial del test set
    indice_inicial = 0
    vec_actual = X_test[indice_inicial]
    #vector_actual = X_test[-5]
    #vec_actual = X_train[-1] #CAMBIO IMPORTANTE PARA QUE LA PREDICION ESTÉ EN FASE

    # Calculamos la prediccion del modelo
    predicciones_adelante = prediccion_pasos_adelante(model,vec_actual,tamano_ventana)
    # Tomamos los valores esperados
    valores_reales = raw_seq_window

    return predicciones_adelante,valores_reales    

