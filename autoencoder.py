from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(layers):
    model = Sequential()
    
    for layer in layers[:-1]:
        model.add(Dense(layer, activation='relu'))
    model.add(Dense(layers[-1], activation='linear'))

    return model
