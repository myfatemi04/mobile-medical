from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from sklearn.model_selection import train_test_split

from eyedata import get_training_data


def get_lstm_model():
    model = models.Sequential([
        layers.LSTM(10, input_shape=(3, 100), activation='tanh'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


"""
Training Data Format

X: ndarray of sequences, each sequence is an ndarray of shape (3, 100), where 100 = lookback length and 3 = features at each timestep
y: ndarray of one-hot encoded labels, each label is an ndarray of shape (3) corresponding to difficult/easy cognitive tasks or control eye movements

"""
X, y = get_training_data()

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train[0] = scaler.fit_transform(X_train[0].T).T
for i in range(1, len(X_train)):
    X_train[i] = scaler.transform(X_train[i].T).T

for i in range(len(X_test)):
    X_test[i] = scaler.transform(X_test[i].T).T

model = get_lstm_model()
model.fit(X_train, y_train, epochs=25)
model.evaluate(X_test, y_test)
