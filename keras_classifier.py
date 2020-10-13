import os

import pandas as pd
import numpy as np
import tensorflow as tf

from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from preprocessing import process_x


def sequential_model():
    model = Sequential()
    model.add(Dense(8, input_dim=17, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

x_train_df = pd.read_csv('Xtrain.csv', index_col=False).values
y_train_df = pd.read_csv('Ytrain.csv', index_col=False).values
encoder = LabelEncoder()
encoder.fit(y_train_df)
encoded_Y = encoder.transform(y_train_df)
encoded_y = np_utils.to_categorical(encoded_Y)

estimator = KerasClassifier(build_fn=sequential_model, epochs=100, batch_size=10, verbose=1)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, x_train_df, encoded_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# process_x((), os.path.join(os.path.dirname(__file__), 'raw_data', 'Xtest.csv'), 'Xtest.csv')

# x_train_df = pd.read_csv('Xtrain.csv', index_col=False)
# y_train_df = pd.read_csv('Ytrain.csv', index_col=False)
# x_test_df = pd.read_csv('Xtest.csv', index_col=False)
# x_train_tensor = tf.convert_to_tensor(x_train_df)
# y_train_tensor = tf.convert_to_tensor(y_train_df)
# x_test_tensor = tf.convert_to_tensor(x_test_df)
# dataset = tf.data.Dataset.from_tensor_slices((x_train_df.values, y_train_df.values))
# train_dataset = dataset.shuffle(len(x_train_df)).batch(1)


# print(f"start to training model at: {datetime.now()}")
# model.fit(x_train_tensor, y_train_tensor, epochs=100, batch_size=10, verbose=0)

# print(f"start to predict at: {datetime.now()}")
# predictions = model.predict(x_test_tensor)
# pd.DataFrame(predictions).to_csv('Yprediction.csv', index=False)
print('finish....')
