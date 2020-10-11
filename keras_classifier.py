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


x_train_df = pd.read_csv('Xtrain.csv', index_col=False)
y_train_df = pd.read_csv('Ytrain.csv', index_col=False)
# x_test_df = pd.read_csv('Xtest.csv', index_col=False)
x_train_tensor = tf.convert_to_tensor(x_train_df)
y_train_tensor = tf.convert_to_tensor(y_train_df)
# x_test_tensor = tf.convert_to_tensor(x_test_df)
# dataset = tf.data.Dataset.from_tensor_slices((x_train_df.values, y_train_df.values))
# train_dataset = dataset.shuffle(len(x_train_df)).batch(1)



model = Sequential()
model.add(Dense(8, input_dim=16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam')
print(f"start to training model at: {datetime.now()}")
model.fit(x_train_tensor, y_train_tensor, epochs=100, batch_size=10, verbose=0)

print(f"start to training model at: {datetime.now()}")
# predictions = model.predict(x_test_tensor)
print('finish....')
