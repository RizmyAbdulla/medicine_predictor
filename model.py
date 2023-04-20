import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

# Step 1: Load the dataset
data = pd.read_csv('dataset.csv')

# Step 2: Preprocess the dataset
mlb = MultiLabelBinarizer()
symptoms = [s.split(',') for s in data['Symptoms']]
X = mlb.fit_transform(symptoms)
disease = pd.get_dummies(data['Disease'])
Y = disease.values

# Step 3: Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Step 4: Build the model
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X.shape[1], 1)))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(256, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(Y.shape[1], activation='softmax'))

# Step 5: Compile and train the model
optimizer = RMSprop(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(np.expand_dims(X_train, axis=2), Y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=2)

model.save('model.h5')

# Step 6: Evaluate the model on the testing set
loss, accuracy = model.evaluate(np.expand_dims(X_test, axis=2), Y_test, verbose=0)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')
