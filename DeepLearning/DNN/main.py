import tensorflow as tf
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

# 1. Load the data from .mat file
loc_x = scipy.io.loadmat('LOC_2_UE_x.mat')['LOC_2_UE_x']
loc_y = scipy.io.loadmat('LOC_2_UE_y.mat')['LOC_2_UE_y']
loc_ris = scipy.io.loadmat('LOC_2_UE_RIS.mat')['LOC_2_UE_RIS']

# 2. Preparing the input and output data
X_data = np.concatenate((loc_x, loc_y), axis=0).T
Y_data = loc_ris.T

# 3. Deciding the data used for training and validation,
# the dataset utilized in this project consists of 19,054 sets
X_train = X_data[:16000]
Y_train = Y_data[:16000]
X_val = X_data[18001:19000]
Y_val = Y_data[18001:19000]

# 4. Feature scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
Y_train_scaled = scaler_y.fit_transform(Y_train)

X_val_scaled = scaler_x.transform(X_val)
Y_val_scaled = scaler_y.transform(Y_val)

# 5. Model Construction
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(16,)),
    tf.keras.layers.Dense(512),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('elu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('elu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('elu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('elu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('elu'),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(24, activation='sigmoid')
])

# 6. Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='mse')

# Record the entire running time
start_time = time.time()

# 7. Training
history = model.fit(X_train_scaled, Y_train_scaled, epochs=1000, batch_size=24, validation_data=(X_val_scaled, Y_val_scaled))

# Save the recorded time as a .mat file
end_time = time.time()
training_time = end_time - start_time
sample_count = X_train.shape[0]
training_time_dict = {'training_time_' + str(sample_count): training_time}
scipy.io.savemat(f'training_time_{sample_count}.mat', training_time_dict)


# Record the loss and val loss value after each epoch
# And save them as a 3xN matrix , N is the num of epochs
# The first row is the loss value, the second row is the validation loss value, and the third row is the epoch counts
loss_history = np.array(history.history['loss'])
val_loss_history = np.array(history.history['val_loss'])
epoch_indices = np.arange(1, len(loss_history) + 1)
loss_matrix = np.vstack((loss_history, val_loss_history, epoch_indices))

sample_count = X_train.shape[0]
epoch_count = len(loss_history)
scipy.io.savemat(f'mse_epoch_{sample_count}_{epoch_count}.mat', {'mse_epoch': loss_matrix})
