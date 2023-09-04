import tensorflow as tf
import scipy.io
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time


loc_2_ue_x = scipy.io.loadmat('LOC_2_UE_x.mat')['LOC_2_UE_x']
loc_2_ue_y = scipy.io.loadmat('LOC_2_UE_y.mat')['LOC_2_UE_y']
loc_2_ue_d = scipy.io.loadmat('LOC_2_UE_d.mat')['LOC_2_UE_d']
loc_2_ue_ris = scipy.io.loadmat('LOC_2_UE_RIS.mat')['LOC_2_UE_RIS']


input_data = []
for i in range(loc_2_ue_x.shape[1]):
    x_data = loc_2_ue_x[:, i]
    y_data = loc_2_ue_y[:, i]
    d_data = loc_2_ue_d[:, i]
    sample = np.stack([x_data, y_data, d_data], axis=1)
    input_data.append(sample)
input_data = np.stack(input_data, axis=0)


output_data = np.transpose(loc_2_ue_ris)

scaler_input = MinMaxScaler()
scaler_output = MinMaxScaler()

train_input_data = input_data[0:16000]
train_output_data = output_data[0:16000]
val_input_data = input_data[18001:19000]
val_output_data = output_data[18001:19000]

train_input_data = np.array([scaler_input.fit_transform(sample) for sample in train_input_data])
train_output_data = scaler_output.fit_transform(train_output_data)

val_input_data = np.array([scaler_input.transform(sample) for sample in val_input_data])
val_output_data = scaler_output.transform(val_output_data)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(8, 3)),

    tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.05),

    tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.05),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(24)
])


optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

# Record the training start time
start_time = time.time()

# Train the model and save the history object
history = model.fit(train_input_data, train_output_data, epochs=1000, batch_size=24,
                    validation_data=(val_input_data, val_output_data))

# Save the recorded training time as a .mat file
end_time = time.time()
training_time = end_time - start_time
sample_count = train_input_data.shape[0]
training_time_dict = {'training_time_' + str(sample_count): training_time}
scipy.io.savemat(f'training_time_CNN_{sample_count}.mat', training_time_dict)

# Record the loss and val loss value after each epoch
# And save them as a 3xN matrix, N is the num of epochs
# The first row is the loss value, the second row is the validation loss value, and the third row is the epoch counts
loss_history = np.array(history.history['loss'])
val_loss_history = np.array(history.history['val_loss'])
epoch_indices = np.arange(1, len(loss_history) + 1)
loss_matrix = np.vstack((loss_history, val_loss_history, epoch_indices))

# Save the loss matrix as a .mat file
scipy.io.savemat(f'Loss_CNN_{sample_count}.mat', {'loss_data': loss_matrix})

model.save(f'CNN_{train_input_data.shape[0]}.h5')
