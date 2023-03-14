import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import librosa
import numpy as np
import tensorflow as tf

# Define the directories
snaps_dir = "dataset/snaps"
not_snaps_dir = "dataset/claps_and_smacks"

# define more stuff
sr = 44100
duration = 1.0
n_mels = 128
frame_size = 2048
hop_size = 512

snaps_files = os.listdir(snaps_dir)
snaps_data = []
for file in snaps_files:
    filepath = os.path.join(snaps_dir, file)
    audio, sr = librosa.load(filepath)
    audio = audio / np.max(np.abs(audio))
    spec = librosa.stft(audio)
    spec_db = librosa.amplitude_to_db(abs(spec)) 

    # print(spec_db.shape)
    snaps_data.append(spec_db)

not_snaps_files = os.listdir(not_snaps_dir)
not_snaps_data = []
for file in not_snaps_files:
    filepath = os.path.join(not_snaps_dir, file)
    audio, sr = librosa.load(filepath)
    audio = audio / np.max(np.abs(audio))
    spec = librosa.stft(audio)
    spec_db = librosa.amplitude_to_db(abs(spec)) 
    not_snaps_data.append(spec_db)

# MY CODE BREAKS IF I DONT DO THIS
snaps_data = tf.data.Dataset.from_tensor_slices(snaps_data)
not_snaps_data = tf.data.Dataset.from_tensor_slices(not_snaps_data)

# Create labels
snaps_labels = tf.ones(len(snaps_files))
not_snaps_labels = tf.zeros(len(not_snaps_files))

# MY CODE BREAKS IF I DONT DO THIS
snaps_labels = tf.data.Dataset.from_tensor_slices(snaps_labels)
not_snaps_labels = tf.data.Dataset.from_tensor_slices(not_snaps_labels)

# Split the dataset
total_size = len(snaps_data) + len(not_snaps_data)
train_size = int(0.75 * total_size)
val_size = total_size - train_size

train_dataset = tf.data.Dataset.zip(((snaps_data.take(int(train_size/2)).concatenate(not_snaps_data.take(int(train_size/2)))), 
                                     (snaps_labels.take(int(train_size/2)).concatenate(not_snaps_labels.take(int(train_size/2))))))
train_dataset = train_dataset.shuffle(train_size).batch(32)

val_dataset = tf.data.Dataset.zip(((snaps_data.skip(int(train_size/2)).concatenate(not_snaps_data.skip(int(train_size/2)))), 
                                   (snaps_labels.skip(int(train_size/2)).concatenate(not_snaps_labels.skip(int(train_size/2))))))
val_dataset = val_dataset.shuffle(val_size).batch(32)


for data, label in train_dataset.take(1):
    print("Input shape:", data.shape)

# # Define model
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(1025, 44, 1)),
#     tf.keras.layers.MaxPooling2D((2,2)),
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2,2)),
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# Load the saved model
model = tf.keras.models.load_model('snap_classifier_v4.h5')

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=20, validation_data=val_dataset)

print("Train accuracy:", history.history['accuracy'][-1])
print("Validation accuracy:", history.history['val_accuracy'][-1])

model.summary()

model.save('snap_not_clap.h5')