import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyaudio
from pydub import AudioSegment
import time
import wave
import random
import shutil

# OLD_DIR = 'dataset/claps_and_smacks/'
# NEW_DIR = 'dataset/not_snaps/'
# START = 88

# for i, file_name in enumerate(os.listdir(OLD_DIR)):
#     if file_name.endswith('.wav'):
#         src_path = os.path.join(OLD_DIR, file_name)
#         dest_name = 'not_snap_{}.wav'.format(START+i)
#         dest_path = os.path.join(NEW_DIR, dest_name)
#         print('Moving {} to {}'.format(src_path, dest_path))
#         shutil.move(src_path, dest_path)

# exit()


# record settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = RATE // 10 # 100ms
RECORD_SECONDS = 1 # 1 second
WAVE_OUTPUT_FILENAME = 'dataset/output.wav'
OUTPUT_DIR = 'dataset/claps_and_smacks/'

model = load_model('snap_classifier_v3.h5')
def preprocess(filepath):
    audio, sr = librosa.load(filepath)
    audio = audio / np.max(np.abs(audio))
    spec = librosa.stft(audio)
    spec_db = librosa.amplitude_to_db(abs(spec)) 
    return spec_db

for filenumber in range(1, 1001):
    # print("recording")
    # record for 1 second
    audio = pyaudio.PyAudio()

    # open stream for recording
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    # print("done")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(os.path.join(OUTPUT_DIR, str(filenumber)+".wav"), "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f'making {filenumber}.wav')

    # wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(audio.get_sample_size(FORMAT))
    # wf.setframerate(RATE)
    # wf.writeframes(b''.join(frames))
    # wf.close()
    
    # # preprocess the audio
    # preprocessed_audio = preprocess(WAVE_OUTPUT_FILENAME)
    # preprocessed_audio = preprocessed_audio[np.newaxis, ..., np.newaxis]
    # predictions = model.predict(preprocessed_audio)

    # if predictions[0][0] > 0.5:
    #     print("Snap detected!")

    #     wf = wave.open(os.path.join(OUTPUT_DIR, str(filenumber)+".wav"), "wb")
    #     wf.setnchannels(CHANNELS)
    #     wf.setsampwidth(audio.get_sample_size(FORMAT))
    #     wf.setframerate(RATE)
    #     wf.writeframes(b''.join(frames))
    #     wf.close()

    #     print(f'making {filenumber}.wav')
    # else:
    #     print("No snap")

