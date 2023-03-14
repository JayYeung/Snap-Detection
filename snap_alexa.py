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
import alexapy

# load the saved model
model = load_model('snap_not_clap.h5')

def preprocess(filepath):
    audio, sr = librosa.load(filepath)
    audio = audio / np.max(np.abs(audio))
    spec = librosa.stft(audio)
    spec_db = librosa.amplitude_to_db(abs(spec)) 
    return spec_db

# record settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = RATE // 10 # 100ms
RECORD_SECONDS = 1 # 1 second
WAVE_OUTPUT_FILENAME = 'dataset/output.wav'

# connect to Alexa
alexa = alexapy.Alexa()

while True:
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
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # preprocess the audio
    preprocessed_audio = preprocess(WAVE_OUTPUT_FILENAME)
    preprocessed_audio = preprocessed_audio[np.newaxis, ..., np.newaxis]
    predictions = model.predict(preprocessed_audio)

    if predictions[0][0] > 0.5:
        print("Snap detected!")
        # pause the music
        alexa.pause()
        # switch to the next song
        alexa.next()
        # resume the music
        alexa.play()
    else:
        print("No snap")
