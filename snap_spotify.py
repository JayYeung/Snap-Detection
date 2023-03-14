import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyaudio
import time
import wave
import spotipy
from spotipy.oauth2 import SpotifyOAuth

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

# Spotify authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='6206f9ef6514488b95a6e6e04d49e71b',
                                            client_secret='3e8fe128e1ee4fcd9c65d94260397238',
                                            redirect_uri='http://localhost:8888/callback',
											scope='user-modify-playback-state user-read-playback-state'))

print('connected to spotify')
print(sp.devices())

while True:
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

        # Get device ID
        devices = sp.devices()
        if not devices['devices']:
            print("No devices found")
            exit()
        device_id = devices['devices'][0]['id']

        # Skip to next track
        current_track = sp.current_playback()
        if current_track:
            track_id = current_track['item']['id']
            sp.next_track(device_id=device_id)
            print("skipping")
        else:
            print("bro ur not even playing anything?!?!")
    else:
        print("No snap")



# stop stream and close PyAudio
stream.stop_stream()    
stream.close()
audio.terminate()
