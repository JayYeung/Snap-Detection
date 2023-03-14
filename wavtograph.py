import os
import moviepy.editor as mp
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Define places
SNAPS_DIR = 'snaps/'
SNAPS_WAV_DIR = 'snaps_wav/'
SNAPS_PROCESSED_DIR = 'snaps_processed/'

NOT_SNAPS_DIR = 'not_snaps/'
NOT_SNAPS_WAV_DIR = 'not_snaps_wav/'
NOT_SNAPS_PROCESSED_DIR = 'not_snaps_processed/'

for filename in os.listdir(NOT_SNAPS_WAV_DIR):
    if filename.endswith('.wav'):
        # Load the audio file using librosa
        audio, sr = librosa.load(os.path.join(NOT_SNAPS_WAV_DIR, filename))
        print(audio.shape)
        # Compute the spectrogram using the stft
        spec = librosa.stft(audio)
        spec_db = librosa.amplitude_to_db(abs(spec))
        print(spec_db.shape)
        
        # Create the output filename by replacing the extension
        output_filename = os.path.splitext(filename)[0] + '.png'
        
        # Save the spectrogram as a PNG image
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(spec_db, sr=sr, cmap='magma')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(NOT_SNAPS_PROCESSED_DIR, output_filename))
        plt.close()