import os
from pydub import AudioSegment

SNAPS_DIR = 'snaps_mp4/'
SNAPS_WAV_DIR = 'dataset/snaps/'
NOT_SNAPS_DIR = 'not_snaps_mp4'
NOT_SNAPS_WAV_DIR = 'dataset/not_snaps/'

is_snap = True  # toggable
target_length = 1000  # in ms, we want it to be 1 second clips only

if is_snap:
    input_dir = SNAPS_DIR
    output_dir = SNAPS_WAV_DIR
else:
    input_dir = NOT_SNAPS_DIR
    output_dir = NOT_SNAPS_WAV_DIR

for i, filename in enumerate(sorted(os.listdir(input_dir))):
    if filename.endswith('.mp4'):
        print(f"Converting file {filename}... to '{'snap' if is_snap else 'not_snap'}_{i+1}.wav' ")
        # Load the video file using PyDub
        audio = AudioSegment.from_file(os.path.join(input_dir, filename), format='mp4')
        
        output_filename = f"{'snap' if is_snap else 'not_snap'}_{i+1}.wav"
        
        # Take the average of the two stereo channels
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)  
        
        # Trim the audio to target length
        if len(audio) > target_length:
            audio = audio[target_length//2:target_length+target_length//2]
        else:
            audio = audio[:target_length]
        
        # Save the audio as a .wav file
        audio.export(os.path.join(output_dir, output_filename), format='wav')
