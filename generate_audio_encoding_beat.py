import os
import sys
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa

# Constants matching the repo's configuration
AUDIO_SAMPLE_RATE = 16000
VIDEO_FPS = 25  # Based on the DnD dataset FPS
WINDOW_SIZE = 128  # Based on the standard window size used in the repo
AUDIO_WINDOW_LENGTH = int((WINDOW_SIZE / VIDEO_FPS) * AUDIO_SAMPLE_RATE)


def load_audio(audio_path):
    """Load and normalize audio file"""
    try:
        audio, _ = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
        if len(audio) == 0:
            print(f"\tWarning: Empty audio file: {audio_path}")
            return None
        audio = librosa.util.normalize(audio)
        return audio
    except Exception as e:
        print(f"\tError loading audio file {audio_path}: {str(e)}")
        return None


def process_speaker_directory(speaker_dir, module):
    """Process all audio files in a speaker's directory"""
    # Get all wav files in the speaker directory
    audio_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]

    pbar = tqdm(total=len(audio_files), desc=f'Processing {os.path.basename(speaker_dir)}')

    for audio_file in audio_files:
        wav_path = os.path.join(speaker_dir, audio_file)

        # Skip if embedding already exists
        output_file = wav_path.replace('.wav', '_audio_embedding.npy')
        if os.path.exists(output_file):
            pbar.update(1)
            continue

        # Load and normalize audio
        audio = load_audio(wav_path)
        if audio is None:
            pbar.update(1)
            continue

        try:
            # Generate embedding
            audio = np.expand_dims(audio, axis=0)
            embedding = module(audio)['embedding'].numpy()
            embedding = np.squeeze(embedding)

            # Save embedding in the same directory as the input file
            np.save(output_file, embedding)

        except Exception as e:
            print(f"\tError processing {wav_path}: {str(e)}")

        pbar.update(1)

    pbar.close()


if __name__ == '__main__':
    # Load the TensorFlow Hub module
    module = hub.KerasLayer('https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson3/1')

    # Define paths
    input_base_path = './datasets/beat_english_v0.2.1'

    # Process each speaker directory (1-30)
    speaker_dirs = sorted([d for d in os.listdir(input_base_path)
                           if os.path.isdir(os.path.join(input_base_path, d))
                           and d.isdigit()])

    # Create overall progress bar for speakers
    for speaker_dir in tqdm(speaker_dirs, desc='Processing speakers'):
        speaker_path = os.path.join(input_base_path, speaker_dir)
        process_speaker_directory(speaker_path, module)