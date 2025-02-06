import os

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
    audio, _ = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
    audio = librosa.util.normalize(audio)
    return audio

def audio_chunks(data, length):
    """Split audio into chunks"""
    for i in range(0, len(data), length):
        yield data[i:i + length]

def audio_embeddings(chunks, module):
    """Generate embeddings for audio chunks"""
    for chunk in chunks:
        # Ensure chunk is the right length by padding if necessary
        if len(chunk) < AUDIO_WINDOW_LENGTH:
            chunk = np.pad(chunk, (0, AUDIO_WINDOW_LENGTH - len(chunk)))
        chunk = np.expand_dims(chunk, axis=0)
        embedding = module(chunk)['embedding'].numpy()
        yield np.squeeze(embedding)

def process_session(session_path, module):
    """Process all audio files in a session"""
    print(f'Processing session: {os.path.basename(session_path)}')
    
    # Process each speaker's audio files
    utterance_folders = sorted([f for f in os.listdir(session_path) if not f.startswith('.')])
    
    for utterance in utterance_folders:
        utterance_path = os.path.join(session_path, utterance)
        
        # Process speaker and listener audio files
        audio_files = [f for f in os.listdir(utterance_path) if f.endswith('.wav')]
        
        for audio_file in audio_files:
            wav_path = os.path.join(utterance_path, audio_file)
            
            # Load and normalize audio
            audio = load_audio(wav_path)
            
            # Generate embedding
            audio = np.expand_dims(audio, axis=0)
            embedding = module(audio)['embedding'].numpy()
            embedding = np.squeeze(embedding)
            
            # Save embedding in the same directory as the input file
            output_file = os.path.join(utterance_path, 
                                     audio_file.replace('.wav', '_audio_embedding.npy'))
            np.save(output_file, embedding)
            
            print(f'\tProcessed {wav_path}')

if __name__ == '__main__':
    # Load the TensorFlow Hub module
    module = hub.KerasLayer('https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson3/1')
    
    # Define paths
    input_base_path = './datasets/utterance_dataset_5sec'
    
    # Process each session
    sessions = sorted([f for f in os.listdir(input_base_path) if not f.startswith('.')])
    for session in sessions:
        session_path = os.path.join(input_base_path, session)
        process_session(session_path, module)