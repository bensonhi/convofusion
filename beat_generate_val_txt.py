import os
import random

def list_npy_files_in_directory(directory, val_file, train_file):
    try:
        val_files = []
        train_files = []

        # Traverse through all subdirectories
        for p in os.listdir(directory):
            folder = os.path.join(directory, p)
            if os.path.isdir(folder):  # Check if it's a folder
                for file in os.listdir(folder):
                    filepath = os.path.join(folder, file)
                    if filepath.endswith('.npy'):  # Check if it's an .npy file
                        # Randomly assign to val or train
                        path = os.path.basename(os.path.join(p, file)).replace('.npy', '')
                        if random.random() > 0.8:
                            val_files.append(path)
                        else:
                            train_files.append(path)

        # Write validation file paths to val_file
        with open(val_file, 'w') as file:
            for npy_file in val_files:
                file.write(npy_file + '\n')

        # Write training file paths to train_file
        with open(train_file, 'w') as file:
            for npy_file in train_files:
                file.write(npy_file + '\n')

        print(f"Validation files written to {val_file}")
        print(f"Training files written to {train_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Specify the directory and output files
directory = '/home/bwang/ar85/ben/convofusion/datasets/beat_english_v0.2.1_processed/'
val_file = '/home/bwang/ar85/ben/convofusion/datasets/beat_english_v0.2.1_processed/val.txt'
train_file = '/home/bwang/ar85/ben/convofusion/datasetsbeat_english_v0.2.1_processed/train.txt'

# Call the function
list_npy_files_in_directory(directory, val_file, train_file)