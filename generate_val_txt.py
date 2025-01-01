import os
import random

def list_folders_in_directory(directory, output_file):
    try:
        # Get all folders in the directory
        for p in os.listdir(directory):
            folder = os.path.join(directory, p)
            folders = [os.path.join(p, file) for file in os.listdir(folder) if ((os.path.isdir(os.path.join(folder, file)))  & (random.random()>0.8))]

        # Write folder names to the output file
        with open(output_file, 'w') as file:
            for folder in folders:
                file.write(folder + '\n')

        print(f"List of folders successfully written to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Specify the directory and output file
directory = '/home/bwang/ar85/ben/convofusion/datasets/dnd_processed_5sec/'
output_file = '/home/bwang/ar85/ben/convofusion/datasets/dnd_processed_5sec/val.txt'

# Call the function
list_folders_in_directory(directory, output_file)