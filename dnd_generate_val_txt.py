import os
import random

def list_folders_in_directory(directory, val_file, train_file, test_file):
    try:
        val_folders = []
        train_folders = []
        test_folders =[]

        # Get all folders in the directory
        for p in os.listdir(directory):
            folder = os.path.join(directory, p)
            if os.path.isdir(folder):  # Check if it is a folder
                for file in os.listdir(folder):
                    subfolder = os.path.join(folder, file)
                    if os.path.isdir(subfolder):  # Check if it is a subfolder
                        variable = random.random()
                        if variable > 0.9:
                            val_folders.append(os.path.join(p, file))
                        elif variable > 0.8:
                            test_folders.append(os.path.join(p, file))
                        else:
                            train_folders.append(os.path.join(p, file))

        # Write validation folder names to the val_file
        with open(val_file, 'w') as file:
            for folder in val_folders:
                file.write(folder + '\n')

        with open(test_file, 'w') as file:
            for folder in test_folders:
                file.write(folder + '\n')

        # Write training folder names to the train_file
        with open(train_file, 'w') as file:
            for folder in train_folders:
                file.write(folder + '\n')

        print(f"Validation folders written to {val_file} {len(val_folders)}")
        print(f"Training folders written to {train_file} {len(train_folders)}")
        print(f"Training folders written to {test_file} {len(test_folders)}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Specify the directory and output files
directory = './datasets/utterance_dataset_5sec/'
val_file = './datasets/utterance_dataset_5sec/val.txt'
test_file = './datasets/utterance_dataset_5sec/test.txt'
train_file = './datasets/utterance_dataset_5sec/train.txt'

# Call the function
list_folders_in_directory(directory, val_file, train_file, test_file)