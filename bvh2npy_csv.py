import numpy as np
import os
import glob
from bvh import Bvh
import torch
from tqdm import tqdm
from jinja2.nodes import Break
import pandas as pd

from convofusion.data.beat_dnd.utils.motion_rep_utils import forward_kinematics_euler


def bvh_to_joint_positions(bvh_file):
    with open(bvh_file) as f:
        mocap = Bvh(f.read())

    # Extract joint offsets and create kinematic tree
    joints = {}
    kinematic_tree = []
    current_chain = []

    def process_joint(joint_name, parent_idx=-1):

        joint_idx = len(joints)
        joints[joint_name] = {
            'offset': np.array(mocap.joint_offset(joint_name)),
            'channels': mocap.joint_channels(joint_name),
            'index': joint_idx
        }

        if parent_idx >= 0:
            while len(kinematic_tree) <= parent_idx:
                kinematic_tree.append([])
            if not kinematic_tree[parent_idx]:
                kinematic_tree[parent_idx] = [parent_idx]
            kinematic_tree[parent_idx].append(joint_idx)

        for child in mocap.get_joint(joint_name).children[2:]:
            if(str(child)=="End Site"):
                break
            process_joint(str(child).split()[1], joint_idx)

    process_joint(str(mocap.get_joints()[0]).split()[1])

    # Extract motion data
    n_frames = mocap.nframes
    joint_rotations = np.zeros((n_frames, len(joints), 3))
    root_positions = np.zeros((n_frames, 3))

    for frame in tqdm(range(n_frames)):
        channels = mocap.frame_joint_channels(frame, str(mocap.get_joints()[0]).split()[1],mocap.joint_channels(str(mocap.get_joints()[0]).split()[1]))
        root_positions[frame] = channels[:3]
        joint_rotations[frame, 0] = channels[3:6]

        for joint_name in list(joints.keys())[1:]:
            channels = mocap.frame_joint_channels(frame,joint_name, mocap.joint_channels(joint_name))
            if(len(channels)==3):
                joint_rotations[frame, joints[joint_name]['index']] = channels
            else:
                joint_rotations[frame, joints[joint_name]['index']] = channels[3:6]


    # Convert rotations to radians
    joint_rotations = np.deg2rad(joint_rotations)

    # Create offset tensor
    offset_tensor = np.zeros((len(joints), 3))
    for joint_name, joint_data in joints.items():
        offset_tensor[joint_data['index']] = joint_data['offset']

    # Convert to torch tensors
    joint_rotations_torch = torch.from_numpy(joint_rotations).float()
    root_positions_torch = torch.from_numpy(root_positions).float()
    offset_tensor_torch = torch.from_numpy(offset_tensor).float()

    # Compute forward kinematics
    positions = forward_kinematics_euler(
        joint_rotations_torch,
        root_positions_torch,
        offset_tensor_torch,
        kinematic_tree
    )

    return positions.numpy()


def convert_to_dnd_format(positions):
    """
    Convert joint positions to DnD CSV format
    positions: numpy array of shape (n_frames, n_joints, 3)
    returns: pandas DataFrame in DnD format
    """
    n_frames, n_joints, _ = positions.shape

    # Create frame numbers
    frame_numbers = np.arange(n_frames)

    # Flatten joint positions for each frame
    # DnD format expects: frame_number, joint1_x, joint1_y, joint1_z, joint2_x, joint2_y, joint2_z, ...
    flattened_positions = positions.reshape(n_frames, -1)

    # Create DataFrame
    data = np.column_stack([frame_numbers, flattened_positions])
    df = pd.DataFrame(data)

    return df

def convert_dataset(input_dir, output_dir, output_format):
    os.makedirs(output_dir, exist_ok=True)

    for bvh_file in glob.glob(os.path.join(input_dir, '**/*.bvh'), recursive=True):
        relative_path = os.path.relpath(bvh_file, input_dir)
        output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.'+output_format)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            print(f"Skipped {bvh_file} -> {output_path} (already processed)")
            continue

        joint_positions = bvh_to_joint_positions(bvh_file)
        if output_format=="npy":
            np.save(output_path, joint_positions)
        elif output_format=="csv":
            convert_to_dnd_format(joint_positions)
            df = convert_to_dnd_format(joint_positions)
            df.to_csv(output_path, index=False, header=False)
        print(f"Converted {bvh_file} -> {output_path}")


if __name__ == "__main__":
    #input_dir = "datasets/beat_english_v0.2.1/"
    #output_dir = "datasets/beat_english_v0.2.1_processed"
    #output_format="npy"
    input_dir = "datasets/aug_ut_data_5sec"
    output_dir = "datasets/aug_ut_data_5sec_processed"
    output_format="csv"
    convert_dataset(input_dir, output_dir, output_format)