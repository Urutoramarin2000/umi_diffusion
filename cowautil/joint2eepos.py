import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from typing import TYPE_CHECKING
from cowautil.IK_solver_pybullet import robot_solver

# if TYPE_CHECKING:
#     from IK_solver_pybullet import robot_solver

def joints2eepos(temp_obs, robot_solver: robot_solver):
    # urdf_path = './assets/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf'
    # wrist_name = 'r_ace'
    # arm_indices = [0, 1, 2, 3, 4, 5]
    # pin_control = PinocchioMotionControl(urdf_path, wrist_name, arm_indices)

    # Extract joint positions, velocities, and camera images from temp_obs
    joints_pos_array = temp_obs["joints_pos"]
    if joints_pos_array.ndim == 1 and joints_pos_array.shape[0] == 7:
        joints_pos_array = np.expand_dims(joints_pos_array, axis=0)
    # print(joints_pos_array.shape)
    joints_vel_array = temp_obs["joints_vel"]
    camera_imgs = temp_obs["camera_img"]
    # print(camera_imgs.shape)
    if camera_imgs.shape == (800, 1280, 3):
        camera_imgs = np.expand_dims(camera_imgs, axis=0)
    # print(camera_imgs.shape)

    gripper_states = joints_pos_array[:, 0]  # 夹爪
    joints_pos_array = joints_pos_array[:, 1:]  # 去掉夹爪

    # Prepare to store poses and other data
    demo_start_pose = []
    demo_end_pose = []
    eepos = []
    ee_rot = []
    gripper = []

    # Loop over each episode
    
    for idx in range(len(joints_pos_array)):
        # Extract joint angles and gripper states for the current frame
        timestamp_joints = joints_pos_array[idx]
        timestamp_gripper = gripper_states[idx]

        # Calculate the end-effector position and rotation for each frame
        timestamp_eepos = []
        timestamp_ee_rot = []
        timestamp_ee_gripper = []
        xyz, rotation = robot_solver.solve_fk(timestamp_joints[:])
        timestamp_eepos.append(xyz)
        timestamp_ee_rot.append(rotation)
        timestamp_ee_gripper.append(timestamp_gripper)

        eepos.extend(timestamp_eepos)
        ee_rot.extend(timestamp_ee_rot)
        gripper.extend(timestamp_ee_gripper)

    # Construct the return dictionary with the desired structure
    obs = {"robot0_eef_pos": list(), "robot0_eef_rot_axis_angle": list(), "robot0_gripper_width": list(), "camera0_rgb": list()}
    # for frame in range(len(joints_pos_array)):
    #     # Get robot's end-effector position and rotation
    #     robot_pose = np.array(list(zip(eepos[frame], ee_rot[frame])))

        
    #     # Add entries to the `obs` dictionary for each robot
    #     obs[f'robot0_eef_pos'].append(np.array([robot_pose[:, 0]]))
    #     obs[f'robot0_eef_rot_axis_angle'].append(np.array([robot_pose[:, 1]]))

    #     # Gripper width for this robot
    #     obs[f'robot0_gripper_width'].append(np.array(gripper[frame]))
        # resized_camera_data [frame] = cv2.resize(camera_imgs[frame], (224, 224))

    for pos, rot, grip_width, img in zip(eepos, ee_rot, gripper, camera_imgs):
        obs[f'robot0_eef_pos'].append(pos)
        obs[f'robot0_eef_rot_axis_angle'].append(rot)
        # if grip_width > 65:
        #     grip_width = 1
        # else:
        #     grip_width = 0
        grip_width = grip_width / 100
        obs[f'robot0_gripper_width'].append(grip_width)
        obs[f"camera0_rgb"].append(cv2.resize(img, (224, 224)))
        
    # print(resized_camera_data.shape)
    obs["robot0_eef_pos"] = np.array(obs["robot0_eef_pos"])  # (T, 3)
    obs["robot0_eef_rot_axis_angle"] = np.array(obs["robot0_eef_rot_axis_angle"])  # (T, 3)
    obs["robot0_gripper_width"] = np.array(obs["robot0_gripper_width"]).reshape(-1, 1)  # (T, 1)
    obs["camera0_rgb"] = np.array(obs["camera0_rgb"])  # (T, *)
    
    return obs
