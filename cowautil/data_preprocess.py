import zarr
import numpy as np
import argparse
from tqdm import tqdm
from IK_solver_pybullet import robot_solver
from scipy.spatial.transform import Rotation as R

def data_process(args, robot_calculator: robot_solver):
    with zarr.open(args.dataset_path, mode='a') as dataset:
        all_ee_pos = []
        print(dataset['/data/joints_pos'].info)

        data_group = dataset['data']
        joints_pos_array = data_group['joints_pos'][:]
        gripper_states = joints_pos_array[:, 0]  # 夹爪
        joints_pos_array = joints_pos_array[:, 1:]  # 去掉夹爪

        # 读取 episode_ends
        meta_group = dataset['meta']
        episode_ends = meta_group['episode_ends'][:]
        total_timesteps = len(joints_pos_array)

        # 准备存储位姿数据
        start = 0
        demo_start_pose = []
        demo_end_pose = []
        eepos = []
        ee_rot = []
        gripper = []

        for idx, end in enumerate(tqdm(episode_ends, desc="Processing episodes")):
            print(f"Processing episode {idx} (frames {start} to {end})...")
            episode_joints = joints_pos_array[start:end]
            episode_gripper = gripper_states[start:end]

            # 计算每帧的末端执行器位姿
            episode_eepos = []
            episode_ee_rot = []
            for joint_angles in episode_joints:
                ee_pos = robot_calculator.solve_fk_homogeneous(joint_angles[:])
                xyz = ee_pos[:3, 3]
                rotation = R.from_matrix(ee_pos[:3, :3])
                rot_vec = rotation.as_rotvec()
                episode_eepos.append(xyz)
                episode_ee_rot.append(rot_vec)

            # 保存当前 episode 的数据
            start_pose = np.hstack((episode_eepos[0], episode_ee_rot[0]))
            end_pose = np.hstack((episode_eepos[-1], episode_ee_rot[-1]))

            demo_start_pose.extend([start_pose] * len(episode_joints))
            demo_end_pose.extend([end_pose] * len(episode_joints))
            eepos.extend(episode_eepos)
            ee_rot.extend(episode_ee_rot)
            gripper.extend(episode_gripper)

            start = end

        # 更新 Zarr 数据集
        chunks = dataset['/data/joints_pos'].chunks
        print(chunks)
        # 确保 chunks 的维度与数据形状一致
        gripper_chunks = (chunks[0],1)  # 如果 chunks 是多维的，只取第一个维度
        eef_pos_chunks = (chunks[0], 3)  # 对于 eepos 和 ee_rot，形状为 (timesteps, 3)
        eef_rot_chunks = (chunks[0], 3)
        demo_pose_chunks = (chunks[0], 6)  # 对于 demo_start_pose 和 demo_end_pose，形状为 (timesteps, 6)

        gripper_array = np.array(gripper).reshape(-1, 1) 
        # 更新 Zarr 数据集
        data_group.create_dataset('robot0_demo_start_pose', data=np.array(demo_start_pose), chunks=demo_pose_chunks)
        data_group.create_dataset('robot0_demo_end_pose', data=np.array(demo_end_pose), chunks=demo_pose_chunks)
        data_group.create_dataset('robot0_eef_pos', data=np.array(eepos), chunks=eef_pos_chunks)
        data_group.create_dataset('robot0_eef_rot_axis_angle', data=np.array(ee_rot), chunks=eef_rot_chunks)
        data_group.create_dataset('robot0_gripper_width', data=np.array(gripper_array), chunks=gripper_chunks)

        eef_rot_chunks = dataset['/data/robot0_gripper_width'].chunks
        print(eef_rot_chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--urdf_path', '-u', type=str,
    #                      help='Path to the urdf', default='../assets/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf')
    parser.add_argument('--dataset_path', '-d', type=str,
                        help='Path to the dataset', default='./0331_nofu.zarr')
    args = parser.parse_args()
    robot_calculator = robot_solver()
    data_process(args, robot_calculator)