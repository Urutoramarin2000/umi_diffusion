import zarr
import numpy as np
import argparse
from tqdm import tqdm
from IK_solver_pybullet import robot_solver
from scipy.spatial.transform import Rotation as R

def data_process(args):
    with zarr.open(args.dataset_path, mode='a') as dataset:
        all_ee_pos = []
        print(dataset['/data/ee_pos'].info)
        error_flag = False
        data_group = dataset['data']
        pos_array = data_group['ee_pos'][:]
        rot_array = data_group['ee_rot'][:]
        gripper_states = data_group['gripper_pos'][:]


        # 读取 episode_ends
        meta_group = dataset['meta']
        episode_ends = meta_group['episode_ends'][:]

        # 夹爪在T265坐标系下
        gripper_in_T265_translation = np.array([0.0, -0.1, -0.1])

        # 将 T265 坐标轴变换到夹爪坐标轴的旋转矩阵（R_gripper_from_t265）
        R_mine = np.array([[0, -1 ,0], [0, 0, 1], [-1, 0, 0]])

        # 准备存储位姿数据
        start = 0
        demo_start_pose = []
        demo_end_pose = []
        ee_pos = []
        ee_rot = []
        gripper = []

        for idx, end in enumerate(tqdm(episode_ends, desc="Processing episodes")):
            print(f"Processing episode {idx} (frames {start} to {end})...")
            episode_pos = pos_array[start:end]
            episode_rot = rot_array[start:end]
            episode_gripper = gripper_states[start:end]

            episode_gripper = np.array(episode_gripper)
            normalized_gripper = episode_gripper / 100
            # for i in range(len(episode_gripper)):
            #     if episode_gripper[i] >= 65:
            #         normalized_gripper[i] = 1
            #     else:
            #         normalized_gripper[i] = 0

            episode_ee_pos = []
            episode_ee_rot = []

            for i in range(len(episode_pos)):
                # T265在世界坐标系下的位置
                t265_pos_world = np.array(episode_pos[i])  # shape: (3,)
                quat = episode_rot[i]  # [x, y, z, w]
                if np.isnan(quat).any():
                    error_flag = True
                    print("发现 NaN quaternion，跳过当前 episode。")
                    break
                print("Quat at current step:", quat)
                # 构造旋转对象
                t265_rot = R.from_quat(quat)
                R_t265 = t265_rot.as_matrix()  # T265在世界坐标下的旋转矩阵

                # === 夹爪的世界旋转 ===
                gripper_rot = R_t265 @ R_mine #（T265 → Gripper坐标变换）
                gripper_rot_matrix = gripper_rot 
                gripper_rotvec = R.from_matrix(gripper_rot_matrix).as_rotvec()

                # === 夹爪的世界位置 ===
                # gripper_pos = t265_pos - R_t265 @ offset
                gripper_pos_world = t265_pos_world + R_t265 @ gripper_in_T265_translation

                episode_ee_pos.append(gripper_pos_world)
                episode_ee_rot.append(gripper_rotvec)
            if error_flag:
                continue
            # 保存当前 episode 的数据
            start_pose = np.hstack((episode_ee_pos[0], episode_ee_rot[0]))
            end_pose = np.hstack((episode_ee_pos[-1], episode_ee_rot[-1]))

            demo_start_pose.extend([start_pose] * len(episode_pos))
            demo_end_pose.extend([end_pose] * len(episode_pos))
            ee_pos.extend(episode_ee_pos)
            ee_rot.extend(episode_ee_rot)
            gripper.extend(normalized_gripper)

            start = end

        # 更新 Zarr 数据集
        chunks = dataset['/data/ee_pos'].chunks
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
        data_group.create_dataset('robot0_eef_pos', data=np.array(ee_pos), chunks=eef_pos_chunks)
        data_group.create_dataset('robot0_eef_rot_axis_angle', data=np.array(ee_rot), chunks=eef_rot_chunks)
        data_group.create_dataset('robot0_gripper_width', data=np.array(gripper_array), chunks=gripper_chunks)

        eef_rot_chunks = dataset['/data/robot0_gripper_width'].chunks
        print(eef_rot_chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--urdf_path', '-u', type=str,
    #                      help='Path to the urdf', default='../assets/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf')
    parser.add_argument('--dataset_path', '-d', type=str,
                        help='Path to the dataset', default='./dataset/0418_driver_box.zarr')
    args = parser.parse_args()
    data_process(args)
