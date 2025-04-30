import zarr
import numpy as np
import os
import imagecodecs
from PIL import Image
import cv2
from tqdm import tqdm
# 加载 Zarr 数据（以读写模式）
zarr_path = "/home/cowa/universal_manipulation_interface/dataset/0328_coffee.zarr"
store = zarr.open(zarr_path, mode='r+')

data_group = store["data"]
state_group = data_group["ee_pos"]
gripper_group = data_group['gripper_pos']
# velocity_group = data_group["joints_vel"]
episodes_end = store["meta/episode_ends"][:]
camera_img_folder = os.path.join(zarr_path, 'data', 'camera0_rgb')

# 查看数组元数据
print("Shape:", state_group.shape)  # 打印数组形状
print("Chunks:", state_group.chunks)  # 打印分块信息

# 创建 `marker` 数据集
marker_shape = (state_group.shape[0],)  # (79761,)
marker_chunks = (state_group.chunks[0],)  # (1480,)

create_velocity_shape = (state_group.shape[0],3)
create_velocity_chunks = (state_group.chunks[0],3)


# robot = robot_solver(render=False)
if "memory_marker" not in data_group:
    marker = data_group.create_dataset(
        "memory_marker", shape=marker_shape, chunks=marker_chunks, dtype='int32', fill_value=0
    )
else:
    marker = data_group["memory_marker"]
    

# if "EE_velocity" not in data_group:
#     ee_velocity = data_group.create_dataset(
#         "EE_velocity", shape=create_velocity_shape, chunks=create_velocity_chunks, dtype='float32', fill_value=0
#     )
#     # 用 tqdm 包装外层循环，显示 episode 进度
#     for i in tqdm(range(len(episodes_end)), desc="Episodes"):
#         start_idx = 0 if i == 0 else episodes_end[i - 1] + 1
#         end_idx = episodes_end[i]

#         state_data = state_group[start_idx:end_idx]
#         # velocity_data = velocity_group[start_idx:end_idx]
#         # 用 tqdm 包装内层循环，显示每个 episode 内的进度
#         for j in tqdm(range(len(state_data)), desc=f"Episode {i} processing", leave=False):
#             EE_6_vel = robot.solve_fk_velocity(state_data[j][1:], velocity_data[j][1:])
#             index = start_idx + j
#             ee_velocity[index] = np.squeeze(EE_6_vel[:3])
#             # print(np.linalg.norm(np.squeeze(EE_6_vel[:3])))

# else:
#     ee_velocity = data_group["EE_velocity"]

# print("finish process velocity")

# if "ave_velocity" not in data_group:
#     ave_velocity = data_group.create_dataset(
#         "ave_velocity", shape=create_velocity_shape, chunks=create_velocity_chunks, dtype='float32', fill_value=0
#     )
#     for i in tqdm(range(len(episodes_end)), desc="Episodes"):
#         start_idx = 0 if i == 0 else episodes_end[i - 1] + 1
#         end_idx = episodes_end[i]

#         ee_velocity_data = ee_velocity[start_idx:end_idx]

#         for j in tqdm(range(len(ee_velocity_data)), desc=f"Episode {i} processing", leave=False):
#             window_start = max(0, j - 29)
#             window_data = ee_velocity_data[window_start:j+1]
#             avg_vector = np.mean(window_data, axis=0)
#             index = start_idx + j
#             ave_velocity[index] = avg_vector

# else:
#     ave_velocity = data_group["ave_velocity"]

# print("finish process average velocity")


# OpenCV 窗口初始化
cv2.namedWindow("Key Frame", cv2.WINDOW_NORMAL)

# 遍历 episodes_end 并更新 marker
for i in tqdm(range(len(episodes_end)), desc="Episodes"):
    start_idx = 0 if i == 0 else episodes_end[i - 1] + 1
    end_idx = episodes_end[i]

    state_data = state_group[start_idx:end_idx]
    gripper_data = gripper_group[start_idx:end_idx]
    traj_len = len(state_data)

    for j in tqdm(range(traj_len - 1), desc=f"Episode {i} processing", leave=False):

        # 从打开到关闭：标记后面10帧
        if gripper_data[j] >= 88 and gripper_data[j + 1] < 88:
            for num in range(10):
                index = start_idx + j + 1 + num  # 标记后续帧
                if 0 <= index < marker.shape[0]:
                    chunk_file = os.path.join(camera_img_folder, f"{index}.0.0.0")
                    if not os.path.exists(chunk_file):
                        print(f"Chunk file not found: {chunk_file}")
                        continue

                    with open(chunk_file, 'rb') as f:
                        encoded_img = f.read()
                        try:
                            decoded_img = imagecodecs.jpegxl_decode(encoded_img)
                            if decoded_img is None:
                                print(f"Failed to decode image at frame {index}: Decoding returned None")
                                continue

                            img = np.array(Image.fromarray(decoded_img))
                            cv2.imshow("Key Frame", img)
                            cv2.waitKey(20)  # 刷新窗口
                        except Exception as e:
                            print(f"Failed to decode image at frame {index}: {e}")
                            continue

                    print("Marker index is", index)
                    marker[index] = 1

        # 从关闭到打开：标记前面10帧
        elif gripper_data[j] <= 88 and gripper_data[j + 1] > 88:
            for num in range(10):
                index = start_idx + j - num  # 标记前面帧
                if 0 <= index < marker.shape[0]:
                    chunk_file = os.path.join(camera_img_folder, f"{index}.0.0.0")
                    if not os.path.exists(chunk_file):
                        print(f"Chunk file not found: {chunk_file}")
                        continue

                    with open(chunk_file, 'rb') as f:
                        encoded_img = f.read()
                        try:
                            decoded_img = imagecodecs.jpegxl_decode(encoded_img)
                            if decoded_img is None:
                                print(f"Failed to decode image at frame {index}: Decoding returned None")
                                continue

                            img = np.array(Image.fromarray(decoded_img))
                            cv2.imshow("Key Frame", img)
                            cv2.waitKey(20)  # 刷新窗口
                        except Exception as e:
                            print(f"Failed to decode image at frame {index}: {e}")
                            continue

                    print("Marker index is", index)
                    marker[index] = 1

print("Marker 数据更新完成")
cv2.destroyAllWindows()
