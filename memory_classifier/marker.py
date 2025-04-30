import zarr
import numpy as np
import os
import imagecodecs
from PIL import Image
import cv2
from ..cowautil.IK_solver_pybullet import robot_solver
from tqdm import tqdm
zarr_path = "/home/cowa/CoRL2025/training_data/0328_driver.zarr"
store = zarr.open(zarr_path, mode='r+')

data_group = store["data"]
state_group = data_group["ee_pos"]
gripper_group = data_group['gripper_pos']
# velocity_group = data_group["joints_vel"]
episodes_end = store["meta/episode_ends"][:]
camera_img_folder = os.path.join(zarr_path, 'data', 'camera0_rgb')

print("Shape:", state_group.shape)  
print("Chunks:", state_group.chunks)  

marker_shape = (state_group.shape[0],)  # (79761,)
marker_chunks = (state_group.chunks[0],)  # (1480,)

create_velocity_shape = (state_group.shape[0],3)
create_velocity_chunks = (state_group.chunks[0],3)


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

epsilon = 0.5  # 容差阈值，定义“几乎相等”的标准

for i in tqdm(range(len(episodes_end)), desc="Episodes"):
    start_idx = 0 if i == 0 else episodes_end[i - 1] + 1
    end_idx = episodes_end[i]

    state_data = state_group[start_idx:end_idx]
    gripper_data = gripper_group[start_idx:end_idx]
    traj_len = len(state_data)

    j = 0
    cooldown = False

    while j < traj_len - 11:  # 确保 j+10 不越界
        if not cooldown:
            # 将严格等号判断改为平稳段判断
            if abs(gripper_data[j] - gripper_data[j + 1]) < epsilon:
                cooldown = True
                num = 0
                for num in range(10):
                    index = start_idx + j + num
                    if index >= marker.shape[0]:
                        break
                    # 同样替换这里的判断
                    if abs(gripper_data[j] - gripper_data[j + num]) < epsilon:
                        chunk_file = os.path.join(camera_img_folder, f"{index}.0.0.0")
                        if not os.path.exists(chunk_file):
                            tqdm.write(f"Chunk file not found: {chunk_file}")
                            continue

                        with open(chunk_file, 'rb') as f:
                            encoded_img = f.read()
                            try:
                                decoded_img = imagecodecs.jpegxl_decode(encoded_img)
                                if decoded_img is None:
                                    tqdm.write(f"Failed to decode image at frame {index}: Decoding returned None")
                                    continue

                                img = np.array(Image.fromarray(decoded_img))
                                cv2.imshow("Key Frame", img)
                                cv2.waitKey(20)
                            except Exception as e:
                                tqdm.write(f"Failed to decode image at frame {index}: {e}")
                                continue

                        tqdm.write(f"Marker index is {index}")
                        marker[index] = 1
                    else:
                        tqdm.write(f"Break marking at frame {index} due to value change.")
                        break

                j += num
                continue  # 跳过 cooldown 检查

        # cooldown 退出条件：观察到明显变化
        if (abs(gripper_data[j] - gripper_data[j + 3]) > epsilon and
            abs(gripper_data[j] - gripper_data[j + 5]) > epsilon):
            cooldown = False

        j += 1
            # 从关闭到打开：标记前面10帧
            # elif gripper_data[j] <= 88 and gripper_data[j + 1] > 88:
            #     for num in range(10):
            #         index = start_idx + j - num  # 标记前面帧
            #         if 0 <= index < marker.shape[0    ]:
            #             chunk_file = os.path.join(camera_img_folder, f"{index}.0.0.0")
            #             if not os.path.exists(chunk_file):
            #                 print(f"Chunk file not found: {chunk_file}")
            #                 continue

            #             with open(chunk_file, 'rb') as f:
            #                 encoded_img = f.read()
            #                 try:
            #                     decoded_img = imagecodecs.jpegxl_decode(encoded_img)
            #                     if decoded_img is None:
            #                         print(f"Failed to decode image at frame {index}: Decoding returned None")
            #                         continue

            #                     img = np.array(Image.fromarray(decoded_img))
            #                     cv2.imshow("Key Frame", img)
            #                     cv2.waitKey(20)  # 刷新窗口
            #                 except Exception as e:
            #                     print(f"Failed to decode image at frame {index}: {e}")
            #                     continue

            #             print("Marker index is", index)
            #             marker[index] = 1
            

print("Marker finished")
cv2.destroyAllWindows()
