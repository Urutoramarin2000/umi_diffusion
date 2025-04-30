import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# # 加载 Zarr 数据
zarr_path = "/home/cowa/universal_manipulation_interface/dataset/0417_driver_box.zarr"
# action_group = zarr.open(os.path.join(zarr_path, "data", "action"), mode='r')
state_group = zarr.open(os.path.join(zarr_path, "data", "robot0_gripper_width"), mode='r')
episodes_end = zarr.open(os.path.join(zarr_path, "meta", "episode_ends"), mode='r')
print("Data:", state_group[0:600])
# 查看数组元数据
print("Shape:", state_group.shape)  # 打印数组形状
print("Chunks:", state_group.chunks)  # 打印分块信息
print("Example Data (First 10 Samples):", len(episodes_end))  # 查看前 10 个数据

# 提取整个数组
state_data  = state_group[:episodes_end[0]]
print("robot_eef_pose:", state_data.shape)

#mutiple axis

# # 假设 all_data 是完整的动作数据，形状为 (T, n_dims)
# state_dims = state_data.shape[-1]

# # 创建子图：每个维度一个子图
# fig, axes = plt.subplots(state_dims, 1, figsize=(10, 5 * state_dims), sharex=True)
# fig.suptitle("Action Data by Dimension", fontsize=16)

# for dim in range(state_dims):
#     axes[dim].plot(state_data[:, dim], label=f"Dimension of ee pose{dim}")
#     axes[dim].set_ylabel("Action Value")
#     axes[dim].legend()
#     axes[dim].grid(True)

# # 统一设置 X 轴
# axes[-1].set_xlabel("Time Step")

# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局，避免标题与子图重叠
# plt.show()

# # 加载 Zarr 数据
# zarr_path = "/home/cowa/universal_manipulation_interface/example_demo_session/0217demo.zarr"
# state_group = zarr.open(os.path.join(zarr_path, "data", "robot0_eef_pos"), mode='r')
# episodes_end = zarr.open(os.path.join(zarr_path, "meta", "episode_ends"), mode='r')

# # 提取数据
# state_data = state_group[:episodes_end[0]]

# # 提取三维坐标
# x = state_data[:, 0]
# y = state_data[:, 1]  # 👈 反转 Y 轴方向
# z = state_data[:, 2]

# # 绘制 3D 轨迹图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z, label='EEF 3D Trajectory', color='b')

# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # 设置观察角度（elev=0 表示水平看，azim=-90 表示从 Z 负方向看）
# # ax.view_init(elev=0, azim=180)

# # 图例与标题
# ax.set_title('3D Trajectory of End Effector (Z into screen, Y down)')
# ax.legend()
# plt.show()

plt.figure(figsize=(10, 4))
plt.plot(state_data, label="Gripper Width")
plt.xlabel("Time Step")
plt.ylabel("Gripper Width")
plt.title("Gripper Width vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()