import torch
from torch.utils.data import Dataset
import zarr
import numpy as np
import os
import imagecodecs
from PIL import Image
# from ..cowautil.IK_solver_pybullet import robot_solver
from torchvision import transforms
class MemoryData(Dataset):
    def __init__(self, zarr_path, train, oversample_factor=10):
        self.root_dir = zarr_path
        # self.robot_solver = robot_solver(render=False)
        store = zarr.open(zarr_path, mode='r')
        data_group = store["data"]
        # pos_group = data_group["joints_pos"]
        # vel_group = data_group["joints_vel"]
        gripper_group = data_group["robot0_gripper_width"]
        label_group = data_group["memory_marker"]
        episodes_end = store["meta/episode_ends"][:]
        camera_img_folder = os.path.join(zarr_path, 'data', 'camera0_rgb')

        self.obs = []
        self.label = []

        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        
        if train:
            indices = range(episodes_end[-5])  # 训练数据
        else:
            indices = range(episodes_end[-5], label_group.shape[0])  # 测试数据

        temp_obs = []
        temp_label = []

        # 读取数据
        for index in indices:
        # for index in [1,2]:
            # print("processing {}".format(index))
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

                    # 转换为 PIL 图像，再转换为 numpy
                    img = np.array(Image.fromarray(decoded_img))
                    low_dim = np.array(gripper_group[index])
                    low_dim[:,] = 0 if low_dim > 0.65 else 1
                    label = label_group[index]
                    temp_obs.append({"image": img, "low_dim": low_dim})
                    temp_label.append(label)

                except Exception as e:
                    print(f"Failed to decode image at frame {index}: {e}")
                    continue
        if train and oversample_factor != 0:
            positive_indices = [i for i, lable in enumerate(temp_label) if label == 1]
            for index in positive_indices:
                for _ in range(oversample_factor - 1):
                    self.obs.append(temp_obs[index])
                    self.label.append(temp_label[index])

        self.obs.extend(temp_obs)
        self.label.extend(temp_label)

        print("process complete")

    def __getitem__(self, index):
        img = self.obs[index]["image"]
        low_dim = self.obs[index]["low_dim"]
        label = self.label[index]

        # img = self.transform(img) 
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0  # 归一化到 0~1

        low_dim = torch.tensor(low_dim, dtype=torch.float32)  
        label = torch.tensor(label, dtype=torch.long)  # True/False 分类

        return img, low_dim, label

    def __len__(self):
        return len(self.obs)
