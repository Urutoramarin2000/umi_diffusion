import os
import time
import cv2
import dill
import hydra
import numpy as np
import torch
import matplotlib.pyplot as plts
import zarr
import imagecodecs
from omegaconf import OmegaConf
from torch import nn
from PIL import Image
OmegaConf.register_new_resolver("eval", eval, replace=True)

from umi.real_world.real_inference_util import get_real_umi_action

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from cowautil.joint2eepos import joints2eepos
from cowautil.IK_solver_pybullet import robot_solver
from memory_classifier.KeyframeClassifier import KeyframeClassifier_small_lowdim, KeyframeClassifier_big_lowdim, KeyframeClassifier_no_lowdim, KeyframeClassifier_resnet

from infer_utils import *
def eval_on_dataset(config):
    device = config['device']
    diffusion_ckpt_path = config['diffusion_ckpt_path']
    dataset_path = config['dataset_path']
    classifier_activated = config['classifier_activated']
    classifier_ckpt_path = config['classifier_ckpt_path']
    classifier_dataparallel = config['classifier_dataparallel']
    classifier_type = config['classifier_type']
    classifier_gapped_history_count = config['classifier_gapped_history_count']
    
    predicted_lines = []
    current_lines = []

    Memory_size = 0
    Memory_limit = 2
    key_frame_count = 0
    memory_array = np.zeros((2, 3, 224, 224), dtype=np.uint8)
    history_count = classifier_gapped_history_count

    diff_ik_solver = robot_solver(render=False)
    # velocity_solver = robot_solver(render=False)

    fig, axs = plts.subplots(3, 1, figsize=(8, 12))
    titles = ["X Coordinate", "Y Coordinate", "Z Coordinate"]
    y_labels = ["X Position", "Y Position", "Z Position"]
    colors = ['r', 'g', 'b']  # Red for X, Green for Y, Blue for Z
    predicted_lines = []
    current_lines = []
    for i in range(3):
        axs[i].set_title(titles[i])
        axs[i].set_ylabel(y_labels[i])
        axs[i].set_xlabel("Prediction Step")
        axs[i].grid(True)
        pred_line, = axs[i].plot([], [], f'{colors[i]}o-', label="Predicted")  # Circle marker for predicted
        curr_line, = axs[i].plot([], [], f'{colors[i]}x-', label="Current")  # 'X' marker for current
        predicted_lines.append(pred_line)
        current_lines.append(curr_line)
    axs[0].legend()

    if classifier_activated:
        if classifier_type == 'vit_big_lowdim':
            classifier_model = KeyframeClassifier_big_lowdim().to(device) 
        elif classifier_type == 'vit_small_lowdim':
            classifier_model = KeyframeClassifier_small_lowdim().to(device)
        elif classifier_type == 'resnet':
            classifier_model = KeyframeClassifier_resnet().to(device)
        else: 
            classifier_model = KeyframeClassifier_no_lowdim().to(device)
        
        if classifier_dataparallel:
            classifier_model = nn.DataParallel(classifier_model)
        classifier_ckpt = torch.load(classifier_ckpt_path, map_location=device)
        classifier_model.load_state_dict(classifier_ckpt['model_state_dict'])
        classifier_model.eval
    
    ckpt_path = diffusion_ckpt_path
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = 16 # DDIM inference iterations
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr

    policy.eval().to(device)
    policy.reset()
    episode_start_pose = list()
    
    img_obs_horizon = cfg.task.img_obs_horizon
    index = 0

    with zarr.open(dataset_path, mode='r') as dataset:
        camera_img_folder = os.path.join(dataset_path, 'data', 'camera0_rgb')
        meta_group = dataset['meta']
        using_transformer = config['using_transformer']
        episode_ends = meta_group['episode_ends'][:]
        data_group = dataset['data']
        data_eef_pos = data_group["robot0_eef_pos"][:]
        data_eef_rot_axis_angle = data_group["robot0_eef_rot_axis_angle"][:]
        data_gripper_width = data_group["robot0_gripper_width"][:]
        data_start_pose = data_group["robot0_demo_start_pose"][:]
        current_episode_end = episode_ends[index]
        episode_start_pose.append(data_start_pose[current_episode_end-1])

        memory_dict = np.zeros((2, 3, 224, 224), dtype=np.uint8)
        memory_array = np.zeros((2, 3, 224, 224), dtype=np.uint8)
        data_camera_imgs = []
        gap = img_obs_horizon*3
        obs = {
            "robot0_eef_pos": [],
            "robot0_eef_rot_axis_angle": [],
            "robot0_gripper_width": [],
            "camera0_rgb": []
        }

        if index == 0:
            start = 0
            img_data_start = 0
        else:
            start = episode_ends[index-1] + img_obs_horizon
            img_data_start = episode_ends[index-1]
        for t in range(img_data_start, current_episode_end):
            chunk_file = os.path.join(camera_img_folder, f"{t}.0.0.0")
            if not os.path.exists(chunk_file):
                print(f"Chunk file not found: {chunk_file}")
                continue
            with open(chunk_file, 'rb') as f:
                encoded_img = f.read()
                try:
                    decoded_img = imagecodecs.jpegxl_decode(encoded_img)
                    img = Image.fromarray(decoded_img)
                    data_camera_imgs.append(np.array(img))
                except Exception as e:
                    print(f"Failed to decode image at frame {t}: {e}")

        start_xyz = episode_start_pose[0][:3]
        start_rotation = episode_start_pose[0][3:]
        q = diff_ik_solver.solve_ik(start_xyz, start_rotation)
        print("move to start pose complete")
        
        for i in range(start, current_episode_end-16, 3):
            obs["robot0_eef_pos"] = np.array([data_eef_pos[i], data_eef_pos[i+gap]])  # Shape (T, 3)
            obs["robot0_eef_rot_axis_angle"] = np.array([data_eef_rot_axis_angle[i], data_eef_rot_axis_angle[i+gap]])  # Shape (T, 3)
            obs["robot0_gripper_width"] = np.array([data_gripper_width[i], data_gripper_width[i+gap]]).reshape(-1, 1)  # Shape (T, 1)
            obs["camera0_rgb"] = np.array([data_camera_imgs[i - start], data_camera_imgs[i - start+gap]])  # Shape (T, W, H, C)
            
            image_to_show = data_camera_imgs[i - start][:, :, ::-1]
            image_to_show = (image_to_show ).astype(np.uint8)  # 转换为 uint8 类型
            cv2.imshow("image", image_to_show)
            cv2.waitKey(1) 

            with torch.no_grad():
                if classifier_activated:
                    print("history_count",history_count)
                    if history_count < 0:
                        current_gripper_pos = obs["robot0_gripper_width"][-1]
                        classifier_img_input = np.array(obs["camera0_rgb"][-1]).copy() / 255.0  # (H, W, C)
                        classifier_img_input = np.transpose(classifier_img_input, (2, 0, 1))  # (C, H, W)
                        classifier_img_input = np.expand_dims(classifier_img_input, axis=0)   # (1, C, H, W)
                        classifier_img_input = torch.from_numpy(classifier_img_input).float().to(device)
                        
                        low_dim_tensor = torch.from_numpy(current_gripper_pos).float().view(1, 1).to(device)
                        logits = classifier_model(classifier_img_input.float().to(device), low_dim_tensor)
                        print("current compare",logits)
                        predictions = torch.argmax(logits, dim=1)

                        if predictions.item() == 1:
                            print("find key frame!!!!!!!!!!!!!!!!!!!!!!!!!")
                            cam_data = np.array(obs["camera0_rgb"][-1])[:, :, ::-1]
                            if cam_data.dtype != np.uint8:
                                cam_data_to_save = (cam_data * 255).astype(np.uint8)
                            else:
                                cam_data_to_save = cam_data
                            img = Image.fromarray(cam_data_to_save)
                            img.save(f'cam_data_{key_frame_count}.png')
                            key_frame_count += 1
                            if Memory_size >= Memory_limit:
                                memory_array = np.roll(memory_array, shift=-1, axis=0)
                                memory_array[-1] = np.transpose(cam_data, (2, 0, 1)) / 255.0
                            else:
                                memory_array[Memory_size] = np.transpose(cam_data, (2, 0, 1)) / 255.0
                                Memory_size += 1
                            history_count = classifier_gapped_history_count
                    else:
                        history_count = history_count - 1
                        
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs,
                    shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=obs_pose_rep,
                    episode_start_pose=episode_start_pose
                )
                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
                )
                mem_dict = {'memory_marker': memory_dict}
                mem_dict = dict_apply(
                    mem_dict,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
                )
                memory_obs = {
                    'obs': obs_dict, 
                    'memory': mem_dict,
                }
                
                if classifier_activated:
                    if using_transformer:
                        result = policy.predict_action(memory_obs)
                    else:
                        result = policy.predict_action(memory_obs, None, True)
                else:
                    result = policy.predict_action(obs_dict)

                raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                action = get_real_umi_action(raw_action, obs, action_pose_repr)
                assert action.shape[-1] == 7
                predicted_xyz = []
                exec_time = time.time()
                for j in range(0,16):
                    current_action_gripper = action[j][6]
                    current_action_pos = action[j][:6]
                    xyz = current_action_pos[:3]
                    rx, ry, rz = current_action_pos[3:]
                    q = diff_ik_solver.solve_ik(xyz, [rx, ry, rz])
                    cmd = np.zeros(7)
                    cmd[0] = current_action_gripper * 100
                    print(current_action_gripper)
                    cmd[1:] = q
                    predicted_xyz.append(xyz)
                    # print("Predicted action gripper:", cmd[0])
                    # cmd_pub.send('arm_action', msg={'action': cmd.tolist()})
                    time.sleep(0.05)
                print('Action execution time:', time.time() - exec_time)
                predicted_xyz = np.array(predicted_xyz)
                for dim in range(3):  
                    predicted_lines[dim].set_data(range(16), predicted_xyz[:, dim])
                    current_lines[dim].set_data(range(16), data_eef_pos[i:i+16, dim])  

                    axs[dim].set_xlim(0, 16)  
                    min_val = np.min([np.min(predicted_xyz[:, dim]), np.min(data_eef_pos[i:i+16, dim])]) - 0.05
                    max_val = np.max([np.max(predicted_xyz[:, dim]), np.max(data_eef_pos[i:i+16, dim])]) + 0.05
                    axs[dim].set_ylim(min_val, max_val)  
                   
                plts.tight_layout()  
                plts.pause(0.01) 
                predicted_xyz = np.array(predicted_xyz)  
                current_xyz = np.array(data_eef_pos[i:i+16])  
                squared_differences = (predicted_xyz - current_xyz) ** 2
                mse_xyz = np.mean(squared_differences)
                rmse_xyz = np.sqrt(mse_xyz)
                a = 1
    
