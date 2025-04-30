"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data
"""
import os
import time
import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from multiprocessing import Process, Queue, Event
from PIL import Image
OmegaConf.register_new_resolver("eval", eval, replace=True)

from umi.real_world.real_inference_util import get_real_umi_action
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from cowautil.msg_communicator import MsgPublisher
from cowautil.joint2eepos import joints2eepos
from cowautil.IK_solver_pybullet import robot_solver
from infer_utils import *

from memory_classifier.KeyframeClassifier import KeyframeClassifier_small_lowdim, KeyframeClassifier_big_lowdim, KeyframeClassifier_no_lowdim, KeyframeClassifier_resnet
from infer_dataset_utils import eval_on_dataset

def load_image(index):
    path = f'image_example/cam_data_{index}.png'
    img = cv2.imread(path)
    if img is None:
        print(f"Image not found: {path}")
        return None
    img = cv2.resize(img, (224, 224))
    return img

def run(config):
    start_pose = config['start_pose']
    device = config['device']
    diffusion_ckpt_path = config['diffusion_ckpt_path']
    using_transformer = config['using_transformer']
    classifier_activated = config['classifier_activated']
    classifier_ckpt_path = config['classifier_ckpt_path']
    classifier_dataparallel = config['classifier_dataparallel']
    classifier_type = config['classifier_type']
    classifier_gapped_history_count = config['classifier_gapped_history_count']
    using_keyboard = config['using_keyboard']
    
    Q = Queue(maxsize=10000)
    terminate = Event()
    p1 = Process(target=arm_state_recv, args=(terminate, Q))
    p1.start()
    cmd_pub = MsgPublisher(port=22222)
    dt = 1/20
    
    Memory_size = 0
    Memory_limit = 2
    key_frame_count = 0
    memory_array = np.zeros((2, 3, 224, 224), dtype=np.uint8)
    history_count = classifier_gapped_history_count

    diff_ik_solver = robot_solver(render=False)
    # velocity_solver = robot_solver(render=False)

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

    obs_real_horizon = 2
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
    episode_start_pose = list(

    )
    
    joints_obs_horizon = cfg.task.low_dim_obs_horizon
    img_obs_horizon = cfg.task.img_obs_horizon
    first_activated = True
    
    while not terminate.is_set():
        if first_activated:
            camera_imgs = []
            joints_pos = []
            joints_vel = []
            while True:
                while Q.qsize() > 0:
                    try:
                        data = Q.get_nowait()
                        joints_pos.append(data['joints_pos'])
                        joints_vel.append(data['joints_vel'])
                        camera_imgs.append(data['camera_img'])
                    except:
                        pass
                if len(joints_pos) >= 1 and len(camera_imgs) >= 1:
                    break
            
            first_obs = {
                "joints_pos": np.stack(joints_pos[-1]),
                "joints_vel": np.stack(joints_vel[-1]),
                "camera_img": np.stack(camera_imgs[-1])
            }
            
            start_obs = joints2eepos(first_obs, diff_ik_solver)
            pose = np.concatenate([
                start_obs[f'robot0_eef_pos'],
                start_obs[f'robot0_eef_rot_axis_angle']
            ], axis=-1)[-1]
            episode_start_pose.append(pose)
            cmd_pub.send('arm_action', msg={'action': start_pose})
            time.sleep(5)
            first_activated = False

        # Get newest obs:
        camera_imgs = []
        joints_pos = []
        joints_vel = []
        obs_timestamps = []

        while True:
            while Q.qsize() > 0:
                try:
                    data = Q.get_nowait()
                    joints_pos.append(data['joints_pos'])
                    joints_vel.append(data['joints_vel'])
                    camera_imgs.append(data['camera_img'])
                    obs_timestamps.append(data['obs_timestamp'])
                except:
                    pass
            if len(joints_pos) >= obs_real_horizon*joints_obs_horizon+1 and len(camera_imgs) >= obs_real_horizon*img_obs_horizon+1:
                break
        
        temp_obs = {
            "joints_pos": np.array([joints_pos[-1-joints_obs_horizon*obs_real_horizon],joints_pos[-1]]),
            "joints_vel": np.array([joints_vel[-1-joints_obs_horizon*obs_real_horizon],joints_vel[-1]]),
            "camera_img": np.array([camera_imgs[-1-img_obs_horizon*obs_real_horizon],camera_imgs[-1]])
        }
        obs = joints2eepos(temp_obs, diff_ik_solver)

        # image_to_show = camera_imgs[-1][:, :, ::-1]
        image_to_show = (camera_imgs[-1]).astype(np.uint8) 
        

        with torch.no_grad():
            if classifier_activated and not using_keyboard:
                print("history_count",history_count)
                if history_count < 0:
                    current_gripper_pos = obs["robot0_gripper_width"][-1]
                    classifier_img_input = np.array(obs["camera0_rgb"][-1]).copy() / 255.0  # (H, W, C)
                    # classifier_img_input = np.array(obs["camera0_rgb"][-1])[:, :, ::-1].copy() # (H, W, C)
                    classifier_img_input = np.transpose(classifier_img_input, (2, 0, 1))  # (C, H, W)
                    classifier_img_input = np.expand_dims(classifier_img_input, axis=0)   # (1, C, H, W)
                    classifier_img_input = torch.from_numpy(classifier_img_input).float().to(device)
                    current_gripper_pos = np.array([0]) if current_action_gripper > 0.65 else np.array([1])
                    print(current_gripper_pos)
                    low_dim_tensor = torch.from_numpy(current_gripper_pos).float().view(1, 1).to(device)
                    print(low_dim_tensor)
                    logits = classifier_model(classifier_img_input.float().to(device), low_dim_tensor)
                    predictions = torch.argmax(logits, dim=1)

                    if predictions.item() == 1:
                        print("find key frame!!!!!!!!!!!!!!!!!!!!!!!!!")
                        cam_data = np.array(obs["camera0_rgb"][-1])
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
            

            if using_keyboard:
                index_key = cv2.waitKey(1) & 0xFF
                # print(index_key)
                if index_key in [ord('0'), ord('1'), ord('2'), ord('3')]:
                    image_number = int(chr(index_key))
                    cam_data = load_image(image_number)

                    if cam_data is not None:
                        if Memory_size >= Memory_limit:
                            memory_array = np.roll(memory_array, shift=-1, axis=0)
                            memory_array[-1] = np.transpose(cam_data, (2, 0, 1)) / 255.0
                        else:
                            memory_array[Memory_size] = np.transpose(cam_data, (2, 0, 1)) / 255.0
                            Memory_size += 1
                        print(f"Image {image_number} loaded into memory. Total: {Memory_size}")
                    else:
                        print("Image not found.")
                elif index_key != 255:
                    print("Invalid key. Please press 0/1/2/3.")

            obs_dict_np = get_real_umi_obs_dict(
                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                obs_pose_repr=obs_pose_rep,
                episode_start_pose=episode_start_pose)
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            
            mem_dict = {'memory_marker': memory_array}
            mem_dict = dict_apply(
                mem_dict,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
            )
            memory_obs = {
                'obs': obs_dict, 
                'memory': mem_dict,
            }
            cv2.imshow("image", image_to_show)
            cv2.waitKey(1) 
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
            # print('Inference latency:', time.time() - s)
            # 添加时间辍
            action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
            exec_time = time.time()
            for i in range(0,10):
                exec_one_time = time.time()
                if action_timestamps[i] < exec_time:
                    # print(f"out of time: index {i}")
                    continue
                current_action_gripper = action[i][6]
                current_action_pos = action[i][:6]
                xyz = current_action_pos[:3]
                rx, ry, rz = current_action_pos[3:]
                q = diff_ik_solver.solve_ik(xyz, [rx, ry, rz])
                cmd = np.zeros(7)
                cmd[0] = current_action_gripper * 100
                cmd[1:] = q
                cmd_pub.send('arm_action', msg={'action': cmd.tolist()})
                if i != 10:
                    time.sleep(0.05)
                # time.sleep(0.2 - (time.time() - exec_one_time))
            
            # time.sleep(2)
            # print('Exec time:', time.time() - exec_time)

if __name__ == '__main__':

    config = {
        'device': 'cuda',
        'diffusion_ckpt_path': './data/outputs/ckpt/diffusion/0411_coffee_latest.ckpt',
        'using_transformer': True,
        # driver 
        # 'start_pose': [100,  -0.002, -1.642, 2.366,  -0.164, 0.660,  0.091],
        # coffee
        'start_pose': [100,  -0.067, -1.732, 2.266,  0.014, 0.552,  0.056],
        'dataset_path': './dataset/0425_0427_baishi_shiwai_all.zarr',
        'classifier_activated': True,
        'classifier_type':'vit_small_lowdim', # 'vit_big_lowdim'
        'classifier_ckpt_path': "./data/outputs/ckpt/classifier/0422_driver_box_heightline_smalllowdim_cp10_epoch_300.pth",
        'classifier_dataparallel': False,
        'classifier_gapped_history_count': 10,
        'using_keyboard': False
    }
    # eval_on_dataset(config)
    run(config)
