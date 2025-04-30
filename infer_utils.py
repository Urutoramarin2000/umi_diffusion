from typing import Dict, List
import numpy as np
import base64
import io
import time
import queue
from threading import Thread
from multiprocessing import Queue
from cowautil.msg_communicator import MsgSubscriber, MsgPublisher
from umi.common.pose_util import pose_to_mat, mat_to_pose, mat_to_pose10d, pose10d_to_mat

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.cv2_util import get_image_transform

def get_real_umi_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        obs_pose_repr: str='abs',
        tx_robot1_robot0: np.ndarray=None,
        episode_start_pose: List[np.ndarray]=None,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=True)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim' and ('eef' not in key):
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in

    pose_mat = pose_to_mat(np.concatenate([
        env_obs['robot0_eef_pos'],
        env_obs['robot0_eef_rot_axis_angle']
    ], axis=-1))

    obs_pose_mat = convert_pose_mat_rep(
        pose_mat, 
        base_pose_mat=pose_mat[-1],
        pose_rep=obs_pose_repr,
        backward=False)

    obs_pose = mat_to_pose10d(obs_pose_mat)
    obs_dict_np['robot0_eef_pos'] = obs_pose[...,:3]
    obs_dict_np['robot0_eef_rot_axis_angle'] = obs_pose[...,3:]


    if episode_start_pose is not None:

        pose_mat = pose_to_mat(np.concatenate([
            env_obs[f'robot0_eef_pos'],
            env_obs[f'robot0_eef_rot_axis_angle']
        ], axis=-1))

        start_pose = episode_start_pose[0]
        start_pose_mat = pose_to_mat(start_pose)
        rel_obs_pose_mat = convert_pose_mat_rep(
            pose_mat,
            base_pose_mat=start_pose_mat,
            pose_rep='relative',
            backward=False)

        rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
        # obs_dict_np[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
        obs_dict_np[f'robot0_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

    return obs_dict_np


def get_real_umi_action( #transform the action with respect to the obs last frame
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs'
    ):

    env_action = list()
    # convert pose to mat
    pose_mat = pose_to_mat(np.concatenate([
        env_obs[f'robot0_eef_pos'][-1],
        env_obs[f'robot0_eef_rot_axis_angle'][-1]
    ], axis=-1))

    start = 0
    action_pose10d = action[..., start:start+9]
    action_grip = action[..., start+9:start+10]
    action_pose_mat = pose10d_to_mat(action_pose10d)

    # solve relative action
    action_mat = convert_pose_mat_rep(
        action_pose_mat, 
        base_pose_mat=pose_mat,
        pose_rep=action_pose_repr,
        backward=True)

    # convert action to pose
    action_pose = mat_to_pose(action_mat)
    env_action.append(action_pose)
    env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action

class ArmMsg:
    def __init__(self):
        self.joints_pos = None
        self.joints_vel = None
        self.camera_img = None
        self.arm_state_sub = MsgSubscriber(ip='192.168.1.3', port=22222, topic='arm_state')
        self.camera_data_sub = MsgSubscriber(ip='192.168.1.3', port=22223, topic='camera_data')
        # self.camera_data_sub = RealSenseTools()

    def run_arm_state_sub(self, terminal):
        while not terminal.is_set():
            arm_state = self.arm_state_sub.recv()
            self.joints_pos = np.array(arm_state['joints_pos'], dtype=np.float32)
            self.joints_vel = np.array(arm_state['joints_vel'], dtype=np.float32)
        return
    # def run_camera_data_sub(self, terminal, Q: Queue):
    #     while not terminal.is_set():
    #         color_img, depth_img, depth_frame, depth_colormap = self.camera_data_sub.recv()
    #         Q.put({'joints_pos': self.joints_pos, 'joints_vel': self.joints_vel, 'camera_img': color_img, "depth_img": depth_img, 'obs_timestamp':time.time()})
    #     return

    def run_camera_data_sub(self, terminal, Q: Queue):
        import av
        rawData = io.BytesIO()
        cur_pos = 0
        can_decode_h264 = False
        while not terminal.is_set():
            msg = self.camera_data_sub.recv()
            # Decode h264
            t1 = time.perf_counter()
            camera_bytes = base64.b64decode(msg['camera_data'].encode('utf-8')) # encode() 将 str 转换为 bytes
            rawData.write(camera_bytes)
            rawData.seek(cur_pos)
            if cur_pos == 0:
                container = av.open(rawData, format='h264', mode='r')
                original_codec_ctx = container.streams.video[0].codec_context
                codec = av.codec.CodecContext.create(original_codec_ctx.name, 'r')
            cur_pos += len(camera_bytes)
            for packet in container.demux():
                if packet.size == 0:
                    continue
                can_decode_h264 = True if packet.is_keyframe else can_decode_h264
                if can_decode_h264:
                    frames = codec.decode(packet)
                    for frame in frames:
                        self.camera_img = frame.to_ndarray(format='bgr24')
                        Q.put({'joints_pos': self.joints_pos, 'joints_vel': self.joints_vel, 'camera_img': self.camera_img, 'obs_timestamp':time.time()})
        return
class ActionMsg:
    def action_state_pub(self, terminal, Q: Queue, cmd_pub):
        while not terminal.is_set():
            try:
                current_action = Q.get_nowait()
                current_time = time.time()
                for i in range (len(current_action['timestamp'])):
                    if current_action['timestamp'][i] <= current_time:
                        print('out of time')
                        continue
                    cmd = current_action['action'][i]
                    cmd_pub.send('arm_action', msg={'action': cmd.tolist()})
                    # print('Pred action:', cmd[:],'timestamp',current_action['timestamp'][i])
                    time.sleep(0.01)
            except queue.Empty:
                pass  # 队列为空时跳过
            except Exception as e:
                print(f"Error in action_state_pub: {e}")
                

def arm_state_recv(terminate, Q: Queue):
    arm_msg = ArmMsg()
    th1 = Thread(target=arm_msg.run_arm_state_sub, args=(terminate,))
    th2 = Thread(target=arm_msg.run_camera_data_sub, args=(terminate, Q))
    th1.start()
    th2.start()
    th1.join()
    th2.join()

def arm_state_pub(terminate, Q: Queue):
    action_msg = ActionMsg()    
    cmd_pub = MsgPublisher(port=22222)
    th3 = Thread(target=action_msg.action_state_pub, args=(terminate, Q, cmd_pub))
    th3.start()
    th3.join()
